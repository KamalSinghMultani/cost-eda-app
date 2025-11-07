# spot_eda_app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression  # NEW: simple ML

st.set_page_config(page_title="AWS EC2 Spot Pricing EDA", layout="wide")
st.title("📉 AWS EC2 Spot Pricing EDA (Cheapest Regions & Trends)")

# ---------------------------------
# Constants & helpers
# ---------------------------------
COLS = ["Timestamp", "InstanceType", "Platform", "AvailabilityZone", "Price"]


def _read_csv_flexible(src, *, has_upload_handle: bool) -> pd.DataFrame:
    """Read a CSV that may or may not have a header row."""
    try:
        df0 = pd.read_csv(src, header=0)
        if all(c in df0.columns for c in COLS):
            return df0[COLS]
        if len(df0.columns) >= 5:
            df0 = df0.iloc[:, :5]
            df0.columns = COLS
            return df0
        raise ValueError("Header attempt did not match expected columns.")
    except Exception:
        if has_upload_handle and hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        df1 = pd.read_csv(src, header=None, names=COLS)
        return df1


@st.cache_data(show_spinner=False)
def load_many_csv_from_folder(folder_path: str) -> pd.DataFrame:
    dfs = []
    for name in os.listdir(folder_path):
        if name.lower().endswith(".csv"):
            fp = os.path.join(folder_path, name)
            try:
                df = _read_csv_flexible(fp, has_upload_handle=False)
                df["Region"] = os.path.splitext(name)[0]
                dfs.append(df)
            except Exception:
                continue
    if not dfs:
        return pd.DataFrame(columns=COLS + ["Region"])
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_many_csv_uploaded(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        region = os.path.splitext(os.path.basename(f.name))[0]
        df = _read_csv_flexible(f, has_upload_handle=True)
        df["Region"] = region
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=COLS + ["Region"])
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce", downcast="float")
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce", utc=False)
    try:
        if pd.api.types.is_datetime64tz_dtype(out["Timestamp"]):
            out["Timestamp"] = out["Timestamp"].dt.tz_convert(None)
    except Exception:
        pass
    for c in ["InstanceType", "Platform", "AvailabilityZone", "Region"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    out = out.dropna(subset=["Price"])
    return out


def cheapest_region_for(df: pd.DataFrame, instance_type: str, platform: str = "Linux/UNIX"):
    sub = df[(df["InstanceType"] == instance_type) & (df["Platform"] == platform)]
    if sub.empty:
        return None
    avg = sub.groupby("Region")["Price"].mean().reset_index().sort_values("Price")
    row = avg.iloc[0]
    return {"InstanceType": instance_type, "Platform": platform,
            "Region": row["Region"], "AvgPrice": float(row["Price"])}


def daily_means(df: pd.DataFrame, instance_type: str, platform: str):
    sub = df[(df["InstanceType"] == instance_type) & (df["Platform"] == platform)].dropna(subset=["Timestamp"])
    if sub.empty:
        return pd.DataFrame(columns=["Region", "Timestamp", "Price"])
    sub = sub.set_index("Timestamp")
    daily = sub.groupby("Region").resample("D")["Price"].mean().dropna().reset_index()
    return daily


def slope_per_region(daily_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in daily_df["Region"].unique():
        r = daily_df[daily_df["Region"] == region].sort_values("Timestamp")
        if len(r) < 2:
            continue
        x = r["Timestamp"].map(pd.Timestamp.toordinal).to_numpy()
        y = r["Price"].to_numpy(dtype=float)
        try:
            m, _ = np.polyfit(x, y, 1)
        except Exception:
            m = np.nan
        rows.append({"Region": region, "Slope": m})
    return pd.DataFrame(rows).sort_values("Slope")


# --------- NEW: very small ML forecaster (LinearRegression) ----------
def fit_forecast_linear(df_daily_region: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
    """
    df_daily_region: rows for ONE region with columns [Timestamp, Price].
    Returns a dataframe with historical + predicted rows (column 'Type' = 'History'/'Forecast').
    """
    r = df_daily_region.sort_values("Timestamp").copy()
    if len(r) < 3:
        return pd.DataFrame(columns=["Timestamp", "Price", "Type"])

    # Encode dates as ordinal integers
    X = r["Timestamp"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    y = r["Price"].astype(float).to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    last_date = r["Timestamp"].max().normalize()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")
    Xf = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    yf = model.predict(Xf)

    hist = r[["Timestamp", "Price"]].copy()
    hist["Type"] = "History"
    fut = pd.DataFrame({"Timestamp": future_dates, "Price": yf, "Type": "Forecast"})
    return pd.concat([hist, fut], ignore_index=True)


# ---------------------------------
# Data input
# ---------------------------------
st.sidebar.header("📥 Data Input")

# Auto-pick the current user's Downloads\archive for local runs.
# On Streamlit Cloud this path won't exist, and the uploader will be used.
default_folder = str(Path.home() / "Downloads" / "archive")

folder_path = st.sidebar.text_input("Folder path with regional CSVs", value=default_folder, key="folder_path")

uploaded = st.sidebar.file_uploader(
    "…or upload multiple regional CSVs (filenames like us-east-1.csv)",
    type=["csv"],
    accept_multiple_files=True,
    key="uploader",
)

if uploaded:
    raw = load_many_csv_uploaded(uploaded)
elif folder_path and os.path.isdir(folder_path):
    raw = load_many_csv_from_folder(folder_path)
else:
    st.warning("Provide a valid folder path or upload CSV files.")
    st.stop()

data = prep(raw)
if len(data) == 0:
    st.warning("No rows loaded. Check CSVs, then try again.")
    st.stop()

st.caption(f"Loaded **{len(data):,}** rows from **{data['Region'].nunique()}** regions.")
with st.expander("Preview raw data"):
    st.dataframe(data.head(20), use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💸 Cheapest Regions", "📈 Trends & Slopes", "🤖 Forecast"])

# ------------------------------
# TAB 1 — Cheapest Regions
# ------------------------------
with tab1:
    st.subheader("Find Cheapest Region(s) by Instance Type")
    platforms = sorted([p for p in data["Platform"].dropna().unique()])
    instances = sorted([i for i in data["InstanceType"].dropna().unique()])

    if not platforms or not instances:
        st.info("Data is missing Platform or InstanceType values.")
    else:
        c1, c2 = st.columns(2)
        platform = c1.selectbox("Platform", platforms, index=0, key="platform_cheapest")
        mode = c2.radio("Mode", ["Single instance", "Batch (multi-select)"], horizontal=True, key="mode_cheapest")

        if mode == "Single instance":
            itype = st.selectbox("Instance type", instances, index=0, key="itype_cheapest_single")
            result = cheapest_region_for(data, itype, platform)
            if result:
                st.success(
                    f"**Cheapest** region for **{itype}** on **{platform}** is "
                    f"**{result['Region']}** at average spot price **${result['AvgPrice']:.4f}**"
                )
                sub = data[(data["InstanceType"] == itype) & (data["Platform"] == platform)]
                per_region = sub.groupby("Region")["Price"].mean().reset_index().sort_values("Price")
                st.dataframe(per_region.style.format({"Price": "${:.4f}"}), use_container_width=True)
            else:
                st.warning("No rows found for that selection.")
        else:
            chosen = st.multiselect("Instance types", instances, default=instances[:5] if len(instances) >= 5 else instances, key="itype_cheapest_batch")
            rows = []
            for it in chosen:
                r = cheapest_region_for(data, it, platform)
                if r: rows.append(r)
            if rows:
                df_out = pd.DataFrame(rows).sort_values("AvgPrice")
                st.dataframe(df_out.style.format({"AvgPrice": "${:.4f}"}), use_container_width=True)
                st.download_button(
                    "Download cheapest-region table (CSV)",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="cheapest_regions.csv",
                    mime="text/csv",
                    key="dl_cheapest_csv",
                )
            else:
                st.info("Pick one or more instance types to compare.")

# ------------------------------
# TAB 2 — Trends & Slopes
# ------------------------------
with tab2:
    st.subheader("Daily Price Trends by Region")
    instances2 = sorted([i for i in data["InstanceType"].dropna().unique()])
    platforms2 = sorted([p for p in data["Platform"].dropna().unique()])

    if not instances2 or not platforms2:
        st.info("Data is missing Platform or InstanceType values.")
    else:
        c1, c2 = st.columns(2)
        it_for_trend = c1.selectbox("Instance type", instances2, key="itype_trend")
        plat_for_trend = c2.selectbox("Platform", platforms2, key="platform_trend")
        daily = daily_means(data, it_for_trend, plat_for_trend)

        if daily.empty:
            st.warning("No time series available for that selection.")
        else:
            fig = px.line(
                daily, x="Timestamp", y="Price", color="Region",
                title=f"Daily Mean Spot Price — {it_for_trend} on {plat_for_trend}",
                markers=False
            )
            st.plotly_chart(fig, use_container_width=True)

            slopes = slope_per_region(daily)
            if not slopes.empty:
                slopes["Trend"] = np.where(slopes["Slope"] > 0, "Increasing", "Decreasing")
                st.markdown("**Regions sorted by regression slope (price trend):**")
                st.dataframe(slopes.sort_values("Slope").style.format({"Slope": "{:.6f}"}), use_container_width=True)
                st.info(
                    f"Decreasing-price regions: **{(slopes['Slope'] < 0).sum()}** • "
                    f"Increasing-price regions: **{(slopes['Slope'] > 0).sum()}**"
                )
            else:
                st.info("Not enough data per region to fit trends.")

# ------------------------------
# TAB 3 — Forecast (ML)
# ------------------------------
with tab3:
    st.subheader("7-Day Price Forecast (Linear Regression)")

    instances3 = sorted([i for i in data["InstanceType"].dropna().unique()])
    platforms3 = sorted([p for p in data["Platform"].dropna().unique()])
    if not instances3 or not platforms3:
        st.info("Data is missing Platform or InstanceType values.")
    else:
        c1, c2, c3 = st.columns(3)
        it_for_fc = c1.selectbox("Instance type", instances3, key="itype_fc")
        plat_for_fc = c2.selectbox("Platform", platforms3, key="platform_fc")

        daily_all = daily_means(data, it_for_fc, plat_for_fc)
        if daily_all.empty:
            st.warning("No time series available for that selection.")
        else:
            regions = sorted(daily_all["Region"].unique().tolist())
            region_fc = c3.selectbox("Region", regions, key="region_fc")

            r = daily_all[daily_all["Region"] == region_fc][["Timestamp", "Price"]]
            out = fit_forecast_linear(r, days_ahead=7)
            if out.empty:
                st.info("Not enough data to fit the model (need >= 3 days).")
            else:
                fig2 = px.line(out, x="Timestamp", y="Price", color="Type",
                               title=f"{region_fc} • {it_for_fc} on {plat_for_fc} — 7-Day Forecast",
                               markers=True)
                st.plotly_chart(fig2, use_container_width=True)

                last_hist = out[out["Type"] == "History"]["Price"].iloc[-1]
                last_fore = out[out["Type"] == "Forecast"]["Price"].iloc[-1]
                delta = last_fore - last_hist
                st.metric("Predicted change by day 7", f"{delta:+.4f}")
                st.download_button(
                    "Download forecast CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{region_fc}_{it_for_fc}_{plat_for_fc}.csv",
                    mime="text/csv",
                    key="dl_forecast_csv",
                )

st.caption("Tip: Filenames should encode the region (e.g., `us-east-1.csv`, `eu-west-1.csv`).")
