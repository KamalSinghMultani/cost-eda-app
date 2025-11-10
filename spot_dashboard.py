# spot_eda_app.py
import os
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AWS EC2 Spot Pricing EDA", layout="wide")
st.title("📉 AWS EC2 Spot Pricing EDA (Cheapest Regions & Trends & ZIP Upload)")
st.caption("Upload CSVs directly or a ZIP containing CSVs (sub-folders OK). Filenames should encode region, e.g., us-east-1.csv")

# -------------------------------------------------
# Constants & helpers
# -------------------------------------------------
COLS = ["Timestamp", "InstanceType", "Platform", "AvailabilityZone", "Price"]


def _read_csv_flexible(src, *, has_upload_handle: bool) -> pd.DataFrame:
    """
    Read a CSV that may or may not have a header row.
    1) Try header=0 and see if we have the expected columns
    2) Otherwise coerce first 5 cols and rename to COLS
    3) If that fails, try header=None with COLS
    """
    try:
        df0 = pd.read_csv(src, header=0, on_bad_lines="skip", low_memory=False)
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
        df1 = pd.read_csv(
            src, header=None, names=COLS, on_bad_lines="skip", low_memory=False
        )
        return df1


@st.cache_data(show_spinner=False)
def load_many_csv_from_folder(folder_path: str) -> pd.DataFrame:
    """Load all *.csv in a folder; region is derived from filename."""
    dfs = []
    try:
        for name in os.listdir(folder_path):
            if name.lower().endswith(".csv"):
                fp = os.path.join(folder_path, name)
                try:
                    df = _read_csv_flexible(fp, has_upload_handle=False)
                    df["Region"] = os.path.splitext(name)[0]
                    dfs.append(df)
                except Exception:
                    continue
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame(columns=COLS + ["Region"])
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_many_csv_from_zip_bytes(zip_bytes: bytes):
    """
    Read all CSVs anywhere inside a .zip (subfolders OK).
    Returns (df, extracted_list, samples) where:
      - df: concatenated dataframe
      - extracted_list: human-readable list of files handled/skipped
      - samples: list of (inner_path, sample_dataframe_head)
    """
    extracted = []
    samples = []
    dfs = []
    try:
        with ZipFile(BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not info.filename.lower().endswith(".csv"):
                    continue

                with zf.open(info) as fh:
                    # robust read
                    try:
                        df = pd.read_csv(
                            fh, header=0, on_bad_lines="skip", low_memory=False
                        )
                    except Exception:
                        fh.seek(0)
                        df = pd.read_csv(
                            fh, header=None, names=COLS,
                            on_bad_lines="skip", low_memory=False
                        )

                    # coerce to expected shape
                    if all(c in df.columns for c in COLS):
                        df = df[COLS]
                    else:
                        if len(df.columns) >= 5:
                            df = df.iloc[:, :5]
                            df.columns = COLS
                        else:
                            extracted.append(f"{info.filename}  (skipped: wrong shape)")
                            continue

                    region = os.path.splitext(os.path.basename(info.filename))[0]
                    df["Region"] = region
                    dfs.append(df)

                    # sample head for validator
                    samples.append((info.filename, df.head(5)))

                extracted.append(info.filename)

    except BadZipFile:
        return pd.DataFrame(columns=COLS + ["Region"]), ["(error) Not a valid ZIP"], []
    except Exception as e:
        return pd.DataFrame(columns=COLS + ["Region"]), [f"(error) {e}"], []

    if not dfs:
        return (
            pd.DataFrame(columns=COLS + ["Region"]),
            extracted if extracted else ["(no CSV files found in ZIP)"],
            samples,
        )
    return pd.concat(dfs, ignore_index=True), extracted, samples


@st.cache_data(show_spinner=False)
def load_many_csv_uploaded(files) -> pd.DataFrame:
    """Load multiple uploaded CSVs (browser upload)."""
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
    """Light cleaning + downcasting to reduce memory."""
    out = df.copy()

    # numeric
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce", downcast="float")

    # datetime (allow tz-naive & tz-aware, then strip TZ)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce", utc=False)
    try:
        if pd.api.types.is_datetime64tz_dtype(out["Timestamp"]):
            out["Timestamp"] = out["Timestamp"].dt.tz_convert(None)
    except Exception:
        pass

    # normalize strings
    for c in ["InstanceType", "Platform", "AvailabilityZone", "Region"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # drop rows with no price
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


def fit_forecast_linear(df_daily_region: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
    """Very small ML forecaster (LinearRegression) for one region's daily series."""
    r = df_daily_region.sort_values("Timestamp").copy()
    if len(r) < 3:
        return pd.DataFrame(columns=["Timestamp", "Price", "Type"])

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


# -------------------------------------------------
# Data input
# -------------------------------------------------
st.sidebar.header("📥 Data Input")

# Allow either local folder OR uploads (CSVs / ZIPs)
default_folder = str(Path.home() / "Downloads" / "archive")
use_local = st.sidebar.toggle("Use local folder", value=os.path.isdir(default_folder))
folder_path = st.sidebar.text_input("Folder path with regional CSVs", value=default_folder)

uploads = st.sidebar.file_uploader(
    "…or upload CSV(s) and/or ZIP(s) with CSVs inside",
    type=["csv", "zip"],
    accept_multiple_files=True,
)

raw = pd.DataFrame(columns=COLS + ["Region"])
zip_findings = []
zip_samples = []  # list[(inner_path, df.head())]

if use_local:
    if folder_path and os.path.isdir(folder_path):
        raw = load_many_csv_from_folder(folder_path)
    else:
        st.warning("Folder path not found. Correct it or uncheck 'Use local folder'.")
else:
    if uploads:
        dfs = []
        for f in uploads:
            name = f.name.lower()
            try:
                if name.endswith(".zip"):
                    zbytes = f.getvalue()  # read bytes BEFORE cached fn
                    df_zip, found, samples = load_many_csv_from_zip_bytes(zbytes)
                    zip_findings.extend([f"ZIP {f.name} → {item}" for item in found])
                    zip_samples.extend([(f"{f.name} :: {p}", dfh) for (p, dfh) in samples])
                    if not df_zip.empty:
                        dfs.append(df_zip)
                else:
                    df_csv = _read_csv_flexible(f, has_upload_handle=True)
                    region = os.path.splitext(os.path.basename(f.name))[0]
                    df_csv["Region"] = region
                    dfs.append(df_csv)
            except Exception as e:
                st.error(f"{f.name}: {e}")

        if dfs:
            raw = pd.concat(dfs, ignore_index=True)

# ZIP import report (never halts the app)
if zip_findings:
    with st.expander("ZIP import details"):
        for line in zip_findings:
            st.write("•", line)
    if st.toggle("Show quick ZIP validator (first 5 rows per inner CSV)", value=False):
        for label, dfh in zip_samples[:12]:  # cap for UI sanity
            st.markdown(f"**{label}**")
            st.dataframe(dfh, use_container_width=True)

if raw.empty:
    st.error("No usable data loaded. Upload individual CSVs or a ZIP with CSV files inside, or point to a folder with CSVs.")
    st.stop()

data = prep(raw)
if len(data) == 0:
    st.error("No rows after cleaning. Check that CSVs have at least these columns: Timestamp, InstanceType, Platform, AvailabilityZone, Price.")
    st.stop()

st.caption(f"Loaded **{len(data):,}** rows from **{data['Region'].nunique()}** regions.")
with st.expander("Preview raw data"):
    st.dataframe(data.head(20), use_container_width=True)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
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
            chosen = st.multiselect(
                "Instance types",
                instances,
                default=instances[:5] if len(instances) >= 5 else instances,
                key="itype_cheapest_batch",
            )
            rows = []
            for it in chosen:
                r = cheapest_region_for(data, it, platform)
                if r:
                    rows.append(r)
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
                st.info("Not enough data to fit the model (need ≥ 3 days).")
            else:
                fig2 = px.line(
                    out, x="Timestamp", y="Price", color="Type",
                    title=f"{region_fc} • {it_for_fc} on {plat_for_fc} — 7-Day Forecast",
                    markers=True
                )
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

st.caption("Tip: If using ZIPs, make sure they actually contain CSV files (not Excel) — sub-folders are OK.")
