# Dashboard.py  — EC2 & S3 EDA + ML
# (adds a new "🤖 ML" tab to train/predict CostUSD for EC2 and S3)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# NEW: simple ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="EC2 & S3 EDA + ML", layout="wide")
st.title("🖥️ EC2 & 🪣 S3 Exploratory Data Analysis (EDA + ML)")

# -------------------------------
# Local file paths (edit if needed)
# -------------------------------
EC2_PATH = r"C:\Users\kamal\Downloads\Week 9 - EDA\aws_resources_compute.csv"
S3_PATH  = r"C:\Users\kamal\Downloads\Week 9 - EDA\aws_resources_S3.csv"

# -------------------------------
# Upload widgets (cloud-friendly)
# -------------------------------
st.sidebar.markdown("### 📤 Upload data (optional)")
ec2_upload = st.sidebar.file_uploader("EC2 CSV (aws_resources_compute.csv)", type=["csv"], key="ec2_up")
s3_upload  = st.sidebar.file_uploader("S3 CSV (aws_resources_S3.csv)", type=["csv"], key="s3_up")


# -------------------------------
# Load helpers
# -------------------------------
def load_local_csv(path):
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    for maybe_date in ["CreationDate"]:
        if maybe_date in df.columns:
            df[maybe_date] = pd.to_datetime(df[maybe_date], errors="coerce")
    return df

ec2 = load_local_csv(EC2_PATH)
s3  = load_local_csv(S3_PATH)

# -------------------------------
# Basic cleaning
# -------------------------------
def fill_numeric_mean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df

def add_outlier_flag(df: pd.DataFrame, cost_col: str) -> pd.DataFrame:
    if df.empty or cost_col not in df.columns:
        return df
    df = df.copy()
    std = df[cost_col].std(skipna=True)
    if std and std != 0:
        df["Outlier"] = (np.abs((df[cost_col] - df[cost_col].mean()) / std) > 3)
    else:
        df["Outlier"] = False
    return df

ec2 = fill_numeric_mean(ec2)
s3  = fill_numeric_mean(s3)
ec2 = add_outlier_flag(ec2, "CostUSD")
s3  = add_outlier_flag(s3, "CostUSD")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Filters")
def safe_unique(df, col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

# EC2
ec2_regions = st.sidebar.multiselect("EC2 Regions", safe_unique(ec2, "Region"), default=safe_unique(ec2, "Region"))
ec2_types   = st.sidebar.multiselect("Instance Types", safe_unique(ec2, "InstanceType"), default=safe_unique(ec2, "InstanceType"))
ec2_states  = st.sidebar.multiselect("States", safe_unique(ec2, "State"), default=safe_unique(ec2, "State"))

# S3
s3_regions  = st.sidebar.multiselect("S3 Regions", safe_unique(s3, "Region"), default=safe_unique(s3, "Region"))
s3_classes  = st.sidebar.multiselect("Storage Classes", safe_unique(s3, "StorageClass"), default=safe_unique(s3, "StorageClass"))
s3_encrypt  = st.sidebar.multiselect("Encryption", safe_unique(s3, "Encryption"), default=safe_unique(s3, "Encryption"))

def apply_ec2_filters(df):
    if df.empty: return df
    m = pd.Series(True, index=df.index)
    if "Region" in df.columns and ec2_regions:       m &= df["Region"].isin(ec2_regions)
    if "InstanceType" in df.columns and ec2_types:   m &= df["InstanceType"].isin(ec2_types)
    if "State" in df.columns and ec2_states:         m &= df["State"].isin(ec2_states)
    return df[m].copy()

def apply_s3_filters(df):
    if df.empty: return df
    m = pd.Series(True, index=df.index)
    if "Region" in df.columns and s3_regions:        m &= df["Region"].isin(s3_regions)
    if "StorageClass" in df.columns and s3_classes:  m &= df["StorageClass"].isin(s3_classes)
    if "Encryption" in df.columns and s3_encrypt:    m &= df["Encryption"].isin(s3_encrypt)
    return df[m].copy()

ec2_f = apply_ec2_filters(ec2)
s3_f  = apply_s3_filters(s3)
st.caption(f"Filtered EC2 rows: **{len(ec2_f)}**  |  Filtered S3 rows: **{len(s3_f)}**")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🖥️ EC2 Analysis", "🪣 S3 Analysis", "⚙️ Optimization", "🤖 ML"])

STATE_COLORS = {
    "running":        "#2ECC71",
    "stopped":        "#E74C3C",
    "stopping":       "#E67E22",
    "terminated":     "#34495E",
    "pending":        "#F1C40F",
    "shutting-down":  "#9B59B6",
    "rebooting":      "#1ABC9C",
}

# ---------- EC2 TAB ----------
with tab1:
    st.subheader("📋 EC2 Overview")
    st.write(f"Shape: {ec2_f.shape}")
    if not ec2_f.empty:
        st.dataframe(ec2_f.head(), use_container_width=True)

    with st.expander("EC2 Info / Summary / Missing"):
        if not ec2_f.empty:
            st.write(ec2_f.describe(include="all"))
            st.write("Missing values per column:")
            st.write(ec2_f.isna().sum())

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Histogram: CPU Utilization**")
        if "CPUUtilization" in ec2_f.columns:
            color_arg = ec2_f["State"].astype(str) if "State" in ec2_f.columns else None
            fig = px.histogram(
                ec2_f, x="CPUUtilization", nbins=20, color=color_arg,
                title="CPU Utilization Histogram",
                color_discrete_map=STATE_COLORS if "State" in ec2_f.columns else None
            )
            fig.update_traces(marker_line_width=0.5, marker_line_color="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'CPUUtilization' not found in EC2 file.")

    with c2:
        st.write("**Scatter: CPU vs Cost (colored by State)**")
        if {"CPUUtilization","CostUSD"}.issubset(ec2_f.columns):
            color_arg = ec2_f["State"].astype(str) if "State" in ec2_f.columns else None
            fig = px.scatter(
                ec2_f, x="CPUUtilization", y="CostUSD",
                color=color_arg,
                hover_data=[c for c in ["ResourceId","InstanceType","Region","State"] if c in ec2_f.columns],
                title="CPU Utilization vs CostUSD",
                color_discrete_map=STATE_COLORS if "State" in ec2_f.columns else None
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=0.8, color="white")))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns 'CPUUtilization' and/or 'CostUSD' not found.")

    st.write("### 🧮 Average EC2 Cost per Region (CostUSD)")
    if {"Region","CostUSD"}.issubset(ec2_f.columns):
        ec2_cost_reg = ec2_f.groupby("Region", dropna=False)["CostUSD"].mean().reset_index()
        st.dataframe(ec2_cost_reg, use_container_width=True)
        st.plotly_chart(px.bar(ec2_cost_reg, x="Region", y="CostUSD", title="Avg Cost per Region"),
                        use_container_width=True)

    st.write("### 🏆 Top 5 Most Expensive EC2 Instances")
    if "CostUSD" in ec2_f.columns:
        cols = [c for c in ["ResourceId","Region","InstanceType","State","CPUUtilization","CostUSD"] if c in ec2_f.columns]
        st.dataframe(ec2_f.sort_values("CostUSD", ascending=False).head(5)[cols], use_container_width=True)

    st.download_button("📥 Download Filtered EC2", ec2_f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_ec2.csv", mime="text/csv")

# ---------- S3 TAB ----------
with tab2:
    st.subheader("📋 S3 Overview")
    st.write(f"Shape: {s3_f.shape}")
    if not s3_f.empty:
        st.dataframe(s3_f.head(), use_container_width=True)

    with st.expander("S3 Info / Summary / Missing"):
        if not s3_f.empty:
            st.write(s3_f.describe(include="all"))
            st.write("Missing values per column:")
            st.write(s3_f.isna().sum())

    d1, d2 = st.columns(2)
    with d1:
        st.write("**Bar: Total Storage by Region (GB)**")
        if {"Region","TotalSizeGB"}.issubset(s3_f.columns):
            s3_storage_reg = s3_f.groupby("Region", dropna=False)["TotalSizeGB"].sum().reset_index()
            st.dataframe(s3_storage_reg, use_container_width=True)
            st.plotly_chart(px.bar(s3_storage_reg, x="Region", y="TotalSizeGB", title="Total Storage per Region"),
                            use_container_width=True)
        else:
            st.info("Columns 'Region' and/or 'TotalSizeGB' not found in S3 file.")

    with d2:
        st.write("**Scatter: Cost vs Storage**")
        if {"TotalSizeGB","CostUSD"}.issubset(s3_f.columns):
            fig = px.scatter(
                s3_f, x="TotalSizeGB", y="CostUSD",
                color=s3_f["StorageClass"].astype(str) if "StorageClass" in s3_f.columns else None,
                hover_data=[c for c in ["BucketName","Region","StorageClass"] if c in s3_f.columns],
                title="Monthly Cost (CostUSD) vs Storage Size (GB)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns 'TotalSizeGB' and/or 'CostUSD' not found in S3 file.")

    st.write("### 🏆 Top 5 Largest S3 Buckets")
    if "TotalSizeGB" in s3_f.columns:
        cols = [c for c in ["BucketName","Region","StorageClass","TotalSizeGB","CostUSD"] if c in s3_f.columns]
        st.dataframe(s3_f.sort_values("TotalSizeGB", ascending=False).head(5)[cols], use_container_width=True)

    st.download_button("📥 Download Filtered S3", s3_f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_s3.csv", mime="text/csv")

# ---------- Optimization TAB ----------
with tab3:
    st.subheader("⚙️ Data-Driven Optimization")

    # EC2: highlight low CPU & high CostUSD
    if {"CPUUtilization","CostUSD"}.issubset(ec2_f.columns) and len(ec2_f) > 0:
        avg_cpu  = ec2_f["CPUUtilization"].mean()
        avg_cost = ec2_f["CostUSD"].mean()
        st.write(f"**EC2 Averages** — CPU: {avg_cpu:.1f}% | CostUSD: ${avg_cost:.3f}")

        color_arg = ec2_f["State"].astype(str) if "State" in ec2_f.columns else None
        fig = px.scatter(
            ec2_f, x="CPUUtilization", y="CostUSD",
            color=color_arg,
            hover_data=[c for c in ["ResourceId","InstanceType","Region","State"] if c in ec2_f.columns],
            title="EC2 Optimization Quadrants (low-CPU / high-cost)",
            color_discrete_map=STATE_COLORS if "State" in ec2_f.columns else None
        )
        fig.add_vline(x=avg_cpu,  line_dash="dash", line_color="gray")
        fig.add_hline(y=avg_cost, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        ineff = ec2_f[(ec2_f["CPUUtilization"] < 40) & (ec2_f["CostUSD"] > avg_cost)]
        if not ineff.empty:
            st.write("**Instances to rightsize (CPU < 40% and CostUSD above average):**")
            show_cols = [c for c in ["ResourceId","Region","InstanceType","State","CPUUtilization","CostUSD"] if c in ineff.columns]
            st.dataframe(ineff.sort_values("CostUSD", ascending=False)[show_cols], use_container_width=True)

        st.markdown("- Consider **rightsizing** or moving to smaller instance families for under-utilized instances.")
        st.markdown("- For steady workloads, consider **Savings Plans / Reserved Instances**; for bursty, try **Spot**.")

    # S3: cost per GB
    if {"TotalSizeGB","CostUSD"}.issubset(s3_f.columns) and len(s3_f) > 0:
        s3_tmp = s3_f.copy()
        s3_tmp["CostPerGB"] = np.where(s3_tmp["TotalSizeGB"] > 0, s3_tmp["CostUSD"] / s3_tmp["TotalSizeGB"], np.nan)
        avg_cpg = s3_tmp["CostPerGB"].mean(skipna=True)
        st.write(f"**Avg S3 Cost/GB (filtered)**: ${avg_cpg:.3f}")

        fig2 = px.scatter(
            s3_tmp, x="TotalSizeGB", y="CostUSD",
            color="CostPerGB",
            hover_data=[c for c in ["BucketName","Region","StorageClass"] if c in s3_tmp.columns],
            title="S3 Cost Efficiency (color = Cost/GB)",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig2, use_container_width=True)

        pricey = s3_tmp[s3_tmp["CostPerGB"] > avg_cpg]
        if not pricey.empty:
            st.write("**Buckets with above-average Cost/GB:**")
            show = [c for c in ["BucketName","Region","StorageClass","TotalSizeGB","CostUSD","CostPerGB"] if c in pricey.columns]
            st.dataframe(pricey.sort_values("CostPerGB", ascending=False)[show], use_container_width=True)

        if "VersionEnabled" in s3_f.columns and s3_f["VersionEnabled"].astype(str).str.lower().eq("true").any():
            st.markdown("- **VersionEnabled = true** detected → add **lifecycle rules** to clean old versions.")

# ---------- NEW: ML TAB ----------
with tab4:
    st.subheader("🤖 ML — Predict Monthly Cost (CostUSD)")

    # ---------- Helpers for ML ----------
    def make_design_matrix(df, y_col, num_cols, cat_cols):
        """Return X, y, colnames after cleaning/encoding."""
        if df.empty or y_col not in df.columns:
            return None, None, []
        use_cols = [c for c in num_cols if c in df.columns] + [c for c in cat_cols if c in df.columns] + [y_col]
        d = df[use_cols].dropna(subset=[y_col]).copy()
        # keep only rows with numeric features present
        for c in num_cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[c for c in num_cols if c in d.columns])
        if d.empty:
            return None, None, []

        X_num = d[[c for c in num_cols if c in d.columns]].copy()
        X_cat = pd.get_dummies(d[[c for c in cat_cols if c in d.columns]].astype(str), drop_first=True) if cat_cols else pd.DataFrame(index=d.index)
        X = pd.concat([X_num, X_cat], axis=1)
        y = d[y_col].astype(float)
        return X, y, X.columns.tolist()

    def train_and_report(X, y):
        model = LinearRegression()
        model.fit(X, y)
        y_hat = model.predict(X)
        r2  = r2_score(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        return model, r2, mae

    def what_if_inputs(cols_numeric, df_ref, label_prefix=""):
        """Build UI controls for numeric features based on quantiles of df_ref."""
        values = {}
        for c in cols_numeric:
            if c not in df_ref.columns: 
                continue
            low, q1, q3, high = df_ref[c].quantile([0.0, 0.25, 0.75, 1.0]).tolist()
            default = float(df_ref[c].median())
            values[c] = st.slider(f"{label_prefix}{c}", float(low), float(max(high, low+1e-9)), float(default))
        return values

    st.markdown("Train simple **Linear Regression** models on the filtered data and run a quick **what-if** prediction.")

    # ----- EC2 model -----
    st.markdown("### EC2 Cost Model")
    if not ec2_f.empty and "CostUSD" in ec2_f.columns:
        ec2_num = [c for c in ["CPUUtilization","MemoryUtilization","NetworkIn_Bps","NetworkOut_Bps"] if c in ec2_f.columns]
        ec2_cat = [c for c in ["Region","InstanceType","State"] if c in ec2_f.columns]
        X_ec2, y_ec2, cols_ec2 = make_design_matrix(ec2_f, "CostUSD", ec2_num, ec2_cat)

        if X_ec2 is None or X_ec2.empty:
            st.info("Not enough EC2 features to train a model. Need at least one of: CPUUtilization, MemoryUtilization, NetworkIn_Bps, NetworkOut_Bps.")
        else:
            model_ec2, r2_ec2, mae_ec2 = train_and_report(X_ec2, y_ec2)
            st.write(f"**Training metrics** — R²: `{r2_ec2:.3f}` | MAE: `${mae_ec2:.3f}`")

            # Coefficients table
            coef_df = pd.DataFrame({"Feature": cols_ec2, "Coefficient": model_ec2.coef_.astype(float)}).sort_values("Coefficient", ascending=False)
            st.dataframe(coef_df.head(12), use_container_width=True)

            st.markdown("**What-if prediction**")
            # Build input row
            num_vals = what_if_inputs(ec2_num, ec2_f, "EC2 • ")
            # categories as selects (use first 1: drop_first in dummies)
            cat_vals = {}
            for c in ec2_cat:
                if c in ec2_f.columns:
                    cat_vals[c] = st.selectbox(f"EC2 • {c}", safe_unique(ec2_f, c), index=0, key=f"ec2_cat_{c}")

            # Make a single-row dataframe and align to training columns
            x_user = {}
            for c in ec2_num:
                if c in num_vals: x_user[c] = [float(num_vals[c])]
            # dummy-encode categories
            cat_df = pd.DataFrame({k:[str(v)] for k,v in cat_vals.items()})
            if not cat_df.empty:
                cat_dummies = pd.get_dummies(cat_df.astype(str), drop_first=True)
            else:
                cat_dummies = pd.DataFrame()
            X_user = pd.concat([pd.DataFrame(x_user), cat_dummies], axis=1)
            # align to cols_ec2
            X_user = X_user.reindex(columns=cols_ec2, fill_value=0.0)
            pred = float(model_ec2.predict(X_user)[0])
            st.metric("Predicted EC2 CostUSD (monthly)", f"${pred:.3f}")

    else:
        st.info("EC2 filtered data has no CostUSD column.")

    st.divider()

    # ----- S3 model -----
    st.markdown("### S3 Cost Model")
    if not s3_f.empty and "CostUSD" in s3_f.columns:
        s3_num = [c for c in ["TotalSizeGB","ObjectCount"] if c in s3_f.columns]
        s3_cat = [c for c in ["Region","StorageClass","Encryption"] if c in s3_f.columns]
        X_s3, y_s3, cols_s3 = make_design_matrix(s3_f, "CostUSD", s3_num, s3_cat)

        if X_s3 is None or X_s3.empty:
            st.info("Not enough S3 features to train a model. Need at least one of: TotalSizeGB, ObjectCount.")
        else:
            model_s3, r2_s3, mae_s3 = train_and_report(X_s3, y_s3)
            st.write(f"**Training metrics** — R²: `{r2_s3:.3f}` | MAE: `${mae_s3:.3f}`")

            coef_df2 = pd.DataFrame({"Feature": cols_s3, "Coefficient": model_s3.coef_.astype(float)}).sort_values("Coefficient", ascending=False)
            st.dataframe(coef_df2.head(12), use_container_width=True)

            st.markdown("**What-if prediction**")
            num_vals2 = what_if_inputs(s3_num, s3_f, "S3 • ")
            cat_vals2 = {}
            for c in s3_cat:
                if c in s3_f.columns:
                    cat_vals2[c] = st.selectbox(f"S3 • {c}", safe_unique(s3_f, c), index=0, key=f"s3_cat_{c}")

            x_user2 = {}
            for c in s3_num:
                if c in num_vals2: x_user2[c] = [float(num_vals2[c])]
            cat_df2 = pd.DataFrame({k:[str(v)] for k,v in cat_vals2.items()})
            cat_dummies2 = pd.get_dummies(cat_df2.astype(str), drop_first=True) if not cat_df2.empty else pd.DataFrame()
            X_user2 = pd.concat([pd.DataFrame(x_user2), cat_dummies2], axis=1).reindex(columns=cols_s3, fill_value=0.0)
            pred2 = float(model_s3.predict(X_user2)[0])
            st.metric("Predicted S3 CostUSD (monthly)", f"${pred2:.3f}")

    else:
        st.info("S3 filtered data has no CostUSD column.")

st.success("✅ EDA + ML complete. Train models on your filtered data and do quick what-if predictions.")
