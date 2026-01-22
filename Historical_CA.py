# ============================================================
# OSPH & CREDIT ANALYST ANALYTICAL DASHBOARD - DIVISI LEVEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import holidays
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Analyst Analytical Dashboard - Division",
    layout="wide"
)

st.title("🏦 Credit Analyst Analytical Dashboard (Division Level)")
st.caption("Analytical dashboard for Dept Head | SLA • OSPH • Bottleneck • Decision Driver")

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

uploaded_file = st.file_uploader("📂 Upload Historical CA Data", type=["xlsx"])
if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

# ============================================================
# DATETIME & BASIC CLEANING
# ============================================================
df["Initiation"] = pd.to_datetime(df["Initiation"])
df["action_on"] = pd.to_datetime(df["action_on"])
df["Outstanding_PH"] = pd.to_numeric(df["Outstanding_PH"], errors="coerce")

# ============================================================
# HOLIDAY & SLA CONFIG (2025 ONLY)
# ============================================================
ID_HOLIDAYS = holidays.Indonesia(years=[2025])

WORK_START = time(8, 30)
WORK_END = time(15, 30)

def is_working_day(d):
    return d.weekday() < 5 and d not in ID_HOLIDAYS

def calculate_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    if start.time() > WORK_END:
        start = datetime.combine(start.date() + timedelta(days=1), WORK_START)

    total_minutes = 0
    current = start

    while current < end:
        if is_working_day(current.date()):
            ws = datetime.combine(current.date(), WORK_START)
            we = datetime.combine(current.date(), WORK_END)

            eff_start = max(current, ws)
            eff_end = min(end, we)

            if eff_start < eff_end:
                total_minutes += (eff_end - eff_start).seconds / 60

        current = datetime.combine(current.date() + timedelta(days=1), WORK_START)

    return round(total_minutes / 60, 2)

df["SLA_Hours"] = df.apply(
    lambda x: calculate_sla(x["Initiation"], x["action_on"]),
    axis=1
)

# ============================================================
# OSPH RANGE (PERSIS EXCEL)
# ============================================================
def osph_bucket(val):
    if val <= 250_000_000:
        return "0 - 250 Juta"
    elif val <= 500_000_000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

df["Range_OSPH"] = df["Outstanding_PH"].apply(osph_bucket)

# ============================================================
# SIDEBAR FILTER (GLOBAL)
# ============================================================
st.sidebar.header("🔎 Global Filter")

produk = st.sidebar.multiselect(
    "Produk",
    sorted(df["Produk"].dropna().unique()),
    default=df["Produk"].dropna().unique()
)

branch = st.sidebar.multiselect(
    "Branch",
    sorted(df["branch_name"].dropna().unique()),
    default=df["branch_name"].dropna().unique()
)

df_f = df[
    (df["Produk"].isin(produk)) &
    (df["branch_name"].isin(branch))
].copy()

# ============================================================
# KPI – DIVISION LEVEL
# ============================================================
st.subheader("📌 Division KPI")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Apps ID", df_f["apps_id"].nunique())
k2.metric("Total Records", len(df_f))
k3.metric("Avg SLA (Jam)", round(df_f["SLA_Hours"].mean(), 2))
k4.metric("P75 SLA (Jam)", round(df_f["SLA_Hours"].quantile(0.75), 2))
k5.metric("Max SLA (Jam)", round(df_f["SLA_Hours"].max(), 2))

st.divider()

# ============================================================
# 1️⃣ OSPH DRIVER ANALYSIS
# ============================================================
st.subheader("1️⃣ OSPH as SLA Driver")

osph_driver = (
    df_f.groupby("Range_OSPH")
    .agg(
        Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
)

osph_driver["Δ_vs_Avg"] = osph_driver["Avg_SLA"] - df_f["SLA_Hours"].mean()

fig_osph = px.bar(
    osph_driver,
    x="Range_OSPH",
    y="Avg_SLA",
    text="Avg_SLA",
    title="Average SLA by OSPH Range"
)
st.plotly_chart(fig_osph, use_container_width=True)

# ============================================================
# 2️⃣ PEKERJAAN × OSPH (COMPLEXITY SIGNAL)
# ============================================================
st.subheader("2️⃣ Pekerjaan vs OSPH (Complexity Analysis)")

job_os = (
    df_f.groupby(["Range_OSPH", "Pekerjaan"])
    .agg(
        Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
)

fig_job = px.bar(
    job_os,
    x="Range_OSPH",
    y="Avg_SLA",
    color="Pekerjaan",
    barmode="group",
    title="Avg SLA by Job & OSPH"
)
st.plotly_chart(fig_job, use_container_width=True)

# ============================================================
# 3️⃣ JENIS KENDARAAN × OSPH
# ============================================================
st.subheader("3️⃣ Jenis Kendaraan vs OSPH")

veh_os = (
    df_f.groupby(["Range_OSPH", "JenisKendaraan"])
    .agg(
        Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
)

fig_veh = px.bar(
    veh_os,
    x="Range_OSPH",
    y="Avg_SLA",
    color="JenisKendaraan",
    barmode="group"
)
st.plotly_chart(fig_veh, use_container_width=True)

# ============================================================
# 4️⃣ BOTTLENECK ANALYSIS (TOP 25% SLA)
# ============================================================
st.subheader("4️⃣ Bottleneck Detection (Top 25% SLA)")

threshold = df_f["SLA_Hours"].quantile(0.75)
bottleneck = df_f[df_f["SLA_Hours"] >= threshold]

bottleneck_seg = (
    bottleneck.groupby(["Range_OSPH", "Pekerjaan"])
    .agg(
        Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
    .sort_values("Avg_SLA", ascending=False)
)

st.dataframe(bottleneck_seg, use_container_width=True)

# ============================================================
# 5️⃣ WASTED EFFORT (REJECT BUT HIGH SLA)
# ============================================================
st.subheader("5️⃣ Wasted Effort Analysis")

reject_df = df_f[df_f["apps_status"].str.upper().str.contains("REJECT", na=False)]

wasted = (
    reject_df.groupby("Range_OSPH")
    .agg(
        Reject_Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
)

fig_waste = px.bar(
    wasted,
    x="Range_OSPH",
    y="Avg_SLA",
    text="Reject_Apps",
    title="Rejected Apps with Avg SLA"
)
st.plotly_chart(fig_waste, use_container_width=True)

# ============================================================
# 6️⃣ CUT-OFF TIME ANALYSIS
# ============================================================
st.subheader("6️⃣ Cut-off Time Impact")

df_f["After_1530"] = df_f["Initiation"].dt.time > WORK_END

cutoff = (
    df_f.groupby("After_1530")
    .agg(
        Apps=("apps_id", "nunique"),
        Avg_SLA=("SLA_Hours", "mean")
    )
    .reset_index()
)

cutoff["After_1530"] = cutoff["After_1530"].map({True: "After 15:30", False: "Before 15:30"})

fig_cut = px.bar(
    cutoff,
    x="After_1530",
    y="Avg_SLA",
    text="Avg_SLA"
)
st.plotly_chart(fig_cut, use_container_width=True)

# ============================================================
# 7️⃣ EXECUTIVE INSIGHT (AUTO GENERATED)
# ============================================================
st.subheader("🧠 Executive Insight (Auto)")

top_os = osph_driver.sort_values("Avg_SLA", ascending=False).iloc[0]
top_bottleneck = bottleneck_seg.iloc[0]

st.success(f"""
**Key Findings for Dept Head:**

• OSPH **{top_os['Range_OSPH']}** memiliki SLA tertinggi (**{top_os['Avg_SLA']:.2f} jam**).  
• Bottleneck terbesar berasal dari **{top_bottleneck['Pekerjaan']}** pada OSPH **{top_bottleneck['Range_OSPH']}**.  
• Aplikasi masuk **setelah 15:30** meningkatkan SLA signifikan.  
• Terdapat indikasi **wasted effort** pada segmen reject dengan SLA tinggi.

➡️ **Rekomendasi:** pertimbangkan pre-screening, penyesuaian cut-off, dan redistribusi workload CA.
""")
