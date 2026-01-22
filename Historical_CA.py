import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import holidays
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="OSPH Analysis - Divisi",
    layout="wide"
)

st.title("📊 OSPH & Credit Analyst Analysis Dashboard (Divisi)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df

uploaded_file = st.file_uploader("Upload Data Excel", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = load_data(uploaded_file)

# =========================
# PREPROCESSING
# =========================
df["Initiation"] = pd.to_datetime(df["Initiation"])
df["action_on"] = pd.to_datetime(df["action_on"])

# =========================
# HOLIDAY FUNCTION (2025)
# =========================
id_holidays = holidays.Indonesia(years=[2025])

def is_working_day(date):
    return date.weekday() < 5 and date not in id_holidays

def calculate_sla(start, end):
    if start.time() > time(15, 30):
        start = datetime.combine(start.date() + timedelta(days=1), time(8, 30))

    total_minutes = 0
    current = start

    while current < end:
        if is_working_day(current.date()):
            work_start = datetime.combine(current.date(), time(8, 30))
            work_end = datetime.combine(current.date(), time(15, 30))

            effective_start = max(current, work_start)
            effective_end = min(end, work_end)

            if effective_start < effective_end:
                total_minutes += (effective_end - effective_start).seconds / 60

        current = datetime.combine(current.date() + timedelta(days=1), time(8, 30))

    return round(total_minutes / 60, 2)

df["SLA_Hours"] = df.apply(
    lambda x: calculate_sla(x["Initiation"], x["action_on"]),
    axis=1
)

# =========================
# OSPH RANGE (EXCEL BASED)
# =========================
def osph_range(val):
    if val <= 250_000_000:
        return "0 - 250 Juta"
    elif val <= 500_000_000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

df["Range_Harga"] = df["Outstanding_PH"].apply(osph_range)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔎 Filter")

produk = st.sidebar.multiselect(
    "Produk",
    options=df["Produk"].unique(),
    default=df["Produk"].unique()
)

branch = st.sidebar.multiselect(
    "Branch",
    options=df["branch_name"].unique(),
    default=df["branch_name"].unique()
)

df_f = df[
    (df["Produk"].isin(produk)) &
    (df["branch_name"].isin(branch))
]

# =========================
# KPI HEADER
# =========================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Apps ID", df_f["apps_id"].nunique())
k2.metric("Total Records", len(df_f))
k3.metric("Avg SLA (Hours)", round(df_f["SLA_Hours"].mean(), 2))
k4.metric("Max SLA (Hours)", round(df_f["SLA_Hours"].max(), 2))

st.divider()

# =========================
# OSPH DISTRIBUTION
# =========================
st.subheader("Distribusi Apps berdasarkan Range OSPH")

osph_summary = (
    df_f.groupby("Range_Harga")
    .agg(
        Total_apps_id=("apps_id", "nunique"),
        Total_Records=("apps_id", "count")
    )
    .reset_index()
)

fig1 = px.bar(
    osph_summary,
    x="Range_Harga",
    y="Total_apps_id",
    text="Total_apps_id"
)
st.plotly_chart(fig1, use_container_width=True)

# =========================
# STATUS BREAKDOWN
# =========================
st.subheader("Breakdown Status Apps per Range OSPH")

status_os = (
    df_f.groupby(["Range_Harga", "apps_status"])
    .size()
    .reset_index(name="Total")
)

fig2 = px.bar(
    status_os,
    x="Range_Harga",
    y="Total",
    color="apps_status",
    barmode="stack"
)
st.plotly_chart(fig2, use_container_width=True)

# =========================
# PEKERJAAN VS OSPH
# =========================
st.subheader("Pekerjaan vs Range OSPH")

job_os = (
    df_f.groupby(["Range_Harga", "Pekerjaan"])
    .agg(Total=("apps_id", "nunique"))
    .reset_index()
)

fig3 = px.bar(
    job_os,
    x="Range_Harga",
    y="Total",
    color="Pekerjaan",
    barmode="stack"
)
st.plotly_chart(fig3, use_container_width=True)

# =========================
# JENIS KENDARAAN
# =========================
st.subheader("Jenis Kendaraan vs Range OSPH")

kendaraan_os = (
    df_f.groupby(["Range_Harga", "JenisKendaraan"])
    .agg(Total=("apps_id", "nunique"))
    .reset_index()
)

fig4 = px.bar(
    kendaraan_os,
    x="Range_Harga",
    y="Total",
    color="JenisKendaraan",
    barmode="stack"
)
st.plotly_chart(fig4, use_container_width=True)

# =========================
# SLA ANALYSIS
# =========================
st.subheader("SLA Distribution (Hours)")

fig5 = px.histogram(
    df_f,
    x="SLA_Hours",
    nbins=30
)
st.plotly_chart(fig5, use_container_width=True)

# =========================
# INSIGHT BOX
# =========================
st.subheader("📌 Insight Otomatis")

peak_range = osph_summary.sort_values("Total_apps_id", ascending=False).iloc[0]["Range_Harga"]

st.info(
    f"""
    - Range OSPH dengan volume tertinggi: **{peak_range}**
    - Rata-rata SLA: **{round(df_f['SLA_Hours'].mean(),2)} jam**
    - Data ini dapat digunakan untuk evaluasi **beban kerja CA & capacity planning divisi**
    """
)
