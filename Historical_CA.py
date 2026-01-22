import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, time
from io import BytesIO

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="CA Historical & OSPH Analysis",
    layout="wide"
)

st.title("📊 CA Historical & OSPH Analysis Dashboard")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data_ca.csv")
    df["Initiation"] = pd.to_datetime(df["Initiation"])
    df["RealisasiDate"] = pd.to_datetime(df["RealisasiDate"])
    return df

df = load_data()

# ======================
# SIDEBAR FILTER
# ======================
st.sidebar.header("Filter")

produk = st.sidebar.multiselect(
    "Produk", sorted(df["Produk"].dropna().unique())
)

branch = st.sidebar.multiselect(
    "Branch", sorted(df["branch_name"].dropna().unique())
)

status = st.sidebar.multiselect(
    "Status Aplikasi", sorted(df["desc_status_apps"].dropna().unique())
)

df_f = df.copy()
if produk:
    df_f = df_f[df_f["Produk"].isin(produk)]
if branch:
    df_f = df_f[df_f["branch_name"].isin(branch)]
if status:
    df_f = df_f[df_f["desc_status_apps"].isin(status)]

# ======================
# OSPH RANGE (EXCEL STYLE)
# ======================
bins = [0, 250_000_000, 500_000_000, 10_000_000_000]
labels = ["0 - 250 Juta", "250 - 500 Juta", "500 Juta+"]

df_f["Range_Harga"] = pd.cut(
    df_f["Outstanding_PH"],
    bins=bins,
    labels=labels,
    right=True
)

# ======================
# KPI SUMMARY
# ======================
st.subheader("📌 KPI Summary")

col1, col2, col3, col4 = st.columns(4)

total_app = df_f["apps_id"].nunique()
approve_rate = (df_f["desc_status_apps"] == "APPROVE").mean()
reject_rate = (df_f["desc_status_apps"] == "REJECT").mean()

col1.metric("Total AppID", total_app)
col2.metric("Approve Rate", f"{approve_rate:.1%}")
col3.metric("Reject Rate", f"{reject_rate:.1%}")
col4.metric("Total Records", len(df_f))

# ======================
# SECTION A – OSPH vs STATUS
# ======================
st.subheader("A. OSPH vs Status Aplikasi")

total_apps_all = df_f["apps_id"].nunique()

summary_status = (
    df_f.groupby(["Range_Harga", "desc_status_apps"])
    .agg(
        total_apps_id=("apps_id", "nunique"),
        total_records=("apps_id", "count")
    )
    .reset_index()
)

pivot_status = summary_status.pivot_table(
    index="Range_Harga",
    columns="desc_status_apps",
    values="total_records",
    fill_value=0
)

pivot_status["Total_apps_id"] = (
    df_f.groupby("Range_Harga")["apps_id"].nunique()
)

pivot_status["% dari Total"] = (
    pivot_status["Total_apps_id"] / total_apps_all * 100
).round(1)

st.dataframe(pivot_status)

# ======================
# SECTION B – OSPH vs PEKERJAAN
# ======================
st.subheader("B. OSPH vs Pekerjaan")

pivot_job = pd.pivot_table(
    df_f,
    index="Range_Harga",
    columns="Pekerjaan",
    values="apps_id",
    aggfunc="nunique",
    fill_value=0
)

st.dataframe(pivot_job)

# ======================
# SECTION C – OSPH vs JENIS KENDARAAN
# ======================
st.subheader("C. OSPH vs Jenis Kendaraan")

pivot_vehicle = pd.pivot_table(
    df_f,
    index="Range_Harga",
    columns="JenisKendaraan",
    values="apps_id",
    aggfunc="nunique",
    fill_value=0
)

st.dataframe(pivot_vehicle)

# ======================
# SLA – BANK STYLE (2025)
# ======================
HOLIDAYS_2025 = [
    date(2025,1,1), date(2025,1,29), date(2025,3,29),
    date(2025,3,31), date(2025,4,1), date(2025,4,18),
    date(2025,5,1), date(2025,5,29), date(2025,6,1),
    date(2025,6,7), date(2025,8,17), date(2025,9,5),
    date(2025,12,25)
]

START_TIME = time(8,30)
END_TIME = time(15,30)

def is_working_day(d):
    return d.weekday() < 5 and d not in HOLIDAYS_2025

def calculate_sla(start, end):
    if start.time() >= END_TIME:
        start = start + pd.Timedelta(days=1)
        start = start.replace(hour=8, minute=30)

    sla = 0
    current = start.date()

    while current <= end.date():
        if is_working_day(current):
            sla += 1
        current += pd.Timedelta(days=1)

    return max(sla - 1, 0)

df_f["SLA_Days"] = df_f.apply(
    lambda x: calculate_sla(x["Initiation"], x["RealisasiDate"]),
    axis=1
)

# ======================
# SLA KPI
# ======================
st.subheader("📌 SLA Performance")

col1, col2 = st.columns(2)
col1.metric("Average SLA (Hari Kerja)", round(df_f["SLA_Days"].mean(), 2))
col2.metric("SLA ≤ 1 Hari (%)", f"{(df_f['SLA_Days']<=1).mean():.1%}")

st.bar_chart(df_f["SLA_Days"].value_counts().sort_index())

# ======================
# CA PERFORMANCE
# ======================
st.subheader("👤 CA Performance")

ca_perf = df_f.groupby("user_name").agg(
    total_app=("apps_id", "nunique"),
    avg_sla=("SLA_Days", "mean"),
    approve_rate=("desc_status_apps", lambda x: (x=="APPROVE").mean())
).reset_index()

st.dataframe(ca_perf)

# ======================
# TREND BULANAN
# ======================
st.subheader("📈 Trend Bulanan (Initiation)")

df_f["Month"] = df_f["Initiation"].dt.to_period("M").astype(str)
trend = df_f.groupby("Month")["apps_id"].nunique()

st.line_chart(trend)

# ======================
# FLAG LAYAK MASUK CA
# ======================
def flag_ca(row):
    if row["Outstanding_PH"] > 250_000_000:
        return "WAJIB CA"
    if row["Pekerjaan"] == "Wiraswasta":
        return "WAJIB CA"
    return "FAST TRACK"

df_f["Flag_CA"] = df_f.apply(flag_ca, axis=1)

st.subheader("🚦 Flag Masuk CA")
st.dataframe(df_f[["apps_id","Range_Harga","Pekerjaan","Flag_CA"]])

# ======================
# EXPORT TO EXCEL
# ======================
st.subheader("⬇️ Export Data")

output = BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    pivot_status.to_excel(writer, sheet_name="OSPH_Status")
    pivot_job.to_excel(writer, sheet_name="OSPH_Pekerjaan")
    pivot_vehicle.to_excel(writer, sheet_name="OSPH_Kendaraan")
    ca_perf.to_excel(writer, sheet_name="CA_Performance", index=False)
    df_f.to_excel(writer, sheet_name="Detail_Data", index=False)

st.download_button(
    label="Download Excel",
    data=output.getvalue(),
    file_name="CA_OSPH_Analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
