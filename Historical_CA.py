import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CA Historical Analytical Dashboard",
    layout="wide"
)

st.title("Credit Analyst Historical Dashboard")

# =========================================================
# LOAD DATA (LANGSUNG DATAFRAME)
# =========================================================
FILE_NAME = "DataHistoricalCA.xlsx"

if not Path(FILE_NAME).exists():
    st.error(f"❌ File '{FILE_NAME}' tidak ditemukan di folder app.py")
    st.stop()

df = pd.read_excel(FILE_NAME)

# =========================================================
# BASIC CLEANING
# =========================================================
date_cols = ["Initiation", "action_on", "RealisasiDate"]
for c in date_cols:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

df_raw["Outstanding_PH"] = pd.to_numeric(
    df_raw["Outstanding_PH"], errors="coerce"
)

# =========================================================
# FILTER POSISI CA (WAJIB)
# =========================================================
df_ca = df_raw[
    df_raw["position_code"].str.upper().isin(
        ["Credit Analyst Staff"]
    )
].copy()

# =========================================================
# SORT HISTORY & AMBIL HASIL TERAKHIR PER APPS_ID
# =========================================================
df_ca = df_ca.sort_values(
    ["apps_id", "action_on"],
    ascending=[True, False]
)

df_ca_last = df_ca.drop_duplicates(
    subset="apps_id",
    keep="first"
)

# =========================================================
# NORMALISASI HASIL SCORING (HARDCODED LOGIC)
# =========================================================
def normalize_scoring(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).upper()
    if "REJECT" in x:
        return "REJECT"
    if "APPROVE" in x:
        return "APPROVE"
    if "REGULER" in x:
        return "REGULER"
    if "SCORING" in x:
        return "SCORING"
    return "OTHERS"

df_ca_last["Final_CA_Result"] = (
    df_ca_last["desc_status_apps"]
    .apply(normalize_scoring)
)

# =========================================================
# OSPH RANGE (SESUAI EXCEL)
# =========================================================
def osph_bucket(x):
    if pd.isna(x):
        return "UNKNOWN"
    if x < 250_000_000:
        return "0 - 250 Juta"
    elif x < 500_000_000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

df_ca_last["OSPH_Range"] = df_ca_last["Outstanding_PH"].apply(osph_bucket)

# =========================================================
# SLA CONFIG
# =========================================================
WORK_START = time(8, 30)
WORK_END = time(15, 30)

HOLIDAYS = pd.to_datetime([
    "01-01-2025", "27-01-2025", "28-01-2025", "29-01-2025", "28-03-2025", "31-03-2025", "01-04-2025", "02-04-2025", "03-04-2025", 
"04-04-2025", "07-04-2025", "18-04-2025", "01-05-2025", "12-05-2025", "29-05-2025", "06-06-2025", "09-06-2025", "27-06-2025", 
"18-08-2025", "05-09-2025", "25-12-2025", "26-12-2025", "31-12-2025", "01-01-2026", "02-01-2026", "16-01-2026", "16-02-2026",
"17-02-2026", "18-03-2026", "19-03-2026", "20-03-2026", "23-03-2026", "24-03-2026", "03-04-2026", "01-05-2026", "14-05-2026",
"27-05-2026", "28-05-2026", "01-06-2026", "16-06-2026", "17-08-2026", "25-08-2026", "25-12-2026", "31-12-2026"
])

def calc_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    total_minutes = 0
    current = start

    while current.date() <= end.date():
        if (
            current.weekday() < 5
            and current.normalize() not in HOLIDAYS
        ):
            day_start = datetime.combine(current.date(), WORK_START)
            day_end = datetime.combine(current.date(), WORK_END)

            s = max(start, day_start)
            e = min(end, day_end)

            if s < e:
                total_minutes += (e - s).total_seconds() / 60

        current += pd.Timedelta(days=1)

    return round(total_minutes / 60, 2)

df_ca_last["SLA_Hours"] = df_ca_last.apply(
    lambda x: calc_sla(x["Initiation"], x["action_on"]),
    axis=1
)

# =========================================================
# SIDEBAR FILTER
# =========================================================
st.sidebar.header(" Filter")

produk = st.sidebar.multiselect(
    "Produk",
    sorted(df_ca_last["Produk"].dropna().unique())
)

branch = st.sidebar.multiselect(
    "Branch",
    sorted(df_ca_last["branch_name"].dropna().unique())
)

osph = st.sidebar.multiselect(
    "OSPH Range",
    df_ca_last["OSPH_Range"].unique()
)

df_f = df_ca_last.copy()

if produk:
    df_f = df_f[df_f["Produk"].isin(produk)]
if branch:
    df_f = df_f[df_f["branch_name"].isin(branch)]
if osph:
    df_f = df_f[df_f["OSPH_Range"].isin(osph)]

# =========================================================
# KPI DIVISI
# =========================================================
st.subheader(" KPI Divisi CA")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Apps ID (Distinct)", df_f["apps_id"].nunique())
k2.metric("Total Record", len(df_f))
k3.metric("Avg SLA (Hours)", round(df_f["SLA_Hours"].mean(),2))
k4.metric(
    "Reject Rate (%)",
    round((df_f["Final_CA_Result"]=="REJECT").mean()*100,2)
)

# =========================================================
# ANALYTICAL – SCORING VS OSPH
# =========================================================
st.subheader(" Scoring vs OSPH")

fig1 = px.histogram(
    df_f,
    x="OSPH_Range",
    color="Final_CA_Result",
    barmode="group"
)
st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# ANALYTICAL – KENDARAAN & PEKERJAAN
# =========================================================
st.subheader(" Jenis Kendaraan vs Reject Rate")

veh = (
    df_f.groupby("JenisKendaraan")
    .apply(lambda x: (x["Final_CA_Result"]=="REJECT").mean())
    .reset_index(name="Reject_Rate")
)

fig2 = px.bar(
    veh,
    x="JenisKendaraan",
    y="Reject_Rate",
    text_auto=".2%"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader(" Pekerjaan vs Decision Pattern")

job = pd.crosstab(
    df_f["Pekerjaan"],
    df_f["Final_CA_Result"],
    normalize="index"
)

fig3 = px.imshow(job, text_auto=".2f", aspect="auto")
st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# EXCEL-LIKE SUMMARY (OSPH)
# =========================================================
st.subheader(" Summary OSPH (Excel Style)")

summary = pd.pivot_table(
    df_f,
    index="OSPH_Range",
    columns="Final_CA_Result",
    values="apps_id",
    aggfunc=pd.Series.nunique,
    fill_value=0
)

summary["Total_apps_id"] = summary.sum(axis=1)
summary["% dari Total"] = round(
    summary["Total_apps_id"] / summary["Total_apps_id"].sum() * 100, 2
)

st.dataframe(summary.reset_index(), use_container_width=True)

# =========================================================
# RAW DETAIL (AUDIT)
# =========================================================
st.subheader(" Raw Detail CA (Audit Layer)")
st.dataframe(
    df_f[[
        "apps_id","Produk","branch_name","OSPH_Range",
        "JenisKendaraan","Pekerjaan",
        "Final_CA_Result","SLA_Hours"
    ]],
    use_container_width=True
)

st.markdown("""
###  Insight untuk Dept Head
- OSPH tinggi menunjukkan reject rate lebih besar → indikasi early risk
- Kombinasi kendaraan + pekerjaan tertentu konsisten masuk CA
- SLA meningkat seiring kompleksitas aplikasi
- Pola ini bisa dipakai sebagai **pre-screening rule**
""")
