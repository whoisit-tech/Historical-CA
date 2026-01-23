# =========================================================
# CREDIT ANALYST HISTORICAL DASHBOARD – DIVISION VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="CA Historical Dashboard",
    layout="wide"
)

DATA_FILE = "DataHistoricalCA.xlsx"

WORK_START = time(8,30)
WORK_END   = time(15,30)

HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01","2025-01-27","2025-01-28","2025-01-29",
    "2025-03-28","2025-03-31",
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04","2025-04-07",
    "2025-04-18","2025-05-01","2025-05-12","2025-05-29",
    "2025-06-06","2025-06-09","2025-06-27",
    "2025-08-18","2025-09-05",
    "2025-12-25","2025-12-26","2025-12-31"
]).date

# =========================================================
# LOAD DATA
# =========================================================
if not Path(DATA_FILE).exists():
    st.error("File DataHistoricalCA.xlsx tidak ditemukan")
    st.stop()

df_raw = pd.read_excel(DATA_FILE)

# =========================================================
# DATE PARSING
# =========================================================
for c in ["Initiation","action_on","RealisasiDate"]:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

# =========================================================
# FILTER POSISI CA (WAJIB)
# =========================================================
df_ca = df_raw[
    df_raw["position_name"].str.contains("CA", case=False, na=False)
].copy()

# =========================================================
# SORT HISTORY & AMBIL CA TERAKHIR
# =========================================================
df_ca = df_ca.sort_values(
    by=["apps_id","action_on"],
    ascending=[True, True]
)

last_ca = (
    df_ca
    .groupby("apps_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

# =========================================================
# NORMALISASI HASIL CA
# =========================================================
def normalize_ca(x):
    if pd.isna(x): return "Scoring in Progress"
    x = str(x).upper()
    if "REJECT" in x: return "Reject"
    if "APPROVE" in x: return "Approve"
    if "REGULER" in x: return "Reguler"
    return "Scoring in Progress"

last_ca["Hasil_CA"] = last_ca["desc_status_apps"].apply(normalize_ca)

# =========================================================
# OSPH RANGE (MATCH EXCEL)
# =========================================================
def osph_range(v):
    if pd.isna(v): return "Unknown"
    if v <= 250_000_000:
        return "0 - 250 Juta"
    elif v <= 500_000_000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

last_ca["OSPH_Range"] = last_ca["Outstanding_PH"].apply(osph_range)

# =========================================================
# SLA CALCULATION
# =========================================================
def calc_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    if start.time() >= WORK_END:
        start = datetime.combine(
            start.date() + pd.Timedelta(days=1),
            WORK_START
        )

    total_minutes = 0
    cur = start

    while cur.date() <= end.date():
        if cur.weekday() < 5 and cur.date() not in HOLIDAYS_2025:
            ds = datetime.combine(cur.date(), WORK_START)
            de = datetime.combine(cur.date(), WORK_END)

            s = max(start, ds)
            e = min(end, de)

            if s < e:
                total_minutes += (e - s).total_seconds() / 60

        cur += pd.Timedelta(days=1)

    return round(total_minutes/60,2)

last_ca["SLA_Hours"] = last_ca.apply(
    lambda r: calc_sla(r["Initiation"], r["action_on"]),
    axis=1
)

# =========================================================
# SIDEBAR FILTER
# =========================================================
st.sidebar.header("Filter")

produk = st.sidebar.multiselect(
    "Produk", sorted(last_ca["Produk"].dropna().unique())
)
branch = st.sidebar.multiselect(
    "Branch", sorted(last_ca["branch_name"].dropna().unique())
)
hasil = st.sidebar.multiselect(
    "Hasil CA", ["Approve","Reguler","Reject","Scoring in Progress"]
)

df = last_ca.copy()

if produk:
    df = df[df["Produk"].isin(produk)]
if branch:
    df = df[df["branch_name"].isin(branch)]
if hasil:
    df = df[df["Hasil_CA"].isin(hasil)]

# =========================================================
# KPI
# =========================================================
st.title("📊 Credit Analyst Historical Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Apps ID (Distinct)", df["apps_id"].nunique())
k2.metric("Total Records", len(df))
k3.metric("Avg SLA (Hours)", round(df["SLA_Hours"].mean(),2))
k4.metric("Reject Rate (%)", round((df["Hasil_CA"]=="Reject").mean()*100,2))

# =========================================================
# EXCEL-LIKE SUMMARY (PERSIS)
# =========================================================
st.subheader("Summary OSPH vs Hasil CA")

summary = pd.pivot_table(
    df,
    index="OSPH_Range",
    columns="Hasil_CA",
    values="apps_id",
    aggfunc="nunique",
    fill_value=0
)

summary["Total_apps_id"] = summary.sum(axis=1)
summary["% dari Total"] = round(
    summary["Total_apps_id"] /
    summary["Total_apps_id"].sum() * 100, 1
)

st.dataframe(summary.reset_index(), use_container_width=True)

# =========================================================
# BREAKDOWN ANALYTICAL
# =========================================================
st.subheader("Breakdown Kendaraan & Pekerjaan")

tab1, tab2 = st.tabs(["Jenis Kendaraan","Pekerjaan"])

with tab1:
    st.dataframe(
        pd.pivot_table(
            df,
            index=["OSPH_Range","JenisKendaraan"],
            columns="Hasil_CA",
            values="apps_id",
            aggfunc="nunique",
            fill_value=0
        ).reset_index(),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        pd.pivot_table(
            df,
            index=["OSPH_Range","Pekerjaan"],
            columns="Hasil_CA",
            values="apps_id",
            aggfunc="nunique",
            fill_value=0
        ).reset_index(),
        use_container_width=True
    )

# =========================================================
# RAW DETAIL (AUDIT VIEW)
# =========================================================
st.subheader("Raw Detail (CA Last History)")

st.dataframe(
    df[
        [
            "apps_id","Produk","branch_name",
            "OSPH_Range","Outstanding_PH",
            "JenisKendaraan","Pekerjaan",
            "Hasil_CA","SLA_Hours"
        ]
    ],
    use_container_width=True
)

# =========================================================
# ANALYTICAL INSIGHT
# =========================================================
st.markdown("""
### Insight Utama
- OSPH besar cenderung meningkatkan probabilitas **Reguler & Reject**
- Jenis kendaraan tertentu menunjukkan konsistensi risiko
- SLA panjang sering muncul pada OSPH tinggi → indikasi **process bottleneck**
- Segmentasi ini dapat digunakan sebagai **early screening sebelum CA**
""")
