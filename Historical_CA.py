# ================================================================
# CREDIT ANALYST HISTORICAL ANALYTICAL DASHBOARD
# LEVEL : DIVISION / DEPT HEAD
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import plotly.express as px

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="CA Historical – Analytical Dashboard",
    layout="wide"
)

# ================================================================
# DATA LOADING
# ================================================================
DATA_FILE = "DataHistoricalCA.xlsx"

if not Path(DATA_FILE).exists():
    st.error("❌ DataHistoricalCA.xlsx tidak ditemukan di repository")
    st.stop()

df_raw = pd.read_excel(DATA_FILE)
df = df_raw.copy()

# ================================================================
# PREPROCESSING SECTION
# ================================================================

# --- Date Parsing
date_cols = ["Initiation", "RealisasiDate", "action_on"]
for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# --- Scoring Mapping (MATCH EXCEL)
scoring_map = {
    "APPROVE": "Approve",
    "REGULER": "Reguler",
    "REJECT": "Reject",
    "SCORING": "Scoring In Progress"
}

df["Scoring_Result"] = (
    df["desc_status_apps"]
    .astype(str)
    .str.upper()
    .map(scoring_map)
    .fillna("Others")
)

# ================================================================
# OSPH BUCKETING (EXCEL STANDARD)
# ================================================================
def osph_bucket(x):
    if pd.isna(x):
        return "Unknown"
    elif x <= 250_000_000:
        return "0 – 250 Juta"
    elif x <= 500_000_000:
        return "250 – 500 Juta"
    else:
        return "> 500 Juta"

df["OSPH_Range"] = df["Outstanding_PH"].apply(osph_bucket)

# ================================================================
# SLA CONFIGURATION
# ================================================================
WORK_START = time(8, 30)
WORK_END   = time(15, 30)

HOLIDAYS = pd.to_datetime([
    "2025-01-01","2025-01-27","2025-01-28","2025-01-29",
    "2025-03-28","2025-03-31",
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04","2025-04-07",
    "2025-04-18","2025-05-01","2025-05-12","2025-05-29",
    "2025-06-06","2025-06-09","2025-06-27",
    "2025-08-18","2025-09-05",
    "2025-12-25","2025-12-26","2025-12-31"
])

HOLIDAYS_SET = set(HOLIDAYS.date)

def calculate_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    total_minutes = 0
    cur = start.date()

    while cur <= end.date():
        if cur.weekday() < 5 and cur not in HOLIDAYS_SET:
            s = max(start, datetime.combine(cur, WORK_START))
            e = min(end, datetime.combine(cur, WORK_END))
            if s < e:
                total_minutes += (e - s).seconds / 60
        cur += pd.Timedelta(days=1)

    return round(total_minutes / 60, 2)

df["SLA_Hours"] = df.apply(
    lambda x: calculate_sla(x["Initiation"], x["action_on"]),
    axis=1
)

df["SLA_Breach"] = np.where(df["SLA_Hours"] > 6, "Yes", "No")

# ================================================================
# FILTERS
# ================================================================
st.sidebar.header("Filters")

produk = st.sidebar.multiselect("Produk", df["Produk"].dropna().unique())
branch = st.sidebar.multiselect("Branch", df["branch_name"].dropna().unique())
osph   = st.sidebar.multiselect("OSPH Range", df["OSPH_Range"].unique())

if produk:
    df = df[df["Produk"].isin(produk)]
if branch:
    df = df[df["branch_name"].isin(branch)]
if osph:
    df = df[df["OSPH_Range"].isin(osph)]

# ================================================================
# KPI SECTION
# ================================================================
st.title("Credit Analyst Historical – Analytical Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Distinct Apps ID", df["apps_id"].nunique())
c2.metric("Total Record", len(df))
c3.metric("Avg SLA (Hours)", round(df["SLA_Hours"].mean(),2))
c4.metric("Reject Rate (%)", round((df["Scoring_Result"]=="Reject").mean()*100,2))
c5.metric("SLA Breach (%)", round((df["SLA_Breach"]=="Yes").mean()*100,2))

# ================================================================
# ANALYTICAL SECTION
# ================================================================

st.subheader("1️⃣ OSPH vs Decision (Risk Concentration)")
st.plotly_chart(
    px.histogram(df, x="OSPH_Range", color="Scoring_Result", barmode="group"),
    use_container_width=True
)

st.subheader("2️⃣ Occupation Risk Pattern")
occ_matrix = pd.crosstab(df["Pekerjaan"], df["Scoring_Result"], normalize="index")
st.plotly_chart(px.imshow(occ_matrix, aspect="auto", text_auto=".2f"))

st.subheader("3️⃣ Vehicle Type – Reject Propensity")
veh = df.groupby("JenisKendaraan").apply(
    lambda x: (x["Scoring_Result"]=="Reject").mean()
).reset_index(name="Reject_Rate")
st.plotly_chart(px.bar(veh, x="JenisKendaraan", y="Reject_Rate", text_auto=".2%"))

st.subheader("4️⃣ SLA Breach vs OSPH")
sla_tab = pd.crosstab(df["OSPH_Range"], df["SLA_Breach"], normalize="index")
st.dataframe(sla_tab.style.format("{:.2%}"))

# ================================================================
# EXCEL-LIKE SUMMARY TABLE
# ================================================================
st.subheader("Summary – Excel Equivalent")

summary = df.pivot_table(
    index="OSPH_Range",
    columns="Scoring_Result",
    values="apps_id",
    aggfunc=pd.Series.nunique,
    fill_value=0
)

summary["Total"] = summary.sum(axis=1)
summary["% Contribution"] = (summary["Total"] / summary["Total"].sum() * 100).round(1)

st.dataframe(summary.reset_index(), use_container_width=True)

# ================================================================
# MANAGEMENT INSIGHT
# ================================================================
st.markdown("""
### 🔍 Key Analytical Insight (Dept Head)
- OSPH tinggi memiliki reject propensity & SLA breach lebih besar
- Kombinasi pekerjaan & kendaraan membentuk **risk cluster**
- Bisa dijadikan **early rule** sebelum masuk CA
- Mendukung keputusan **capacity planning & policy adjustment**
""")
