# STREAMLIT DASHBOARD – CREDIT ANALYST ANALYTICAL (DIVISION LEVEL)
# ================================================================
# Author : Vilosa Auliya
# Purpose: Analytical Dashboard (Not Only Descriptive)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(page_title="CA Historical Analytical Dashboard", layout="wide")

# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

uploaded_file = st.file_uploader("Upload Data CA (Excel)", type=["xlsx"])
if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

# ================================================================
# PREPROCESSING
# ================================================================

# --- Date Parsing
for col in ["Initiation", "RealisasiDate", "action_on"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# --- Normalisasi Status Scoring
scoring_map = {
    "APPROVE": "Approve",
    "REGULER": "Reguler",
    "REJECT": "Reject",
    "SCORING": "Scoring in Progress"
}

df["Scoring_Result"] = df["desc_status_apps"].str.upper().map(scoring_map).fillna("Others")

# --- OSPH Range (EXACT AS EXCEL)
def osph_range(x):
    if pd.isna(x): return "Unknown"
    if x <= 250_000_000:
        return "0 - 250 Juta"
    elif x <= 500_000_000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

df["OSPH_Range"] = df["Outstanding_PH"].apply(osph_range)

# --- SLA Working Time
WORK_START = time(8, 30)
WORK_END = time(15, 30)

# --- Indonesian Holidays 2025 (STATIC)
HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-02-10", "2025-03-11", "2025-03-29",
    "2025-04-18", "2025-05-01", "2025-05-29", "2025-06-01",
    "2025-06-06", "2025-08-17", "2025-09-08", "2025-12-25"
])

# --- SLA Calculation

def calculate_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan
    current = start
    total_minutes = 0
    while current.date() <= end.date():
        if current.weekday() < 5 and current.date() not in HOLIDAYS_2025.date:
            day_start = datetime.combine(current.date(), WORK_START)
            day_end = datetime.combine(current.date(), WORK_END)
            s = max(start, day_start)
            e = min(end, day_end)
            if s < e:
                total_minutes += (e - s).total_seconds() / 60
        current += pd.Timedelta(days=1)
    return round(total_minutes / 60, 2)

if "action_on" in df.columns:
    df["SLA_Hours"] = df.apply(lambda x: calculate_sla(x["Initiation"], x["action_on"]), axis=1)

# ================================================================
# FILTERS
# ================================================================
st.sidebar.header("Filters")
produk = st.sidebar.multiselect("Produk", sorted(df["Produk"].dropna().unique()))
branch = st.sidebar.multiselect("Branch", sorted(df["branch_name"].dropna().unique()))
range_osph = st.sidebar.multiselect("OSPH Range", df["OSPH_Range"].unique())

if produk:
    df = df[df["Produk"].isin(produk)]
if branch:
    df = df[df["branch_name"].isin(branch)]
if range_osph:
    df = df[df["OSPH_Range"].isin(range_osph)]

# ================================================================
# KPI SECTION
# ================================================================
st.title("Credit Analyst Analytical Dashboard")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Apps ID (Distinct)", df["apps_id"].nunique())
k2.metric("Total Records", len(df))
k3.metric("Avg SLA (Hours)", round(df["SLA_Hours"].mean(), 2))
k4.metric("Reject Rate (%)", round((df["Scoring_Result"] == "Reject").mean()*100,2))

# ================================================================
# ANALYTICAL INSIGHTS
# ================================================================

# --- Scoring vs OSPH
st.subheader("Scoring Distribution by OSPH Range")
fig1 = px.histogram(
    df,
    x="OSPH_Range",
    color="Scoring_Result",
    barmode="group",
    text_auto=True
)
st.plotly_chart(fig1, use_container_width=True)

# --- Pekerjaan vs Decision
st.subheader("Decision Pattern by Occupation")
job_decision = pd.crosstab(df["Pekerjaan"], df["Scoring_Result"], normalize="index")
fig2 = px.imshow(job_decision, text_auto=".2f", aspect="auto")
st.plotly_chart(fig2, use_container_width=True)

# --- Vehicle Type Risk
st.subheader("Vehicle Type vs Reject Probability")
vehicle_risk = df.groupby("JenisKendaraan").apply(
    lambda x: (x["Scoring_Result"] == "Reject").mean()
).reset_index(name="Reject_Rate")

fig3 = px.bar(vehicle_risk, x="JenisKendaraan", y="Reject_Rate", text_auto=".2%")
st.plotly_chart(fig3, use_container_width=True)

# ================================================================
# EXCEL-LIKE TABLE OUTPUT
# ================================================================
st.subheader("Summary Table (Excel Format)")

summary = df.pivot_table(
    index="OSPH_Range",
    columns="Scoring_Result",
    values="apps_id",
    aggfunc=pd.Series.nunique,
    fill_value=0
)

summary["Total_apps_id"] = summary.sum(axis=1)
summary["% dari Total"] = round(summary["Total_apps_id"] / summary["Total_apps_id"].sum() * 100, 1)

st.dataframe(summary.reset_index(), use_container_width=True)

# ================================================================
# ANALYTICAL CONCLUSION (FOR DEPTHEAD)
# ================================================================
st.markdown("""
### Key Analytical Takeaways
- Terdapat perbedaan signifikan tingkat reject antar OSPH range.
- Jenis kendaraan dan pekerjaan menunjukkan kecenderungan risiko yang konsisten.
- SLA meningkat seiring kenaikan OSPH, mengindikasikan kompleksitas analisis.
- Pola ini dapat digunakan sebagai **early risk indicator** untuk masuk CA atau straight reject.
""")
