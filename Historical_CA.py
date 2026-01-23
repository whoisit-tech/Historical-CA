# ================================================================
# CREDIT ANALYST HISTORICAL – HARDCORE DIVISION DASHBOARD
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="CA Division Hardcore Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# FILE CONFIG
# ================================================================
DATA_FILE = "DataHistoricalCA.xlsx"

WORK_START = time(8, 30)
WORK_END   = time(15, 30)

# ================================================================
# HOLIDAYS 2025 (FIX)
# ================================================================
HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01","2025-01-27","2025-01-28","2025-01-29",
    "2025-03-28","2025-03-31",
    "2025-04-01","2025-04-02","2025-04-03","2025-04-04","2025-04-07",
    "2025-04-18","2025-05-01","2025-05-12","2025-05-29",
    "2025-06-06","2025-06-09","2025-06-27",
    "2025-08-18","2025-09-05",
    "2025-12-25","2025-12-26","2025-12-31"
]).date

# ================================================================
# LOAD DATA
# ================================================================
if not Path(DATA_FILE).exists():
    st.error("❌ DataHistoricalCA.xlsx tidak ditemukan")
    st.stop()

df_raw = pd.read_excel(DATA_FILE)

# ================================================================
# RAW BACKUP (IMPORTANT FOR AUDIT)
# ================================================================
df_raw_backup = df_raw.copy()

# ================================================================
# DATE PARSING
# ================================================================
DATE_COLS = ["Initiation","RealisasiDate","action_on"]
for c in DATE_COLS:
    if c in df_raw.columns:
        df_raw[c] = pd.to_datetime(df_raw[c], errors="coerce")

# ================================================================
# APPS STATUS NORMALIZATION (HARDCORE)
# ================================================================
def normalize_apps_status(x):
    if pd.isna(x): return "UNKNOWN"
    x = str(x).upper().strip()

    if "REJECT" in x:
        return "REJECT"
    if "APPROVE" in x:
        return "APPROVE"
    if "REGULER" in x:
        return "REGULER"
    if "SCORING" in x:
        return "SCORING"
    if x in ["-", ""]:
        return "UNDEFINED"

    return "OTHERS"

df_raw["apps_status_raw"] = df_raw["desc_status_apps"]
df_raw["apps_status_group"] = df_raw["desc_status_apps"].apply(normalize_apps_status)

# ================================================================
# DECISION FINAL (BANK STYLE)
# ================================================================
def decision_engine(row):
    status = row["apps_status_group"]

    if status == "REJECT":
        return "FAIL"
    if status == "APPROVE":
        return "PASS"
    if status == "REGULER":
        return "CONDITIONAL"
    if status == "SCORING":
        return "PENDING"
    return "REVIEW"

df_raw["decision_final"] = df_raw.apply(decision_engine, axis=1)

# ================================================================
# OSPH BUCKET (EXCEL MATCH)
# ================================================================
def osph_bucket(v):
    if pd.isna(v): return "UNKNOWN"
    if v <= 250_000_000:
        return "0–250 Juta"
    elif v <= 500_000_000:
        return "250–500 Juta"
    elif v <= 1_000_000_000:
        return "500 Juta–1 M"
    else:
        return ">1 M"

df_raw["OSPH_RANGE"] = df_raw["Outstanding_PH"].apply(osph_bucket)

# ================================================================
# SLA ENGINE (WORKING HOURS + HOLIDAY)
# ================================================================
def calculate_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    total_minutes = 0
    current = start

    while current.date() <= end.date():
        if current.weekday() < 5 and current.date() not in HOLIDAYS_2025:
            day_start = datetime.combine(current.date(), WORK_START)
            day_end   = datetime.combine(current.date(), WORK_END)

            s = max(start, day_start)
            e = min(end, day_end)

            if s < e:
                total_minutes += (e - s).total_seconds() / 60

        current += pd.Timedelta(days=1)

    return round(total_minutes / 60, 2)

df_raw["SLA_HOURS"] = df_raw.apply(
    lambda r: calculate_sla(r["Initiation"], r["action_on"]),
    axis=1
)

# ================================================================
# RISK ENGINE (ANALYTICAL)
# ================================================================
def risk_engine(row):
    if row["decision_final"] == "FAIL":
        return "HIGH RISK"
    if row["OSPH_RANGE"] in [">1 M","500 Juta–1 M"] and row["SLA_HOURS"] > 24:
        return "PROCESS + CREDIT RISK"
    if row["decision_final"] == "PASS" and row["SLA_HOURS"] <= 8:
        return "LOW RISK"
    return "MEDIUM RISK"

df_raw["risk_category"] = df_raw.apply(risk_engine, axis=1)

# ================================================================
# SIDEBAR FILTER
# ================================================================
st.sidebar.header("Filter")

branch = st.sidebar.multiselect(
    "Branch", sorted(df_raw["branch_name"].dropna().unique())
)

produk = st.sidebar.multiselect(
    "Produk", sorted(df_raw["Produk"].dropna().unique())
)

decision = st.sidebar.multiselect(
    "Decision Final", sorted(df_raw["decision_final"].unique())
)

df = df_raw.copy()

if branch:
    df = df[df["branch_name"].isin(branch)]
if produk:
    df = df[df["Produk"].isin(produk)]
if decision:
    df = df[df["decision_final"].isin(decision)]

# ================================================================
# KPI DIVISION
# ================================================================
st.title("🏦 Credit Analyst Hardcore Division Dashboard")

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Apps ID (Distinct)", df["apps_id"].nunique())
k2.metric("Total Records", len(df))
k3.metric("Reject Rate (%)", round((df["decision_final"]=="FAIL").mean()*100,2))
k4.metric("Conditional (%)", round((df["decision_final"]=="CONDITIONAL").mean()*100,2))
k5.metric("Avg SLA (Hours)", round(df["SLA_HOURS"].mean(),2))
k6.metric("High Risk (%)", round((df["risk_category"].str.contains("HIGH")).mean()*100,2))

# ================================================================
# ANALYTICAL – STATUS DISTRIBUTION
# ================================================================
st.subheader("Apps Status Distribution (RAW vs GROUP)")

fig1 = px.histogram(
    df,
    x="apps_status_raw",
    color="apps_status_group",
    barmode="group"
)
st.plotly_chart(fig1, use_container_width=True)

# ================================================================
# OSPH vs DECISION
# ================================================================
st.subheader("OSPH Range vs Decision Outcome")

fig2 = px.histogram(
    df,
    x="OSPH_RANGE",
    color="decision_final",
    barmode="group",
    text_auto=True
)
st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# RAW DETAIL TABLE (DIVISI WAJIB ADA)
# ================================================================
st.subheader("Raw Detail (Audit View)")

st.dataframe(
    df[[
        "apps_id","branch_name","Produk",
        "apps_status_raw","apps_status_group","decision_final",
        "OSPH_RANGE","Outstanding_PH",
        "SLA_HOURS","risk_category"
    ]],
    use_container_width=True
)

# ================================================================
# EXCEL-LIKE SUMMARY
# ================================================================
st.subheader("Executive Summary (Excel Output)")

summary = pd.pivot_table(
    df,
    index="OSPH_RANGE",
    columns="decision_final",
    values="apps_id",
    aggfunc=pd.Series.nunique,
    fill_value=0
)

summary["TOTAL"] = summary.sum(axis=1)
summary["PERCENT"] = round(summary["TOTAL"]/summary["TOTAL"].sum()*100,2)

st.dataframe(summary.reset_index(), use_container_width=True)

# ================================================================
# MANAGEMENT INSIGHT
# ================================================================
st.markdown("""
### 🔥 Division-Level Analytical Insight
- Multiple approval layers (Approve 1/2, Reguler 1/2) terbukti **tidak homogen**
- OSPH tinggi + SLA panjang → indikasi **process & credit risk**
- Branch dengan dominasi conditional perlu **policy tightening**
- Framework ini bisa jadi **early filtering sebelum CA**
""")
