import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="OSPH Executive Dashboard",
    layout="wide"
)

WORK_START = time(8,30)
WORK_END   = time(15,30)

# =========================
# LOAD HOLIDAYS
# =========================

@st.cache_data
def load_data():
    df = pd.read_excel("DataHistoricalCA.xlsx")

    df["initiation"] = pd.to_datetime(df["initiation"])
    df["action_on"]  = pd.to_datetime(df["action_on"])

    df["osph_range"] = np.where(
        df["osph"] <= 250_000_000, "0-250 Juta",
        np.where(df["osph"] <= 500_000_000, "250-500 Juta", "500 Juta+")
    )

    df["sla_hours"] = df.apply(
        lambda x: calculate_sla(x["initiation"], x["action_on"]),
        axis=1
    )

    return df

# =========================
# SLA FUNCTIONS
# =========================
def is_workday(dt):
    return dt.weekday() < 5 and dt.date() not in HOLIDAYS

def next_workday(dt):
    dt += timedelta(days=1)
    while not is_workday(dt):
        dt += timedelta(days=1)
    return datetime.combine(dt.date(), WORK_START)

def calculate_sla(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan

    total_minutes = 0
    current = start

    if current.time() > WORK_END:
        current = next_workday(current)

    while current < end:
        if is_workday(current):
            day_start = datetime.combine(current.date(), WORK_START)
            day_end   = datetime.combine(current.date(), WORK_END)

            start_time = max(current, day_start)
            end_time   = min(end, day_end)

            if start_time < end_time:
                total_minutes += (end_time - start_time).seconds / 60

        current = datetime.combine(current.date(), time(0,0)) + timedelta(days=1)

    return round(total_minutes / 60, 2)

# =========================
# PREPROCESS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("historical.csv")

    df["initiation"] = pd.to_datetime(df["initiation"])
    df["action_on"]  = pd.to_datetime(df["action_on"])

    # OSPH Range
    df["osph_range"] = np.where(
        df["osph"] <= 250_000_000, "0-250 Juta",
        np.where(df["osph"] <= 500_000_000, "250-500 Juta", "500 Juta+")
    )

    # SLA
    df["sla_hours"] = df.apply(
        lambda x: calculate_sla(x["initiation"], x["action_on"]),
        axis=1
    )

    return df

df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.title("Filter Global")

produk = st.sidebar.multiselect(
    "Produk",
    sorted(df["product"].dropna().unique()),
    default=sorted(df["product"].dropna().unique())
)

df = df[df["product"].isin(produk)]

# =========================
# HEADER
# =========================
st.title("📊 OSPH Executive Dashboard")
st.caption("Decision-level dashboard for Division Head")

# =========================
# KPI EXECUTIVE
# =========================
total_app = df["appid"].nunique()
approve_rate = (df["hasil_scoring"]=="APPROVE").mean()*100
reject_rate  = (df["hasil_scoring"]=="REJECT").mean()*100
avg_sla      = df["sla_hours"].mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total AppID", total_app)
c2.metric("Approval Rate", f"{approve_rate:.1f}%")
c3.metric("Reject Rate", f"{reject_rate:.1f}%")
c4.metric("Avg SLA (Jam Kerja)", f"{avg_sla:.1f}")

st.divider()

# =========================
# EXECUTIVE INSIGHT
# =========================
top_risk = (
    df[df["hasil_scoring"]=="REJECT"]
    .groupby(["osph_range","product"])
    .appid.nunique()
    .sort_values(ascending=False)
    .reset_index()
)

if len(top_risk) > 0:
    tr = top_risk.iloc[0]
    st.info(
        f"""
        🔍 **Key Executive Insight**

        Reject tertinggi terjadi pada:
        - **OSPH {tr['osph_range']}**
        - **Produk {tr['product']}**
        """
    )

# =========================
# OSPH RISK
# =========================
st.subheader("📌 OSPH Risk Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribusi AppID per OSPH Range**")
    st.dataframe(
        df.groupby("osph_range")
        .appid.nunique()
        .reset_index(name="Total AppID")
    )

with col2:
    st.markdown("**Approval Rate per OSPH Range**")
    st.dataframe(
        df.groupby("osph_range")
        .apply(lambda x: (x["hasil_scoring"]=="APPROVE").mean()*100)
        .reset_index(name="Approve %")
    )

st.divider()

# =========================
# PRODUCT PERFORMANCE
# =========================
st.subheader("📦 Product Performance")

st.dataframe(
    df.groupby("product")
    .agg(
        AppID=("appid","nunique"),
        ApproveRate=("hasil_scoring", lambda x: (x=="APPROVE").mean()*100),
        AvgSLA=("sla_hours","mean"),
        AvgOSPH=("osph","mean")
    )
    .reset_index()
)

st.divider()

# =========================
# RISK SEGMENTATION
# =========================
st.subheader("⚠️ Risk Segmentation")

st.markdown("**OSPH × Jenis Kendaraan × Pekerjaan**")

st.dataframe(
    df.pivot_table(
        index=["osph_range","jenis_kendaraan"],
        columns="pekerjaan",
        values="appid",
        aggfunc="nunique",
        fill_value=0
    )
)

st.divider()

# =========================
# CA PERFORMANCE
# =========================
st.subheader("👤 CA Performance (Governance)")

st.dataframe(
    df.groupby("user_name")
    .agg(
        AppID=("appid","nunique"),
        ApproveRate=("hasil_scoring", lambda x: (x=="APPROVE").mean()*100),
        AvgSLA=("sla_hours","mean")
    )
    .reset_index()
)

st.divider()

# =========================
# SLA BOTTLENECK
# =========================
st.subheader("⏱️ SLA Bottleneck")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Avg SLA by Produk**")
    st.dataframe(
        df.groupby("product")["sla_hours"].mean().reset_index()
    )

with col2:
    st.markdown("**Avg SLA by Branch**")
    st.dataframe(
        df.groupby("branch_name")["sla_hours"].mean().reset_index()
    )
