import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
st.set_page_config(page_title="CA Analytics Dashboard", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Tanggal merah
TANGGAL_MERAH = [
    "01-01-2025", "27-01-2025", "28-01-2025", "29-01-2025", "28-03-2025", "31-03-2025",
    "01-04-2025", "02-04-2025", "03-04-2025", "04-04-2025", "07-04-2025", "18-04-2025",
    "01-05-2025", "12-05-2025", "29-05-2025", "06-06-2025", "09-06-2025", "27-06-2025",
    "18-08-2025", "05-09-2025", "25-12-2025", "26-12-2025", "31-12-2025", "01-01-2026",
    "02-01-2026", "16-01-2026", "16-02-2026", "17-02-2026", "18-03-2026", "19-03-2026",
    "20-03-2026", "23-03-2026", "24-03-2026", "03-04-2026", "01-05-2026", "14-05-2026",
    "27-05-2026", "28-05-2026", "01-06-2026", "16-06-2026", "17-08-2026", "25-08-2026",
    "25-12-2026", "31-12-2026"
]
TANGGAL_MERAH_DT = [datetime.strptime(d, "%d-%m-%Y").date() for d in TANGGAL_MERAH]

# ========== FUNGSI HELPER ==========

def parse_date(date_str):
    if pd.isna(date_str) or date_str == '-':
        return None
    formats = ["%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).split('.')[0], fmt)
        except:
            continue
    return None

def is_working_day(date):
    if date.weekday() >= 5 or date.date() in TANGGAL_MERAH_DT:
        return False
    return True

def calculate_sla_days(start_dt, end_dt):
    if not start_dt or not end_dt:
        return None
    start_adjusted = start_dt
    if start_dt.time() >= datetime.strptime("15:30", "%H:%M").time():
        start_adjusted = start_dt + timedelta(days=1)
        start_adjusted = start_adjusted.replace(hour=8, minute=30, second=0)
        while not is_working_day(start_adjusted):
            start_adjusted += timedelta(days=1)
    working_days = 0
    current = start_adjusted.date()
    end_date = end_dt.date()
    while current <= end_date:
        if is_working_day(datetime.combine(current, datetime.min.time())):
            working_days += 1
        current += timedelta(days=1)
    return working_days

def get_osph_category(osph_value):
    if pd.isna(osph_value):
        return "Unknown"
    osph_value = float(osph_value)
    if osph_value <= 250000000:
        return "0 - 250 Juta"
    elif osph_value <= 500000000:
        return "250 - 500 Juta"
    else:
        return "500 Juta+"

def preprocess_data(df):
    df = df.copy()
    
    # Parse dates
    for col in ['action_on', 'Initiation', 'RealisasiDate']:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(parse_date)
    
    # Calculate SLA
    if 'action_on_parsed' in df.columns and 'RealisasiDate_parsed' in df.columns:
        df['SLA_Days'] = df.apply(
            lambda row: calculate_sla_days(row['action_on_parsed'], row['RealisasiDate_parsed']), axis=1
        )
    
    # Process OSPH
    if 'Outstanding_PH' in df.columns:
        df['OSPH_clean'] = pd.to_numeric(df['Outstanding_PH'].astype(str).str.replace(',', ''), errors='coerce')
        df['OSPH_Category'] = df['OSPH_clean'].apply(get_osph_category)
    
    # Standardize scoring
    if 'Hasil_Scoring_1' in df.columns:
        df['Scoring_Clean'] = df['Hasil_Scoring_1'].fillna('-')
        df['Scoring_Group'] = df['Scoring_Clean'].apply(lambda x: 
            'APPROVE' if 'APPROVE' in str(x).upper() or 'Approve' in str(x) else
            'REGULER' if 'REGULER' in str(x).upper() or 'Reguler' in str(x) else
            'REJECT' if 'REJECT' in str(x).upper() or 'Reject' in str(x) else
            'IN PROGRESS' if 'PROGRESS' in str(x).upper() else 'OTHER'
        )
    
    # Time features
    if 'action_on_parsed' in df.columns:
        df['Hour'] = df['action_on_parsed'].dt.hour
        df['DayOfWeek'] = df['action_on_parsed'].dt.dayofweek
        df['Month'] = df['action_on_parsed'].dt.month
        df['Week'] = df['action_on_parsed'].dt.isocalendar().week
        df['YearMonth'] = df['action_on_parsed'].dt.to_period('M').astype(str)
    
    # Risk score
    if 'OSPH_clean' in df.columns and 'SLA_Days' in df.columns:
        osph_norm = (df['OSPH_clean'] - df['OSPH_clean'].min()) / (df['OSPH_clean'].max() - df['OSPH_clean'].min() + 1)
        sla_norm = (df['SLA_Days'] - df['SLA_Days'].min()) / (df['SLA_Days'].max() - df['SLA_Days'].min() + 1)
        df['Risk_Score'] = (osph_norm * 0.6 + sla_norm * 0.4) * 100
    
    return df

@st.cache_data
def load_data():
    try:
        # URL file Excel di GitHub (raw)
        url = "https://raw.githubusercontent.com/whoisit-tech/Historical-CA/main/HistoricalCA.xlsx"
        df = pd.read_excel(url)
        return preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ========== MAIN APP ==========

def main():
    st.title("🎯 Credit Analyst Analytics Dashboard")
    st.markdown("**Comprehensive Analytics for Historical CA Performance**")
    st.markdown("---")
    
    # Load data langsung dari GitHub
    with st.spinner("Loading data from GitHub..."):
        df = load_data()
    
    if df is None or df.empty:
        st.error("❌ Failed to load data from GitHub. Please check the URL.")
        st.stop()
    
    st.success(f"✅ Data loaded successfully! Total records: {len(df):,}")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    products = ['All'] + sorted(df['Produk'].unique().tolist()) if 'Produk' in df.columns else ['All']
    selected_product = st.sidebar.selectbox("Produk", products)
    
    branches = ['All'] + sorted(df['branch_name'].dropna().unique().tolist()) if 'branch_name' in df.columns else ['All']
    selected_branch = st.sidebar.selectbox("Branch", branches)
    
    if 'action_on_parsed' in df.columns:
        min_date = df['action_on_parsed'].min().date()
        max_date = df['action_on_parsed'].max().date()
        date_range = st.sidebar.date_input("Periode", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    # Apply filters
    df_filtered = df.copy()
    if selected_product != 'All':
        df_filtered = df_filtered[df_filtered['Produk'] == selected_product]
    if selected_branch != 'All':
        df_filtered = df_filtered[df_filtered['branch_name'] == selected_branch]
    if 'action_on_parsed' in df.columns and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['action_on_parsed'].dt.date >= date_range[0]) &
            (df_filtered['action_on_parsed'].dt.date <= date_range[1])
        ]
    
    # ========== KPI SECTION ==========
    st.header("📊 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        total_apps = df_filtered['apps_id'].nunique() if 'apps_id' in df_filtered.columns else len(df_filtered)
        st.metric("📝 Total Apps", f"{total_apps:,}")
    
    with kpi_col2:
        if 'SLA_Days' in df_filtered.columns:
            avg_sla = df_filtered['SLA_Days'].mean()
            st.metric("⏱️ Avg SLA", f"{avg_sla:.1f}d" if not pd.isna(avg_sla) else "N/A")
        else:
            st.metric("⏱️ Avg SLA", "N/A")
    
    with kpi_col3:
        if 'Scoring_Group' in df_filtered.columns:
            approve_pct = (df_filtered['Scoring_Group'] == 'APPROVE').sum() / len(df_filtered) * 100
            st.metric("✅ Approval", f"{approve_pct:.1f}%")
        else:
            st.metric("✅ Approval", "N/A")
    
    with kpi_col4:
        if 'Scoring_Group' in df_filtered.columns:
            reguler_pct = (df_filtered['Scoring_Group'] == 'REGULER').sum() / len(df_filtered) * 100
            st.metric("⚡ Reguler", f"{reguler_pct:.1f}%")
        else:
            st.metric("⚡ Reguler", "N/A")
    
    with kpi_col5:
        if 'Scoring_Group' in df_filtered.columns:
            reject_pct = (df_filtered['Scoring_Group'] == 'REJECT').sum() / len(df_filtered) * 100
            st.metric("❌ Reject", f"{reject_pct:.1f}%")
        else:
            st.metric("❌ Reject", "N/A")
    
    st.markdown("---")
    
    # ========== HIGHLIGHT OUTPUTS ==========
    st.header("📌 Highlight Analysis - OSPH Breakdown")
    
    highlight_tabs = st.tabs(["📊 By Range Harga", "👔 By Pekerjaan", "🚗 By Jenis Kendaraan"])
    
    # TAB 1: Range Harga Analysis
    with highlight_tabs[0]:
        st.subheader("Analysis by OSPH Range")
        
        if 'OSPH_Category' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            range_analysis = df_filtered.groupby('OSPH_Category').agg({
                'apps_id': 'nunique',
                'OSPH_clean': ['min', 'max']
            }).reset_index()
            
            range_analysis.columns = ['Range Harga', 'Total Apps ID', 'Harga Min', 'Harga Max']
            
            # Add scoring counts
            scoring_counts = df_filtered.groupby('OSPH_Category')['Scoring_Group'].apply(
                lambda x: pd.Series({
                    'Approve 2': (x == 'APPROVE').sum(),
                    'Reguler 1': (x == 'REGULER').sum(),
                    'Reject 1': (x == 'REJECT').sum(),
                    'Scoring in Progress': (x == 'IN PROGRESS').sum()
                })
            ).reset_index()
            
            range_analysis = range_analysis.merge(scoring_counts, left_on='Range Harga', right_on='OSPH_Category').drop('OSPH_Category', axis=1)
            
            range_analysis['% dari Total'] = (range_analysis['Total Apps ID'] / total_apps * 100).round(1)
            range_analysis['Harga Min'] = range_analysis['Harga Min'].apply(lambda x: f"Rp {x/1e6:,.1f}M" if pd.notna(x) else "-")
            range_analysis['Harga Max'] = range_analysis['Harga Max'].apply(lambda x: f"Rp {x/1e6:,.1f}M" if pd.notna(x) else "-")
            
            st.dataframe(range_analysis, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(range_analysis, values='Total Apps ID', names='Range Harga',
                           title="Distribution by OSPH Range")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(range_analysis, x='Range Harga', y=['Approve 2', 'Reguler 1', 'Reject 1'],
                           title="Scoring Results by OSPH Range", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Pekerjaan Analysis
    with highlight_tabs[1]:
        st.subheader("Analysis by Pekerjaan & OSPH Range")
        
        if all(col in df_filtered.columns for col in ['OSPH_Category', 'Pekerjaan', 'Scoring_Group']):
            job_pivot = df_filtered.pivot_table(
                index='OSPH_Category',
                columns='Pekerjaan',
                values='apps_id',
                aggfunc='nunique',
                fill_value=0
            )
            
            st.dataframe(job_pivot, use_container_width=True)
            
            fig = px.imshow(job_pivot, text_auto=True, aspect="auto",
                          title="Heatmap: OSPH Range vs Pekerjaan",
                          color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            job_dist = df_filtered.groupby(['OSPH_Category', 'Pekerjaan']).size().reset_index(name='Count')
            fig = px.bar(job_dist, x='OSPH_Category', y='Count', color='Pekerjaan',
                       title="Distribution: OSPH Range by Pekerjaan", barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Jenis Kendaraan Analysis
    with highlight_tabs[2]:
        st.subheader("Analysis by Jenis Kendaraan & OSPH Range")
        
        if all(col in df_filtered.columns for col in ['OSPH_Category', 'JenisKendaraan']):
            vehicle_pivot = df_filtered.pivot_table(
                index='OSPH_Category',
                columns='JenisKendaraan',
                values='apps_id',
                aggfunc='nunique',
                fill_value=0
            )
            
            st.dataframe(vehicle_pivot, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.imshow(vehicle_pivot, text_auto=True, aspect="auto",
                              title="Heatmap: OSPH Range vs Jenis Kendaraan",
                              color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                vehicle_dist = df_filtered.groupby(['OSPH_Category', 'JenisKendaraan']).size().reset_index(name='Count')
                fig = px.bar(vehicle_dist, x='OSPH_Category', y='Count', color='JenisKendaraan',
                           title="Distribution by Vehicle Type", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========== DETAILED TABS ==========
    tabs = st.tabs(["📈 Trend & Pattern", "🎯 Conversion Analysis", "⏱️ SLA Deep Dive", 
                   "🔍 Segmentation", "💡 Insights", "📋 Raw Data"])
    
    # TAB 1: Trend & Pattern
    with tabs[0]:
        st.header("📈 Trend Analysis & Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'YearMonth' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
                monthly_trend = df_filtered.groupby('YearMonth').agg({
                    'apps_id': 'nunique',
                }).reset_index()
                monthly_trend['Approval_Rate'] = df_filtered.groupby('YearMonth')['Scoring_Group'].apply(
                    lambda x: (x == 'APPROVE').sum() / len(x) * 100
                ).values
                monthly_trend.columns = ['Month', 'Volume', 'Approval_Rate']
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=monthly_trend['Month'], y=monthly_trend['Volume'], name="Volume"), secondary_y=False)
                fig.add_trace(go.Scatter(x=monthly_trend['Month'], y=monthly_trend['Approval_Rate'], 
                                       name="Approval %", mode='lines+markers'), secondary_y=True)
                fig.update_layout(title="Monthly Volume & Approval Rate Trend")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'DayOfWeek' in df_filtered.columns:
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                dow_data = df_filtered.groupby('DayOfWeek').size().reset_index(name='Count')
                dow_data['Day'] = dow_data['DayOfWeek'].apply(lambda x: dow_names[x])
                
                fig = px.bar(dow_data, x='Day', y='Count', title="Weekly Pattern",
                           color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        if all(col in df_filtered.columns for col in ['OSPH_clean', 'SLA_Days', 'LastOD', 'max_OD']):
            st.subheader("🔗 Correlation Matrix")
            corr_df = df_filtered[['OSPH_clean', 'SLA_Days', 'LastOD', 'max_OD']].corr()
            fig = px.imshow(corr_df, text_auto=True, aspect="auto",
                          title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Conversion Analysis
    with tabs[1]:
        st.header("🎯 Conversion Funnel & Rate Analysis")
        
        if 'Scoring_Group' in df_filtered.columns:
            funnel_data = [
                ('Total Applications', len(df_filtered)),
                ('Scored', len(df_filtered[df_filtered['Scoring_Group'] != 'OTHER'])),
                ('Approved', len(df_filtered[df_filtered['Scoring_Group'] == 'APPROVE']))
            ]
            funnel_df = pd.DataFrame(funnel_data, columns=['Stage', 'Count'])
            
            fig = go.Figure(go.Funnel(y=funnel_df['Stage'], x=funnel_df['Count'],
                                     textinfo="value+percent total"))
            fig.update_layout(title="Conversion Funnel")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'OSPH_Category' in df_filtered.columns:
                    osph_conv = df_filtered.groupby('OSPH_Category')['Scoring_Group'].apply(
                        lambda x: (x == 'APPROVE').sum() / len(x) * 100
                    ).reset_index(name='Approval_Rate')
                    fig = px.bar(osph_conv, x='OSPH_Category', y='Approval_Rate',
                               title="Approval Rate by OSPH", color='Approval_Rate')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Pekerjaan' in df_filtered.columns:
                    job_conv = df_filtered.groupby('Pekerjaan')['Scoring_Group'].apply(
                        lambda x: (x == 'APPROVE').sum() / len(x) * 100
                    ).reset_index(name='Approval_Rate')
                    fig = px.bar(job_conv, x='Pekerjaan', y='Approval_Rate',
                               title="Approval Rate by Pekerjaan", color='Approval_Rate')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                if 'JenisKendaraan' in df_filtered.columns:
                    vehicle_conv = df_filtered.groupby('JenisKendaraan')['Scoring_Group'].apply(
                        lambda x: (x == 'APPROVE').sum() / len(x) * 100
                    ).reset_index(name='Approval_Rate')
                    fig = px.bar(vehicle_conv, x='JenisKendaraan', y='Approval_Rate',
                               title="Approval Rate by Vehicle", color='Approval_Rate')
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: SLA Deep Dive
    with tabs[2]:
        st.header("⏱️ SLA Performance Deep Dive")
        
        if 'SLA_Days' in df_filtered.columns:
            df_sla = df_filtered[df_filtered['SLA_Days'].notna()].copy()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min SLA", f"{df_sla['SLA_Days'].min():.1f}d")
            with col2:
                st.metric("Median SLA", f"{df_sla['SLA_Days'].median():.1f}d")
            with col3:
                st.metric("Mean SLA", f"{df_sla['SLA_Days'].mean():.1f}d")
            with col4:
                st.metric("Max SLA", f"{df_sla['SLA_Days'].max():.1f}d")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df_sla, x='SLA_Days', nbins=30, title="SLA Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'OSPH_Category' in df_sla.columns:
                    fig = px.box(df_sla, x='OSPH_Category', y='SLA_Days',
                               title="SLA by OSPH Category")
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Segmentation
    with tabs[3]:
        st.header("🔍 Customer Segmentation Analysis")
        
        if all(col in df_filtered.columns for col in ['OSPH_Category', 'Pekerjaan', 'JenisKendaraan']):
            segment_data = df_filtered.groupby(['OSPH_Category', 'Pekerjaan', 'JenisKendaraan']).size().reset_index(name='Count')
            
            fig = px.sunburst(segment_data, path=['OSPH_Category', 'Pekerjaan', 'JenisKendaraan'],
                            values='Count', title="Hierarchical Segmentation")
            st.plotly_chart(fig, use_container_width=True)
            
            top_segments = segment_data.nlargest(10, 'Count')
            st.dataframe(top_segments, use_container_width=True)
    
    # TAB 5: Insights
    with tabs[4]:
        st.header("💡 Key Insights & Recommendations")
        
        insights = []
        
        if 'OSPH_Category' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            osph_approval = df_filtered.groupby('OSPH_Category')['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            best_osph = max(osph_approval, key=osph_approval.get)
            insights.append(f"🎯 **Best OSPH Segment**: {best_osph} with {osph_approval[best_osph]:.1f}% approval rate")
        
        if 'SLA_Days' in df_filtered.columns:
            target_sla = 3
            within_sla = (df_filtered['SLA_Days'] <= target_sla).sum()
            within_sla_pct = within_sla / len(df_filtered[df_filtered['SLA_Days'].notna()]) * 100
            insights.append(f"⏱️ **SLA Performance**: {within_sla_pct:.1f}% applications processed within {target_sla} days")
        
        if 'Hour' in df_filtered.columns:
            peak_hour = df_filtered['Hour'].mode()[0]
            peak_count = len(df_filtered[df_filtered['Hour'] == peak_hour])
            insights.append(f"🕐 **Peak Hour**: {peak_hour}:00 with {peak_count} applications ({peak_count/len(df_filtered)*100:.1f}%)")
        
        if 'Pekerjaan' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            job_approval = df_filtered.groupby('Pekerjaan')['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            best_job = max(job_approval, key=job_approval.get)
            insights.append(f"👔 **Best Job Type**: {best_job} with {job_approval[best_job]:.1f}% approval rate")
        
        if 'JenisKendaraan' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            vehicle_approval = df_filtered.groupby('JenisKendaraan')['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            best_vehicle = max(vehicle_approval, key=vehicle_approval.get)
            insights.append(f"🚗 **Best Vehicle Type**: {best_vehicle} with {vehicle_approval[best_vehicle]:.1f}% approval rate")
        
        if 'branch_name' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            branch_approval = df_filtered.groupby('branch_name')['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            best_branch = max(branch_approval, key=branch_approval.get)
            worst_branch = min(branch_approval, key=branch_approval.get)
            insights.append(f"🏢 **Top Branch**: {best_branch} ({branch_approval[best_branch]:.1f}% approval)")
            insights.append(f"⚠️ **Focus Branch**: {worst_branch} ({branch_approval[worst_branch]:.1f}% approval) - needs improvement")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        st.subheader("📋 Actionable Recommendations")
        recommendations = [
            "1. **Optimize Resources**: Focus CA resources during peak hours to reduce SLA",
            "2. **Segment Strategy**: Apply different approval criteria for high-performing OSPH segments",
            "3. **Branch Support**: Provide additional training/support to lower-performing branches",
            "4. **Risk-Based Pricing**: Consider risk scores in pricing strategy for better profitability",
            "5. **Process Automation**: Automate low-risk, high-approval segments to improve efficiency"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
        
        st.subheader("📊 Statistical Summary")
        
        if all(col in df_filtered.columns for col in ['OSPH_clean', 'SLA_Days']):
            stats_data = {
                'Metric': ['OSPH (Rp)', 'SLA (Days)'],
                'Mean': [
                    f"{df_filtered['OSPH_clean'].mean()/1e6:.1f}M",
                    f"{df_filtered['SLA_Days'].mean():.2f}"
                ],
                'Median': [
                    f"{df_filtered['OSPH_clean'].median()/1e6:.1f}M",
                    f"{df_filtered['SLA_Days'].median():.2f}"
                ],
                'Std Dev': [
                    f"{df_filtered['OSPH_clean'].std()/1e6:.1f}M",
                    f"{df_filtered['SLA_Days'].std():.2f}"
                ],
                'Min': [
                    f"{df_filtered['OSPH_clean'].min()/1e6:.1f}M",
                    f"{df_filtered['SLA_Days'].min():.2f}"
                ],
                'Max': [
                    f"{df_filtered['OSPH_clean'].max()/1e6:.1f}M",
                    f"{df_filtered['SLA_Days'].max():.2f}"
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    # TAB 6: Raw Data
    with tabs[5]:
        st.header("📋 Raw Data Explorer")
        
        all_columns = df_filtered.columns.tolist()
        default_cols = ['apps_id', 'Produk', 'OSPH_Category', 'Pekerjaan', 'JenisKendaraan', 
                       'Scoring_Group', 'SLA_Days', 'branch_name']
        display_cols = [col for col in default_cols if col in all_columns]
        
        selected_cols = st.multiselect("Select columns to display", all_columns, default=display_cols)
        
        if selected_cols:
            search_term = st.text_input("🔍 Search in data")
            
            display_df = df_filtered[selected_cols].copy()
            
            if search_term:
                mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                display_df = display_df[mask]
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"CA_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.subheader("📊 Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(display_df))
            with col2:
                st.metric("Total Columns", len(selected_cols))
            with col3:
                st.metric("Unique Apps", display_df['apps_id'].nunique() if 'apps_id' in selected_cols else "N/A")
    
    st.markdown("---")
    st.header("🔬 Advanced Analytics")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.subheader("📈 Approval Rate Trends")
        
        if all(col in df_filtered.columns for col in ['YearMonth', 'Scoring_Group']):
            monthly_approval = df_filtered.groupby('YearMonth')['Scoring_Group'].apply(
                lambda x: pd.Series({
                    'Approval_Rate': (x == 'APPROVE').sum() / len(x) * 100,
                    'Reject_Rate': (x == 'REJECT').sum() / len(x) * 100,
                    'Reguler_Rate': (x == 'REGULER').sum() / len(x) * 100
                })
            ).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_approval['YearMonth'], y=monthly_approval['Approval_Rate'],
                                   mode='lines+markers', name='Approval', line=dict(color='#10b981', width=3)))
            fig.add_trace(go.Scatter(x=monthly_approval['YearMonth'], y=monthly_approval['Reguler_Rate'],
                                   mode='lines+markers', name='Reguler', line=dict(color='#f59e0b', width=3)))
            fig.add_trace(go.Scatter(x=monthly_approval['YearMonth'], y=monthly_approval['Reject_Rate'],
                                   mode='lines+markers', name='Reject', line=dict(color='#ef4444', width=3)))
            
            fig.update_layout(title="Monthly Approval/Reguler/Reject Rate Trends",
                            xaxis_title="Month", yaxis_title="Rate (%)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
    
    with adv_col2:
        st.subheader("🎯 User Performance")
        
        if all(col in df_filtered.columns for col in ['user_name', 'Scoring_Group']):
            user_perf = df_filtered.groupby('user_name').agg({
                'apps_id': 'count',
            }).reset_index()
            user_perf['Approval_Rate'] = df_filtered.groupby('user_name')['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).values
            user_perf['Avg_SLA'] = df_filtered.groupby('user_name')['SLA_Days'].mean().values if 'SLA_Days' in df_filtered.columns else 0
            user_perf.columns = ['User', 'Total_Apps', 'Approval_Rate', 'Avg_SLA']
            user_perf = user_perf.sort_values('Total_Apps', ascending=False)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=user_perf['User'], y=user_perf['Total_Apps'], 
                               name="Total Apps", marker_color='#667eea'), secondary_y=False)
            fig.add_trace(go.Scatter(x=user_perf['User'], y=user_perf['Approval_Rate'],
                                   name="Approval %", mode='lines+markers', 
                                   line=dict(color='#10b981', width=3)), secondary_y=True)
            
            fig.update_layout(title="CA Performance: Volume vs Approval Rate")
            fig.update_xaxes(tickangle=-45)
            fig.update_yaxes(title_text="Total Applications", secondary_y=False)
            fig.update_yaxes(title_text="Approval Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("⚖️ Comparative Analysis")
    
    comp_tabs = st.tabs(["OSPH vs Pekerjaan", "OSPH vs Vehicle", "Product vs Branch", "Time Series Comparison"])
    
    with comp_tabs[0]:
        if all(col in df_filtered.columns for col in ['OSPH_Category', 'Pekerjaan', 'Scoring_Group']):
            comparison_df = df_filtered.groupby(['OSPH_Category', 'Pekerjaan']).apply(
                lambda x: pd.Series({
                    'Count': len(x),
                    'Approval_Rate': (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
                })
            ).reset_index()
            
            fig = px.scatter(comparison_df, x='OSPH_Category', y='Approval_Rate', 
                           size='Count', color='Pekerjaan',
                           title="OSPH vs Pekerjaan: Approval Rate & Volume",
                           hover_data=['Count'])
            st.plotly_chart(fig, use_container_width=True)
    
    with comp_tabs[1]:
        if all(col in df_filtered.columns for col in ['OSPH_Category', 'JenisKendaraan', 'Scoring_Group']):
            vehicle_comp = df_filtered.groupby(['OSPH_Category', 'JenisKendaraan']).apply(
                lambda x: pd.Series({
                    'Count': len(x),
                    'Approval_Rate': (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
                })
            ).reset_index()
            
            fig = px.bar(vehicle_comp, x='OSPH_Category', y='Approval_Rate', 
                       color='JenisKendaraan', barmode='group',
                       title="Approval Rate: OSPH Category by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
    
    with comp_tabs[2]:
        if all(col in df_filtered.columns for col in ['Produk', 'branch_name', 'Scoring_Group']):
            prod_branch = df_filtered.groupby(['Produk', 'branch_name'])['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).reset_index(name='Approval_Rate')
            
            fig = px.bar(prod_branch, x='branch_name', y='Approval_Rate', 
                       color='Produk', barmode='group',
                       title="Approval Rate by Product & Branch")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with comp_tabs[3]:
        if all(col in df_filtered.columns for col in ['YearMonth', 'Produk', 'Scoring_Group']):
            time_comp = df_filtered.groupby(['YearMonth', 'Produk'])['Scoring_Group'].apply(
                lambda x: (x == 'APPROVE').sum() / len(x) * 100
            ).reset_index(name='Approval_Rate')
            
            fig = px.line(time_comp, x='YearMonth', y='Approval_Rate', 
                        color='Produk', markers=True,
                        title="Time Series: Approval Rate by Product")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>📊 Credit Analyst Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
