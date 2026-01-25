import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
st.set_page_config(page_title="Credit Analyst Analytics Dashboard", layout="wide", page_icon="📊")

# Custom CSS untuk styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Daftar tanggal merah (hardcoded)
TANGGAL_MERAH = [
    "01-01-2025", "27-01-2025", "28-01-2025", "29-01-2025",
    "28-03-2025", "31-03-2025", "01-04-2025", "02-04-2025",
    "03-04-2025", "04-04-2025", "07-04-2025", "18-04-2025",
    "01-05-2025", "12-05-2025", "29-05-2025", "06-06-2025",
    "09-06-2025", "27-06-2025", "18-08-2025", "05-09-2025",
    "25-12-2025", "26-12-2025", "31-12-2025",
    "01-01-2026", "02-01-2026", "16-01-2026", "16-02-2026",
    "17-02-2026", "18-03-2026", "19-03-2026", "20-03-2026",
    "23-03-2026", "24-03-2026", "03-04-2026", "01-05-2026",
    "14-05-2026", "27-05-2026", "28-05-2026", "01-06-2026",
    "16-06-2026", "17-08-2026", "25-08-2026",
    "25-12-2026", "31-12-2026"
]

TANGGAL_MERAH_DT = [datetime.strptime(d, "%d-%m-%Y").date() for d in TANGGAL_MERAH]

# Mapping OSPH Range
OSPH_RANGES = [
    (0, 250000000, "0 - 250 Juta"),
    (250000001, 500000000, "250 - 500 Juta"),
    (500000001, float('inf'), "500 Juta+")
]

# ========== FUNGSI ANALYTICS ==========

def parse_date(date_str):
    """Parse berbagai format tanggal"""
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
    """Cek apakah hari kerja"""
    if date.weekday() >= 5:
        return False
    if date.date() in TANGGAL_MERAH_DT:
        return False
    return True

def calculate_sla_days(start_dt, end_dt):
    """Hitung SLA dalam hari kerja dengan logika jam kerja"""
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
    """Kategorikan nilai OSPH"""
    if pd.isna(osph_value):
        return "Unknown"
    
    osph_value = float(osph_value)
    for min_val, max_val, label in OSPH_RANGES:
        if min_val <= osph_value <= max_val:
            return label
    return "Unknown"

def preprocess_data(df):
    """Preprocessing dan feature engineering"""
    df = df.copy()
    
    # Parse tanggal
    date_columns = ['action_on', 'Initiation', 'RealisasiDate']
    for col in date_columns:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(parse_date)
    
    # Hitung SLA
    if 'action_on_parsed' in df.columns and 'RealisasiDate_parsed' in df.columns:
        df['SLA_Days'] = df.apply(
            lambda row: calculate_sla_days(row['action_on_parsed'], row['RealisasiDate_parsed']),
            axis=1
        )
    
    # Kategorikan OSPH
    if 'Outstanding_Principal' in df.columns:
        df['Outstanding_Principal_clean'] = pd.to_numeric(
            df['Outstanding_Principal'].astype(str).str.replace(',', ''), 
            errors='coerce'
        )
        df['OSPH_Category'] = df['Outstanding_Principal_clean'].apply(get_osph_category)
    
    # Standarisasi Hasil Scoring
    if 'Hasil_Scoring_1' in df.columns:
        df['Hasil_Scoring_Clean'] = df['Hasil_Scoring_1'].fillna('-')
        df['Scoring_Group'] = df['Hasil_Scoring_Clean'].apply(lambda x: 
            'APPROVE' if 'APPROVE' in str(x).upper() else
            'REGULER' if 'REGULER' in str(x).upper() else
            'REJECT' if 'REJECT' in str(x).upper() else
            'IN PROGRESS' if 'PROGRESS' in str(x).upper() else
            'OTHER'
        )
    
    # Feature Engineering: Waktu
    if 'action_on_parsed' in df.columns:
        df['Hour_of_Day'] = df['action_on_parsed'].dt.hour
        df['Day_of_Week'] = df['action_on_parsed'].dt.dayofweek
        df['Month'] = df['action_on_parsed'].dt.month
        df['Week_of_Year'] = df['action_on_parsed'].dt.isocalendar().week
    
    # Risk Score (composite metric)
    if 'Outstanding_Principal_clean' in df.columns and 'SLA_Days' in df.columns:
        # Normalize dan create risk score
        osph_normalized = (df['Outstanding_Principal_clean'] - df['Outstanding_Principal_clean'].min()) / \
                         (df['Outstanding_Principal_clean'].max() - df['Outstanding_Principal_clean'].min())
        sla_normalized = (df['SLA_Days'] - df['SLA_Days'].min()) / \
                        (df['SLA_Days'].max() - df['SLA_Days'].min())
        df['Risk_Score'] = (osph_normalized * 0.6 + sla_normalized * 0.4) * 100
    
    return df

def perform_statistical_analysis(df, group_col, target_col):
    """Perform statistical tests (ANOVA/Chi-Square)"""
    try:
        groups = df.groupby(group_col)[target_col].apply(list)
        
        # ANOVA test
        f_stat, p_value = stats.f_oneway(*groups)
        
        result = {
            'test': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        return result
    except:
        return None

def calculate_conversion_rates(df):
    """Calculate conversion rates across different dimensions"""
    results = {}
    
    if 'Scoring_Group' in df.columns:
        # Overall conversion
        total = len(df)
        approve = (df['Scoring_Group'] == 'APPROVE').sum()
        results['overall_approval_rate'] = (approve / total * 100) if total > 0 else 0
        
        # By OSPH Category
        if 'OSPH_Category' in df.columns:
            osph_conv = df.groupby('OSPH_Category').apply(
                lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            results['osph_conversion'] = osph_conv
        
        # By Product
        if 'Product' in df.columns:
            prod_conv = df.groupby('Product').apply(
                lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            results['product_conversion'] = prod_conv
        
        # By Pekerjaan
        if 'Pekerjaan' in df.columns:
            job_conv = df.groupby('Pekerjaan').apply(
                lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
            ).to_dict()
            results['job_conversion'] = job_conv
    
    return results

def build_predictive_model(df):
    """Build simple predictive model for scoring outcome"""
    try:
        # Select features
        feature_cols = []
        if 'Outstanding_Principal_clean' in df.columns:
            feature_cols.append('Outstanding_Principal_clean')
        if 'SLA_Days' in df.columns:
            feature_cols.append('SLA_Days')
        
        # Encode categorical
        categorical_cols = ['Product', 'Pekerjaan', 'Jenis_Kendaraan', 'OSPH_Category']
        le_dict = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
                feature_cols.append(f'{col}_encoded')
        
        # Prepare data
        if 'Scoring_Group' not in df.columns or len(feature_cols) == 0:
            return None
        
        df_model = df[df['Scoring_Group'].isin(['APPROVE', 'REJECT', 'REGULER'])].copy()
        
        if len(df_model) < 50:  # Minimum samples
            return None
        
        X = df_model[feature_cols].fillna(0)
        y = df_model['Scoring_Group']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).sum() / len(y_test) * 100
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': importance_df,
            'encoders': le_dict
        }
    except Exception as e:
        st.error(f"Model building error: {str(e)}")
        return None

def cohort_analysis(df):
    """Perform cohort analysis based on application month"""
    if 'action_on_parsed' not in df.columns or 'Scoring_Group' not in df.columns:
        return None
    
    df_cohort = df.copy()
    df_cohort['Cohort'] = df_cohort['action_on_parsed'].dt.to_period('M')
    
    # Approval rate by cohort
    cohort_approval = df_cohort.groupby('Cohort').apply(
        lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
    ).reset_index(name='Approval_Rate')
    
    cohort_approval['Cohort'] = cohort_approval['Cohort'].astype(str)
    
    return cohort_approval

def segment_analysis(df):
    """Advanced segmentation using multiple dimensions"""
    segments = {}
    
    # High-risk segment: High OSPH + Long SLA
    if 'OSPH_Category' in df.columns and 'SLA_Days' in df.columns:
        high_risk = df[
            (df['OSPH_Category'] == '500 Juta+') & 
            (df['SLA_Days'] > df['SLA_Days'].median())
        ]
        segments['high_risk'] = {
            'count': len(high_risk),
            'approval_rate': (high_risk['Scoring_Group'] == 'APPROVE').sum() / len(high_risk) * 100 if len(high_risk) > 0 else 0
        }
    
    # VIP segment: High OSPH + Quick processing
    if 'OSPH_Category' in df.columns and 'SLA_Days' in df.columns:
        vip = df[
            (df['OSPH_Category'] == '500 Juta+') & 
            (df['SLA_Days'] <= df['SLA_Days'].quantile(0.25))
        ]
        segments['vip'] = {
            'count': len(vip),
            'approval_rate': (vip['Scoring_Group'] == 'APPROVE').sum() / len(vip) * 100 if len(vip) > 0 else 0
        }
    
    # Standard segment
    if 'OSPH_Category' in df.columns:
        standard = df[df['OSPH_Category'] == '0 - 250 Juta']
        segments['standard'] = {
            'count': len(standard),
            'approval_rate': (standard['Scoring_Group'] == 'APPROVE').sum() / len(standard) * 100 if len(standard) > 0 else 0
        }
    
    return segments

# ========== LOAD DATA ==========

@st.cache_data
def load_data():
    """Load data dengan sample yang lebih realistis"""
    np.random.seed(42)
    n_samples = 500
    
    sample_data = {
        'apps_id': range(4760000, 4760000 + n_samples),
        'position_name': ['CREDIT ANALYST'] * n_samples,
        'user_name': np.random.choice([
            'Iman Eko Ardianto', 'TAN IRWAN LAXMANA', 'Demastiana Saputri',
            'Karyawan User 1', 'Karyawan User 2'
        ], n_samples),
        'apps_status': np.random.choice([
            'RECOMMENDED CCS NEW', 'PENDING CA', 'NOT RECOMMENDED CA',
            'RECOMMENDED CA', 'RECOMMENDED CA WITH COND', 'Pending CA Completed'
        ], n_samples, p=[0.3, 0.2, 0.15, 0.2, 0.1, 0.05]),
        'Product': np.random.choice(['CS NEW', 'CS USED'], n_samples, p=[0.6, 0.4]),
        'action_on': pd.date_range(start='2024-11-01', periods=n_samples, freq='2H'),
        'Initiation': pd.date_range(start='2024-10-20', periods=n_samples, freq='2H'),
        'RealisasiDate': pd.date_range(start='2024-11-02', periods=n_samples, freq='3H'),
        'Outstanding_Principal': np.random.choice([
            120000000, 180000000, 250000000, 350000000, 
            420000000, 550000000, 700000000
        ], n_samples),
        'Pekerjaan': np.random.choice([
            'Karyawan', 'Wiraswasta', 'Profesional', 'Pengusaha'
        ], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Jabatan': np.random.choice([
            'STAFF OPERASIONAL', 'SALES EXECUTIVE', 'PEMILIK', 
            'DIREKTUR', 'MANAGER', 'SUPERVISOR'
        ], n_samples),
        'Jenis_Kendaraan': np.random.choice([
            'Mb. Penumpang', 'Mb. Beban'
        ], n_samples, p=[0.7, 0.3]),
        'Hasil_Scoring_1': np.random.choice([
            'REJECT', 'Reject 1', 'REGULER', 'Reguler 1', 
            'APPROVE', 'Approve 1', 'Scoring in Progress', '-'
        ], n_samples, p=[0.15, 0.05, 0.25, 0.1, 0.25, 0.1, 0.05, 0.05]),
        'branch_name': np.random.choice([
            'PEJAJALAN S2P', 'KARAWANG MOBIL', 'KKB DARMO',
            'BANDUNG MOBIL', 'JAKARTA TIMUR', 'CILEGON'
        ], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    return preprocess_data(df)

# ========== MAIN APP ==========

def main():
    st.title("🎯 Credit Analyst Analytics Dashboard")
    st.markdown("**Advanced Analytics & Predictive Insights**")
    st.markdown("---")
    
    # Load data
    with st.spinner("🔄 Loading and processing data..."):
        df = load_data()
    
    # ========== SIDEBAR ==========
    st.sidebar.header("🔍 Analytics Filters")
    
    # Filters
    products = ['All'] + sorted(df['Product'].unique().tolist())
    selected_product = st.sidebar.selectbox("📦 Produk", products)
    
    if 'action_on_parsed' in df.columns:
        min_date = df['action_on_parsed'].min().date()
        max_date = df['action_on_parsed'].max().date()
        date_range = st.sidebar.date_input(
            "📅 Periode Analisis",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    status_options = ['All'] + sorted(df['apps_status'].unique().tolist())
    selected_status = st.sidebar.selectbox("📊 Status", status_options)
    
    # Analytics options
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analytics Options")
    show_predictions = st.sidebar.checkbox("🤖 Enable Predictive Model", value=True)
    show_statistical = st.sidebar.checkbox("📈 Show Statistical Tests", value=True)
    confidence_level = st.sidebar.slider("Confidence Level (%)", 90, 99, 95)
    
    # Apply filters
    df_filtered = df.copy()
    if selected_product != 'All':
        df_filtered = df_filtered[df_filtered['Product'] == selected_product]
    if selected_status != 'All':
        df_filtered = df_filtered[df_filtered['apps_status'] == selected_status]
    if 'action_on_parsed' in df.columns and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['action_on_parsed'].dt.date >= date_range[0]) &
            (df_filtered['action_on_parsed'].dt.date <= date_range[1])
        ]
    
    # ========== EXECUTIVE SUMMARY ==========
    st.header("📊 Executive Summary & Key Insights")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_apps = df_filtered['apps_id'].nunique()
        prev_period_apps = len(df) // 2  # Simplified
        growth = ((total_apps - prev_period_apps) / prev_period_apps * 100) if prev_period_apps > 0 else 0
        st.metric("📝 Total Applications", f"{total_apps:,}", f"{growth:+.1f}%")
    
    with col2:
        if 'SLA_Days' in df_filtered.columns:
            avg_sla = df_filtered['SLA_Days'].mean()
            target_sla = 3.0
            sla_performance = ((target_sla - avg_sla) / target_sla * 100) if not pd.isna(avg_sla) else 0
            st.metric("⏱️ Avg SLA Days", f"{avg_sla:.1f}" if not pd.isna(avg_sla) else "N/A", 
                     f"{sla_performance:+.1f}% vs target")
    
    with col3:
        if 'Scoring_Group' in df_filtered.columns:
            approval_rate = (df_filtered['Scoring_Group'] == 'APPROVE').sum() / len(df_filtered) * 100
            st.metric("✅ Approval Rate", f"{approval_rate:.1f}%")
    
    with col4:
        if 'Scoring_Group' in df_filtered.columns:
            reject_rate = (df_filtered['Scoring_Group'] == 'REJECT').sum() / len(df_filtered) * 100
            st.metric("❌ Reject Rate", f"{reject_rate:.1f}%")
    
    with col5:
        if 'Risk_Score' in df_filtered.columns:
            avg_risk = df_filtered['Risk_Score'].mean()
            st.metric("⚠️ Avg Risk Score", f"{avg_risk:.1f}")
    
    # Key Insights Box
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    # Insight 1: Conversion by OSPH
    if 'OSPH_Category' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
        osph_approval = df_filtered.groupby('OSPH_Category').apply(
            lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
        ).to_dict()
        
        best_osph = max(osph_approval, key=osph_approval.get)
        worst_osph = min(osph_approval, key=osph_approval.get)
        
        insights.append(f"🎯 **Best performing OSPH segment**: {best_osph} with {osph_approval[best_osph]:.1f}% approval rate")
        insights.append(f"⚠️ **Lowest performing OSPH segment**: {worst_osph} with {osph_approval[worst_osph]:.1f}% approval rate")
    
    # Insight 2: SLA Performance
    if 'SLA_Days' in df_filtered.columns:
        fast_apps = (df_filtered['SLA_Days'] <= 2).sum()
        fast_pct = fast_apps / len(df_filtered) * 100
        insights.append(f"⚡ **{fast_pct:.1f}%** of applications processed within 2 working days")
    
    # Insight 3: Peak hours
    if 'Hour_of_Day' in df_filtered.columns:
        peak_hour = df_filtered['Hour_of_Day'].mode()[0]
        insights.append(f"🕐 **Peak submission hour**: {peak_hour}:00 - Consider resource allocation")
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Trend Analysis", 
        "💰 OSPH Deep Dive", 
        "⏱️ SLA Analytics",
        "🎯 Conversion Funnel",
        "🔮 Predictive Model",
        "👥 Segmentation",
        "📋 Raw Data"
    ])
    
    # ========== TAB 1: TREND ANALYSIS ==========
    with tab1:
        st.header("📈 Trend Analysis & Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume trend
            if 'action_on_parsed' in df_filtered.columns:
                df_trend = df_filtered.copy()
                df_trend['Date'] = df_trend['action_on_parsed'].dt.date
                trend_data = df_trend.groupby('Date').size().reset_index(name='Count')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trend_data['Date'], 
                    y=trend_data['Count'],
                    mode='lines+markers',
                    name='Applications',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                
                # Add trend line
                z = np.polyfit(range(len(trend_data)), trend_data['Count'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=trend_data['Date'],
                    y=p(range(len(trend_data))),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Application Volume Trend with Regression Line",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Approval rate trend
            if 'action_on_parsed' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
                df_approval_trend = df_filtered.copy()
                df_approval_trend['Date'] = df_approval_trend['action_on_parsed'].dt.date
                
                approval_trend = df_approval_trend.groupby('Date').apply(
                    lambda x: (x['Scoring_Group'] == 'APPROVE').sum() / len(x) * 100
                ).reset_index(name='Approval_Rate')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=approval_trend['Date'],
                    y=approval_trend['Approval_Rate'],
                    mode='lines+markers',
                    name='Approval Rate',
                    line=dict(color='#10b981', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.2)'
                ))
                
                # Add moving average
                if len(approval_trend) >= 7:
                    approval_trend['MA7'] = approval_trend['Approval_Rate'].rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=approval_trend['Date'],
                        y=approval_trend['MA7'],
                        mode='lines',
                        name='7-Day MA',
                        line=dict(color='orange', dash='dash')
                    ))
                
                fig.update_layout(
                    title="Approval Rate Trend with Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Approval Rate (%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        st.subheader("📅 Weekly Pattern Analysis")
        
        if 'Day_of_Week' in df_filtered.columns and 'Scoring_Group' in df_filtered.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            dow_analysis = df_filtered.groupby('Day_of_Week').agg({
                'apps_id': 'count',
                'Scoring_Group': lambda x: (x == 'APPROVE').sum() / len(x) * 100
            }).reset_index()
            
            dow_analysis['Day_Name'] = dow_analysis['Day_of_Week'].apply(lambda x: day_names[x])
            dow_analysis.columns = ['Day_of_Week', 'Volume', 'Approval_Rate', 'Day_Name']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=dow_analysis['Day_Name'], y=dow_analysis['Volume'], 
                      name="Volume", marker_color='#667eea'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=
