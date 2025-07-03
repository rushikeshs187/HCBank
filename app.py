# --- ORIGINAL DASHBOARD CODE ---

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

# ----- TAB LAYOUT -----
tab_names = [
    "🌟 Objectives",
    "💡 How to Use",
    "📊 Data Visualisation",
    "🤖 Classification",
    "🯩 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "⏳ Time Series Trends"
]
tabs = st.tabs(tab_names)

# ----- SIDEBAR -----
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (with 'Cleaned data' sheet)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload data\n2. Explore tabs\n3. Download insights!", icon="ℹ️")

# ---- DATA LOADING ----
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    st.warning("📁 Please upload your Excel file to unlock dashboard features.", icon="⚠️")
    st.stop()

df = st.session_state['df']

# --- UPDATED LIGHT-MODE + SIDEBAR FILTER VERSION STARTS HERE ---

# Streamlit enhancements in progress.
# Core changes: filters moved to sidebar, light theme enforced, reactivity improved.

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")
    gender = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
    account_type = st.multiselect("Account Type", options=df['Account_Type'].unique(), default=list(df['Account_Type'].unique()))
    region = st.multiselect("Region", options=df['Region'].unique(), default=list(df['Region'].unique()))
    marital_status = st.multiselect("Marital Status", options=df['Marital_Status'].unique(), default=list(df['Marital_Status'].unique()))

    age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.slider("Annual Income", int(df['Annual_Income'].min()), int(df['Annual_Income'].max()), (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())))

# Apply filters
filtered_df = df[
    df['Gender'].isin(gender) &
    df['Account_Type'].isin(account_type) &
    df['Region'].isin(region) &
    df['Marital_Status'].isin(marital_status) &
    df['Age'].between(*age_range) &
    df['Annual_Income'].between(*income_range)
]

st.title(":bar_chart: Bank Customer Analytics Dashboard")
st.markdown("Filtered Data Records: **{}**".format(len(filtered_df)))

# Simple summary section
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Churn Rate (%)", f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df.columns else "N/A")
kpi2.metric("Avg. Satisfaction", f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df.columns else "N/A")
kpi3.metric("Avg. Account Balance", f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df.columns else "N/A")

st.divider()
st.subheader("Coming Next: Reactivity + Visualisation tabs refactoring")

# Note: Full dashboard code refactor will continue based on this foundation.
