import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

tab_names = [
    "🎯 Objectives",
    "💡 How to Use",
    "📊 Data Visualisation",
    "🤖 Classification",
    "🧩 Clustering",
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

# ---- OBJECTIVES TAB ----
with tabs[0]:
    st.markdown("## 🎯 Dashboard Objectives")
    st.markdown("""
**This dashboard helps you:**
- Predict Customer Churn (retain clients)
- Estimate Satisfaction Scores (focus on at-risk clients)
- Segment Customers for Offers (target personas)
- Find High Retention Patterns (build loyalty)
- Quantify FinAdvisor's impact (business case for tech)
_Navigate tabs above to explore each goal!_
""")

# ---- HOW TO USE TAB ----
with tabs[1]:
    st.markdown("## 💡 How to Use This Dashboard")
    st.markdown("""
**Steps:**
1. Upload your Excel data (`Cleaned data` sheet).
2. Set filters for your segment.
3. Use analysis tabs to explore insights.
4. Download results for presentations or action.
""")

# ---- DATA VISUALISATION TAB ----
with tabs[2]:
    st.header("📊 Data Visualisation")
    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique())) if 'Gender' in df.columns else []
            account_type = st.multiselect("Account Type", options=df['Account_Type'].unique(), default=list(df['Account_Type'].unique())) if 'Account_Type' in df.columns else []
        with col2:
            region = st.multiselect("Region", options=df['Region'].unique(), default=list(df['Region'].unique())) if 'Region' in df.columns else []
            marital_status = st.multiselect("Marital Status", options=df['Marital_Status'].unique(), default=list(df['Marital_Status'].unique())) if 'Marital_Status' in df.columns else []
        with col3:
            min_age, max_age = (int(df['Age'].min()), int(df['Age'].max())) if 'Age' in df.columns else (0, 100)
            age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age)) if 'Age' in df.columns else (0, 100)
            min_income, max_income = (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())) if 'Annual_Income' in df.columns else (0, 1000000)
            income_range = st.slider("Annual Income Range", min_income, max_income, (min_income, max_income)) if 'Annual_Income' in df.columns else (0, 1000000)

        filtered_df = df.copy()
        if 'Gender' in df.columns:
            filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
        if 'Account_Type' in df.columns:
            filtered_df = filtered_df[filtered_df['Account_Type'].isin(account_type)]
        if 'Region' in df.columns:
            filtered_df = filtered_df[filtered_df['Region'].isin(region)]
        if 'Marital_Status' in df.columns:
            filtered_df = filtered_df[filtered_df['Marital_Status'].isin(marital_status)]
        if 'Age' in df.columns:
            filtered_df = filtered_df[filtered_df['Age'].between(*age_range)]
        if 'Annual_Income' in df.columns:
            filtered_df = filtered_df[filtered_df['Annual_Income'].between(*income_range)]

        st.success(f"Filtered records: **{len(filtered_df)}**")

        import plotly.express as px
        import matplotlib.pyplot as plt
        import seaborn as sns

        # KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Churn Rate (%)", f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df.columns else "N/A")
        kpi2.metric("Avg. Satisfaction", f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df.columns else "N/A")
        kpi3.metric("Avg. Account Balance", f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df.columns else "N/A")

        # 1. Churn Rate by Account Type
        if 'Account_Type' in filtered_df.columns and 'Churn_Label' in filtered_df.columns:
            st.subheader("1. Churn Rate by Account Type")
            churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
            fig_cr = px.bar(churn_rate, x='Account_Type', y='Churn_Label', color='Churn_Label', text_auto='.2%', color_continuous_scale="Reds")
            fig_cr.update_layout(showlegend=False, yaxis_title="Churn Rate")
            st.plotly_chart(fig_cr, use_container_width=True)
            st.caption("Accounts with higher churn rates should be targeted with retention offers and outreach campaigns.")

        # 2. Average Account Balance by Region
        if 'Region' in filtered_df.columns and 'Account_Balance' in filtered_df.columns:
            st.subheader("2. Average Account Balance by Region")
            region_balance = filtered_df.groupby('Region')['Account_Balance'].mean().reset_index().sort_values("Account_Balance")
            fig_ab = px.bar(region_balance, x='Region', y='Account_Balance', text_auto='.2s', color='Account_Balance', color_continuous_scale="Blues")
            st.plotly_chart(fig_ab, use_container_width=True)
            st.caption("Regions with lower average balances may indicate untapped growth opportunities.")

        # 3. Churn by Age Group
        if 'Age' in filtered_df.columns and 'Churn_Label' in filtered_df.columns:
            st.subheader("3. Churn Rate by Age Group")
            bins = [17, 25, 35, 45, 55, 65, 80]
            labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, include_lowest=True)
            churn_by_age = filtered_df.groupby('Age_Group')['Churn_Label'].mean().reset_index()
            fig_cage = px.line(churn_by_age, x='Age_Group', y='Churn_Label', markers=True)
            fig_cage.update_traces(line_color='red')
            st.plotly_chart(fig_cage, use_container_width=True)
            st.caption("Young adults and seniors may have unique churn drivers—customized products can help retention.")

        # 4. Satisfaction by Account Type
        if 'Account_Type' in filtered_df.columns and 'Customer_Satisfaction_Score' in filtered_df.columns:
            st.subheader("4. Customer Satisfaction by Account Type")
            satisfaction = filtered_df.groupby('Account_Type')['Customer_Satisfaction_Score'].mean().reset_index()
            fig_sat = px.bar(satisfaction, x='Account_Type', y='Customer_Satisfaction_Score', color='Customer_Satisfaction_Score', text_auto='.2f', color_continuous_scale="Greens")
            st.plotly_chart(fig_sat, use_container_width=True)
            st.caption("Account types with low satisfaction need improved support or new value propositions.")

        # 5. Loan Amount Distribution by Loan Type
        if 'Loan_Type' in filtered_df.columns and 'Loan_Amount' in filtered_df.columns:
            st.subheader("5. Loan Amount Distribution by Loan Type")
            loan_dist = filtered_df.groupby('Loan_Type')['Loan_Amount'].sum().reset_index().sort_values("Loan_Amount", ascending=False)
            fig_loan = px.bar(loan_dist, x='Loan_Type', y='Loan_Amount', text_auto='.2s', color='Loan_Amount', color_continuous_scale="Viridis")
            st.plotly_chart(fig_loan, use_container_width=True)
            st.caption("Monitor loan types with large balances for risk concentration.")

        # 6. Credit Score: Churned vs. Non-Churned
        if 'Churn_Label' in filtered_df.columns and 'Credit_Score' in filtered_df.columns:
            st.subheader("6. Credit Score Distribution: Churned vs. Non-Churned")
            fig_box = px.box(filtered_df, x='Churn_Label', y='Credit_Score', color='Churn_Label',
                            labels={'Churn_Label': 'Churned'}, points="all")
            fig_box.update_xaxes(tickvals=[0, 1], ticktext=['Not Churned', 'Churned'])
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("Lower credit scores may be associated with higher churn—risk models should be updated accordingly.")

        # 7. Customer Count by Branch
        if 'Branch' in filtered_df.columns:
            st.subheader("7. Top 10 Branches by Customer Count")
            top_branches = filtered_df['Branch'].value_counts().head(10).reset_index()
            top_branches.columns = ['Branch', 'Count']
            fig_br = px.bar(top_branches, x='Branch', y='Count', color='Count', color_continuous_scale="teal")
            st.plotly_chart(fig_br, use_container_width=True)
            st.caption("Branches with high customer counts can be leveraged for cross-selling; underperformers may need marketing.")

        # 8. Transaction Type Pie Chart (Plotly)
        if 'Transaction_Type' in filtered_df.columns:
            st.subheader("8. Transaction Type Distribution")
            trx_dist = filtered_df['Transaction_Type'].value_counts().reset_index()
            trx_dist.columns = ['Transaction Type', 'Count']
            fig3 = px.pie(
                trx_dist,
                names='Transaction Type',
                values='Count',
                title='Transaction Type Distribution',
                hole=0.3
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Observe the share of each transaction type to optimize digital and branch resources.")

        # 9. Monthly Transaction Amount Trend
        if 'Transaction_Date' in filtered_df.columns and 'Transaction_Amount' in filtered_df.columns:
            st.subheader("9. Monthly Transaction Amount Trend")
            filtered_df['Transaction_Month'] = pd.to_datetime(filtered_df['Transaction_Date']).dt.to_period('M').astype(str)
            monthly_trx = filtered_df.groupby('Transaction_Month')['Transaction_Amount'].sum().reset_index()
            fig_mt = px.line(monthly_trx, x='Transaction_Month', y='Transaction_Amount', markers=True)
            st.plotly_chart(fig_mt, use_container_width=True)
            st.caption("Monitor trends for seasonality or anomalies in transactions, for fraud or marketing planning.")

        # 10. Correlation Heatmap of Key Numeric Variables
        numeric_cols = filtered_df.select_dtypes(include='number').drop(columns=['Churn_Label'], errors='ignore')
        if len(numeric_cols.columns) > 1:
            st.subheader("10. Correlation Heatmap")
            corr = numeric_cols.corr()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
            st.pyplot(fig2)
            st.caption("High correlation between features can indicate redundant information or risk factors.")

    except Exception as e:
        st.error(f"Data Visualisation failed: {e}")

# ---- CLASSIFICATION TAB ----
# (You can keep the classification code from the previous complete answer!)

# ---- CLUSTERING TAB ----
# (You can keep your clustering code from the previous complete answer!)

# ---- ASSOCIATION RULES TAB ----
# (You can keep your association rule code from the previous complete answer!)

# ---- REGRESSION TAB (RandomForest+GBRT) ----
with tabs[6]:
    st.header("📈 Regression (RandomForest vs. GradientBoosting)")
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        regression_targets = [c for c in ['Account_Balance', 'Annual_Income', 'Customer_Satisfaction_Score'] if c in df.columns]
        if not regression_targets:
            st.warning("No regression targets found.")
        else:
            target_reg = st.selectbox("Regression target", regression_targets)
            reg_drop_cols = [
                "Customer_ID", "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date",
                "Churn_Label", "Simulated_New_Churn_Label", "Churn_Timeframe"
            ] + regression_targets
            reg_features = [col for col in df.columns if col not in reg_drop_cols]
            if not reg_features:
                st.warning("No valid features for regression.")
            else:
                X_reg = df[reg_features].copy()
                y_reg = df[target_reg]
                # Robust encoding
                for col in X_reg.select_dtypes(include=['object', 'category']):
                    X_reg[col] = LabelEncoder().fit_transform(X_reg[col].astype(str))
                X_reg = X_reg.fillna(0)
                # Remove columns with only one unique value
                constant_cols = [c for c in X_reg.columns if X_reg[c].nunique() == 1]
                if constant_cols:
                    X_reg = X_reg.drop(columns=constant_cols)
                if X_reg.shape[1] == 0:
                    st.error("No valid features after encoding. Add more varied columns to your data.")
                else:
                    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)
                    # Random Forest
                    rf = RandomForestRegressor(n_estimators=200, random_state=42)
                    rf.fit(Xr_train, yr_train)
                    y_rf = rf.predict(Xr_test)
                    # Gradient Boosting
                    gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
                    gb.fit(Xr_train, yr_train)
                    y_gb = gb.predict(Xr_test)
                    # Metrics
                    data_metrics = {
                        "Model": ["Random Forest", "Gradient Boosting"],
                        "R²": [r2_score(yr_test, y_rf), r2_score(yr_test, y_gb)],
                        "MAE": [mean_absolute_error(yr_test, y_rf), mean_absolute_error(yr_test, y_gb)],
                        "RMSE": [np.sqrt(mean_squared_error(yr_test, y_rf)), np.sqrt(mean_squared_error(yr_test, y_gb))]
                    }
                    st.dataframe(pd.DataFrame(data_metrics).set_index("Model").style.format("{:.2f}"))
                    reg_select = st.selectbox("Select Model for Feature Importances", ["Random Forest", "Gradient Boosting"])
                    if reg_select == "Random Forest":
                        importances = pd.Series(rf.feature_importances_, index=X_reg.columns)
                        st.subheader("Random Forest Feature Importances")
                        st.bar_chart(importances.sort_values(ascending=False).head(10))
                    else:
                        importances = pd.Series(gb.feature_importances_, index=X_reg.columns)
                        st.subheader("Gradient Boosting Feature Importances")
                        st.bar_chart(importances.sort_values(ascending=False).head(10))
                    st.caption("Use the most important features to drive targeted product development and risk models.")
    except Exception as e:
        st.error(f"Regression failed: {e}")

# ---- TIME SERIES TAB ----
# (Keep your time series code as before!)

st.markdown("---\n*If a feature doesn't show up, it's because your data doesn't have the required columns.*")
