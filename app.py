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

        # ... (You can keep/add more visualisations as before!) ...

    except Exception as e:
        st.error(f"Data Visualisation failed: {e}")

# ---- CLASSIFICATION TAB ----
with tabs[3]:
    st.header("🤖 Churn Prediction: Model Comparison")
    try:
        drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 'Churn_Timeframe', 'Simulated_New_Churn_Label']
        target = 'Churn_Label'
        features = [col for col in df.columns if col not in drop_cols + [target]]
        if target not in df.columns or len(features) < 1:
            st.warning("Not enough features or missing Churn_Label for classification.")
        else:
            X = df[features].copy()
            y = df[target]
            for col in X.select_dtypes(include=['object', 'category']):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.fillna(0)
            # Remove constant columns
            constant_cols = [c for c in X.columns if X[c].nunique() == 1]
            if constant_cols: X = X.drop(columns=constant_cols)
            if X.shape[1] == 0:
                st.error("No valid features after encoding. Add more varied columns to your data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y)
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

                models = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "KNN": KNeighborsClassifier()
                }

                results = []
                probs = {}
                for name, mdl in models.items():
                    mdl.fit(X_train, y_train)
                    pred = mdl.predict(X_test)
                    acc = accuracy_score(y_test, pred)
                    prec = precision_score(y_test, pred, average='macro', zero_division=0)
                    rec = recall_score(y_test, pred, average='macro', zero_division=0)
                    f1 = f1_score(y_test, pred, average='macro', zero_division=0)
                    results.append(dict(Model=name, Accuracy=acc, Precision=prec, Recall=rec, F1=f1))
                    # For ROC/AUC
                    if hasattr(mdl, "predict_proba"):
                        probs[name] = mdl.predict_proba(X_test)[:,1]
                    else:
                        # Some classifiers (like KNN with 1 class) may not have predict_proba; handle gracefully
                        probs[name] = np.zeros_like(y_test)
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}"
                }), height=180)

                # Model selection for Confusion Matrix & ROC
                model_select = st.selectbox(
                    "Select Model for Details (Confusion Matrix & ROC)",
                    list(models.keys()), key="clf_model_select"
                )
                model = models[model_select]
                pred = model.predict(X_test)
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, pred)
                cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Pred: Not Churned", "Pred: Churned"])
                st.dataframe(cm_df)

                st.subheader("ROC Curve")
                fig, ax = plt.subplots()
                for name, prob in probs.items():
                    fpr, tpr, _ = roc_curve(y_test, prob)
                    auc_val = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
                ax.plot([0,1],[0,1],"k--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves")
                ax.legend()
                st.pyplot(fig)

                # Feature Importances for tree-based models
                if model_select in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                    st.subheader("Top Feature Importances")
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    st.bar_chart(importances.sort_values(ascending=False).head(10))

    except Exception as e:
        st.error(f"Classification failed: {e}")

# ---- CLUSTERING TAB ----
with tabs[4]:
    st.header("🧩 Customer Clustering")
    try:
        from sklearn.cluster import KMeans
        numeric_exclude = [
            "Customer_ID", "Churn_Label", "Simulated_New_Churn_Label",
            "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date", "Churn_Timeframe"
        ]
        cluster_features = df.select_dtypes(include=['number']).drop(columns=numeric_exclude, errors="ignore").columns.tolist()
        if len(cluster_features) < 2:
            st.warning("Not enough numeric features for clustering.")
        else:
            scaler = StandardScaler()
            X_cluster = scaler.fit_transform(df[cluster_features])
            k = st.slider("Select clusters (k)", min_value=2, max_value=200, value=5, step=1)
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_cluster)
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = cluster_labels
            # Elbow chart only up to k=20 for visual clarity
            inertias = []
            elbow_range = range(2, min(21, len(df)))
            for ki in elbow_range:
                km = KMeans(n_clusters=ki, random_state=42)
                km.fit(X_cluster)
                inertias.append(km.inertia_)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(list(elbow_range), inertias, marker="o")
            ax.set_xlabel("Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Chart (up to 20 clusters)")
            st.pyplot(fig)
            persona = df_with_clusters.groupby('Cluster')[cluster_features].mean().round(2)
            st.dataframe(persona)
            st.download_button(
                label="Download Cluster Data (CSV)",
                data=df_with_clusters.to_csv(index=False).encode("utf-8"),
                file_name="clustered_customers.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ---- ASSOCIATION RULES TAB ----
with tabs[5]:
    st.header("🔗 Association Rule Mining")
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns for association rule mining.")
        else:
            apriori_cols = st.multiselect("Select 2+ categorical columns:", options=cat_cols, default=cat_cols[:2])
            min_support = st.slider("Min Support", 0.01, 0.2, 0.05, step=0.01)
            min_conf = st.slider("Min Confidence", 0.01, 1.0, 0.3, step=0.01)
            min_lift = st.slider("Min Lift", 1.0, 5.0, 1.2, step=0.1)
            if len(apriori_cols) >= 2:
                encoded_df = pd.get_dummies(df[apriori_cols].astype(str))
                freq_items = apriori(encoded_df, min_support=min_support, use_colnames=True)
                rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
                rules = rules[rules["lift"] >= min_lift]
                rules = rules.sort_values("confidence", ascending=False).head(10)
                if not rules.empty:
                    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(rules[display_cols])
                else:
                    st.warning("No rules found. Try different columns or lower thresholds.")
    except Exception as e:
        st.error(f"Association rules failed: {e}")

# ---- REGRESSION TAB (RandomForest) ----
with tabs[6]:
    st.header("📈 Regression (Satisfaction & Value) [RandomForest]")
    try:
        from sklearn.ensemble import RandomForestRegressor
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
                    reg = RandomForestRegressor(n_estimators=200, random_state=42)
                    reg.fit(Xr_train, yr_train)
                    y_pred = reg.predict(Xr_test)
                    st.metric("R²", f"{r2_score(yr_test, y_pred):.2f}")
                    st.metric("MAE", f"{mean_absolute_error(yr_test, y_pred):.2f}")
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(yr_test, y_pred)):.2f}")
                    # Show top driver importances
                    st.subheader("Top Regression Feature Importances")
                    importances = pd.Series(reg.feature_importances_, index=X_reg.columns)
                    st.bar_chart(importances.sort_values(ascending=False).head(10))
                    st.caption("Tip: Try removing low-importance features or engineering new ones for better R².")
    except Exception as e:
        st.error(f"Regression failed: {e}")

# ---- TIME SERIES TAB ----
with tabs[7]:
    st.header("⏳ Time Series Trends")
    try:
        if 'Transaction_Date' in df.columns:
            df['Transaction_Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M').astype(str)
            metric_cols = [col for col in ['Transaction_Amount', 'Account_Balance', 'Customer_Satisfaction_Score'] if col in df.columns]
            if metric_cols:
                monthly_metrics = df.groupby('Transaction_Month')[metric_cols].mean().reset_index()
                import plotly.express as px
                fig = px.line(monthly_metrics, x='Transaction_Month', y=metric_cols, markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric metrics found for time series.")
        else:
            st.warning("Transaction_Date column not found for time series analysis.")
    except Exception as e:
        st.error(f"Time Series Analysis failed: {e}")

st.markdown("---\n*If a feature doesn't show up, it's because your data doesn't have the required columns.*")
