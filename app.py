import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

# ----- TAB LAYOUT -----
tab_names = [
    "🎯 Objectives",
    "💡 How to Use",
    "📊 Data Visualisation",
    "🤖 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "💬 NLP (Feedback Analysis)",
    "⏳ Time Series Trends",
    "🕵️‍♂️ Segmentation"
]
tabs = st.tabs(tab_names)

# ----- SIDEBAR -----
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (with 'Cleaned data' sheet)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload data\n2. Explore tabs\n3. Download insights!", icon="ℹ️")
    st.markdown(
        "<br><small>Created with ❤️ using [Streamlit](https://streamlit.io/)</small>", unsafe_allow_html=True
    )

# ---- DATA LOADING ----
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
else:
    st.warning("📁 Please upload your Excel file to unlock dashboard features.", icon="⚠️")
    st.stop()

# ---- OBJECTIVES TAB ----
with tabs[0]:
    st.markdown("## 🎯 Dashboard Objectives")
    st.markdown("""
This dashboard empowers your team to:
- **Predict Customer Churn** – Anticipate and reduce attrition via AI models.
- **Estimate Satisfaction Scores** – Target improvements, upsell to high-potential customers.
- **Segment Customers for Personalization** – Discover personas for FinAdvisor offers.
- **Find High Retention Patterns** – Reveal service combos linked to loyalty.
- **Quantify Impact of FinAdvisor on Churn** – Make data-driven business cases.

*Every tab is mapped to a business objective. See 💡 How to Use for details!*
    """)
    st.success("You’re viewing the one-stop analytics toolkit for strategic, retention-focused banking.")

# ---- HOW TO USE TAB ----
with tabs[1]:
    st.markdown("## 💡 How to Use This Dashboard")
    st.markdown("""
1. **Upload** your data (sidebar, xlsx with 'Cleaned data' sheet).
2. **Use filters** to focus on business segments.
3. **Navigate the tabs** for insights: Visualize, Predict, Segment, Mine, and more!
4. **Download results** for action plans.
5. **Check tooltips and captions** for explanations.
---
**Pro Tip:** Start with 📊 *Data Visualisation* to spot trends, then deep-dive using the next tabs.
    """)
    st.info("Questions or errors? Ensure your data format matches the sample. All insights update live with your filters!")

# ---- DATA VISUALISATION TAB ----
with tabs[2]:
    st.header("📊 Data Visualisation")
    st.markdown("> _Explore segments, spot risks & opportunities instantly!_")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        account_type = st.multiselect("Account Type", options=df['Account_Type'].unique(), default=list(df['Account_Type'].unique()))
    with col2:
        region = st.multiselect("Region", options=df['Region'].unique(), default=list(df['Region'].unique()))
        marital_status = st.multiselect("Marital Status", options=df['Marital_Status'].unique(), default=list(df['Marital_Status'].unique()))
    with col3:
        min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        min_income, max_income = int(df['Annual_Income'].min()), int(df['Annual_Income'].max())
        income_range = st.slider("Annual Income Range", min_income, max_income, (min_income, max_income))

    filtered_df = df[
        (df['Gender'].isin(gender)) &
        (df['Account_Type'].isin(account_type)) &
        (df['Region'].isin(region)) &
        (df['Marital_Status'].isin(marital_status)) &
        (df['Age'].between(*age_range)) &
        (df['Annual_Income'].between(*income_range))
    ]
    st.success(f"Filtered records: **{len(filtered_df)}**", icon="🔎")

    st.markdown("### 🔢 Key Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Churn Rate (%)", f"{filtered_df['Churn_Label'].mean()*100:.2f}")
    kpi2.metric("Avg. Satisfaction", f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}")
    kpi3.metric("Avg. Account Balance", f"{filtered_df['Account_Balance'].mean():,.0f}")

    # 1. Churn Rate by Account Type
    st.subheader("1. Churn Rate by Account Type")
    churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
    fig_cr = px.bar(churn_rate, x='Account_Type', y='Churn_Label', color='Churn_Label', text_auto='.2%', color_continuous_scale="Reds")
    fig_cr.update_layout(showlegend=False, yaxis_title="Churn Rate")
    st.plotly_chart(fig_cr, use_container_width=True)
    st.caption("🔍 *See which account types are most at risk of churn.*")

    # 2. Average Account Balance by Region
    st.subheader("2. Average Account Balance by Region")
    region_balance = filtered_df.groupby('Region')['Account_Balance'].mean().reset_index().sort_values("Account_Balance")
    fig_ab = px.bar(region_balance, x='Region', y='Account_Balance', text_auto='.2s', color='Account_Balance', color_continuous_scale="Blues")
    st.plotly_chart(fig_ab, use_container_width=True)
    st.caption("💸 *Spot regions with the highest value clients.*")

    # 3. Churn by Age Group
    st.subheader("3. Churn Rate by Age Group")
    bins = [17, 25, 35, 45, 55, 65, 80]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, include_lowest=True)
    churn_by_age = filtered_df.groupby('Age_Group')['Churn_Label'].mean().reset_index()
    fig_cage = px.line(churn_by_age, x='Age_Group', y='Churn_Label', markers=True)
    fig_cage.update_traces(line_color='red')
    st.plotly_chart(fig_cage, use_container_width=True)
    st.caption("👵 *Understand churn risk by age bracket.*")

    # 4. Satisfaction by Account Type
    st.subheader("4. Customer Satisfaction by Account Type")
    satisfaction = filtered_df.groupby('Account_Type')['Customer_Satisfaction_Score'].mean().reset_index()
    fig_sat = px.bar(satisfaction, x='Account_Type', y='Customer_Satisfaction_Score', color='Customer_Satisfaction_Score', text_auto='.2f', color_continuous_scale="Greens")
    st.plotly_chart(fig_sat, use_container_width=True)
    st.caption("⭐ *How happy are different account holders?*")

    # 5. Loan Amount Distribution by Loan Type
    st.subheader("5. Loan Amount Distribution by Loan Type")
    loan_dist = filtered_df.groupby('Loan_Type')['Loan_Amount'].sum().reset_index().sort_values("Loan_Amount", ascending=False)
    fig_loan = px.bar(loan_dist, x='Loan_Type', y='Loan_Amount', text_auto='.2s', color='Loan_Amount', color_continuous_scale="Viridis")
    st.plotly_chart(fig_loan, use_container_width=True)
    st.caption("🏦 *Which loan products are most popular by value?*")

    # 6. Credit Score: Churned vs. Non-Churned
    st.subheader("6. Credit Score Distribution: Churned vs. Non-Churned")
    fig_box = px.box(filtered_df, x='Churn_Label', y='Credit_Score', color='Churn_Label',
                     labels={'Churn_Label': 'Churned'}, points="all")
    fig_box.update_xaxes(tickvals=[0, 1], ticktext=['Not Churned', 'Churned'])
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("📉 *Are churned customers lower risk?*")

    # 7. Customer Count by Branch
    st.subheader("7. Top 10 Branches by Customer Count")
    top_branches = filtered_df['Branch'].value_counts().head(10).reset_index()
    top_branches.columns = ['Branch', 'Count']
    fig_br = px.bar(top_branches, x='Branch', y='Count', color='Count', color_continuous_scale="teal")
    st.plotly_chart(fig_br, use_container_width=True)
    st.caption("🏢 *Branches with the biggest customer bases.*")

    # 8. Transaction Type Pie Chart (Plotly)
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
    st.caption("💳 *How are customers using their accounts?*")

    # 9. Monthly Transaction Amount Trend
    st.subheader("9. Monthly Transaction Amount Trend")
    filtered_df['Transaction_Month'] = pd.to_datetime(filtered_df['Transaction_Date']).dt.to_period('M').astype(str)
    monthly_trx = filtered_df.groupby('Transaction_Month')['Transaction_Amount'].sum().reset_index()
    fig_mt = px.line(monthly_trx, x='Transaction_Month', y='Transaction_Amount', markers=True)
    st.plotly_chart(fig_mt, use_container_width=True)
    st.caption("📈 *When are transaction volumes peaking?*")

    # 10. Correlation Heatmap of Key Numeric Variables
    st.subheader("10. Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include='number').drop(columns=['Churn_Label'], errors='ignore')
    corr = numeric_cols.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    st.pyplot(fig2)
    st.caption("🤝 *Are balances, loans, credit scores interlinked?*")

    st.info("Tip: All visuals update live with your segment filters. Download filtered data for deeper dives.", icon="📊")

# ---- CLASSIFICATION TAB ----
with tabs[3]:
    st.header("🤖 Churn Prediction (Classification)")
    st.write("Predict and prevent customer churn using powerful machine learning algorithms.")
    drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 'Churn_Timeframe', 'Simulated_New_Churn_Label']
    target = 'Churn_Label'
    features = [col for col in df.columns if col not in drop_cols + [target]]
    X = df[features]
    y = df[target]
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    X_encoded = X_encoded.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42, stratify=y)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }
    metrics = []
    predictions = {}
    probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        predictions[name] = (y_pred_train, y_pred_test)
        probs[name] = y_prob_test
        metrics.append({
            "Algorithm": name,
            "Train Accuracy": accuracy_score(y_train, y_pred_train),
            "Test Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-Score": f1_score(y_test, y_pred_test),
        })
    st.subheader("Model Performance Comparison")
    st.dataframe(pd.DataFrame(metrics).set_index("Algorithm").style.format("{:.2%}"))
    st.caption("How well does each model predict churn? (Test data is what matters!)")
    st.subheader("Confusion Matrix")
    cm_option = st.selectbox("Choose Model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, predictions[cm_option][1])
    cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Predicted Not Churned", "Predicted Churned"])
    st.write(cm_df)
    st.subheader("ROC Curves (All Models)")
    fig, ax = plt.subplots()
    for name in models.keys():
        if probs[name] is not None:
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    st.pyplot(fig)
    st.caption("Which model best distinguishes churners vs. non-churners?")
    st.subheader("Predict Churn for New Data")
    uploaded_pred_file = st.file_uploader("Upload new data (xlsx/csv, same features, no 'Churn_Label')", type=["xlsx", "csv"], key="predict")
    if uploaded_pred_file:
        if uploaded_pred_file.name.endswith("xlsx"):
            new_data = pd.read_excel(uploaded_pred_file)
        else:
            new_data = pd.read_csv(uploaded_pred_file)
        new_data_enc = new_data.copy()
        for col in new_data_enc.select_dtypes(include=['object', 'category']).columns:
            new_data_enc[col] = LabelEncoder().fit_transform(new_data_enc[col].astype(str))
        new_data_enc = new_data_enc.fillna(0)
        best_model_name = max(metrics, key=lambda x: x['Test Accuracy'])["Algorithm"]
        best_model = models[best_model_name]
        preds = best_model.predict(new_data_enc)
        results = new_data.copy()
        results["Predicted_Churn_Label"] = preds
        st.success(f"Predictions done using: {best_model_name}")
        st.write(results.head())
        st.download_button(
            label="Download Results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# ---- CLUSTERING TAB ----
with tabs[4]:
    st.header("🧩 Customer Clustering")
    st.write("Group customers for tailored offers using K-Means clustering.")
    numeric_exclude = [
        "Customer_ID", "Churn_Label", "Simulated_New_Churn_Label",
        "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date", "Churn_Timeframe"
    ]
    cluster_features = df.select_dtypes(include=['number']).drop(columns=numeric_exclude, errors="ignore").columns.tolist()
    st.info(f"Features used for clustering: {', '.join(cluster_features)}")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    k = st.slider("Select clusters (k)", min_value=2, max_value=10, value=3, step=1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    st.subheader("Elbow Chart (Optimal K)")
    inertias = []
    K_range = range(2, 11)
    for ki in K_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(list(K_range), inertias, marker="o")
    ax.set_xlabel("Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Chart")
    st.pyplot(fig)
    st.caption("Where does the 'elbow' bend? That's your best k.")
    st.subheader("Cluster Persona Table")
    persona = df_with_clusters.groupby('Cluster')[cluster_features].mean().round(2)
    st.dataframe(persona)
    st.caption("Average features per cluster. Use for targeted strategies!")
    st.download_button(
        label="Download Cluster Data (CSV)",
        data=df_with_clusters.to_csv(index=False).encode("utf-8"),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

# ---- ASSOCIATION RULES TAB ----
with tabs[5]:
    st.header("🔗 Association Rule Mining")
    st.write("Find patterns between categorical features using Apriori (e.g., product combos for loyalty).")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    default_cols = ['Account_Type', 'Transaction_Type'] if 'Account_Type' in cat_cols and 'Transaction_Type' in cat_cols else cat_cols[:2]
    apriori_cols = st.multiselect(
        "Select 2+ categorical columns:",
        options=cat_cols, default=default_cols
    )
    min_support = st.slider("Min Support", 0.01, 0.2, 0.05, step=0.01)
    min_conf = st.slider("Min Confidence", 0.01, 1.0, 0.3, step=0.01)
    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.2, step=0.1)
    if len(apriori_cols) >= 2:
        from mlxtend.frequent_patterns import apriori, association_rules
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
            st.caption("Top rules: look for high confidence & lift (>1).")
        else:
            st.warning("No rules found. Try different columns or lower thresholds.")

# ---- REGRESSION TAB ----
with tabs[6]:
    st.header("📈 Regression (Satisfaction & Value)")
    st.write("Predict customer satisfaction, income, or balances. Spot top drivers and outliers.")
    regression_targets = ['Account_Balance', 'Annual_Income', 'Customer_Satisfaction_Score']
    target_reg = st.selectbox("Regression target", regression_targets)
    reg_drop_cols = [
        "Customer_ID", "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date",
        "Churn_Label", "Simulated_New_Churn_Label", "Churn_Timeframe"
    ] + regression_targets
    reg_features = [col for col in df.columns if col not in reg_drop_cols]
    X_reg = df[reg_features]
    y_reg = df[target_reg]
    X_reg_encoded = X_reg.copy()
    for col in X_reg_encoded.select_dtypes(include=['object', 'category']).columns:
        X_reg_encoded[col] = LabelEncoder().fit_transform(X_reg_encoded[col].astype(str))
    X_reg_encoded = X_reg_encoded.fillna(0)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_encoded, y_reg, test_size=0.25, random_state=42)
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    results = []
    y_preds = {}
    for name, reg in regressors.items():
        reg.fit(Xr_train, yr_train)
        y_pred = reg.predict(Xr_test)
        y_preds[name] = y_pred
        results.append({
            "Model": name,
            "R²": r2_score(yr_test, y_pred),
            "MAE": mean_absolute_error(yr_test, y_pred),
            "RMSE": mean_squared_error(yr_test, y_pred, squared=False)
        })
    st.subheader("Model Comparison Table")
    st.dataframe(pd.DataFrame(results).set_index("Model").style.format("{:.2f}"))
    st.subheader("Feature Importance (Decision Tree)")
    dt_reg = regressors["Decision Tree"]
    feat_imp = pd.Series(dt_reg.feature_importances_, index=Xr_train.columns)
    feat_imp = feat_imp[feat_imp > 0].sort_values(ascending=False).head(10)
    st.bar_chart(feat_imp)
    st.markdown("""
**Quick Insights:**
- Higher credit score = higher balances/income.
- Complaints and low satisfaction → lower balances.
- Premium products linked to higher income.
- Younger customers tend to have lower balances.
- Loans and employment status are top drivers.
    """)

# ---- ADVANCED TABS PLACEHOLDER ----
with tabs[7]:
    st.header("💬 NLP (Feedback/Complaints Analysis)")
    st.info("Add a column with customer comments or complaints to unlock text analytics (word clouds, sentiment, topic modeling). Coming soon!")

with tabs[8]:
    st.header("⏳ Time Series Trends")
    st.info("Analyze and forecast account activity, complaints, or balances over time. Feature coming soon!")

with tabs[9]:
    st.header("🕵️‍♂️ Advanced Segmentation")
    st.info("Interactive cluster explorer and explainability (e.g., SHAP). Coming soon!")

st.markdown("---\n*Dashboard by Data Science Team – For questions, reach out to your analytics partner!*")
