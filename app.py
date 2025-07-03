import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Bank Analytics Dashboard", layout="wide")

st.sidebar.title("Bank Analytics Dashboard")
st.sidebar.write("""
Built with ❤️ by [Your Name]

[GitHub Repo](your-repo-link)
""")

st.title("Bank Customer Data Analytics Dashboard")

# --- File uploader for Excel ---
uploaded_file = st.file_uploader("Upload your Excel dataset (must contain a 'Cleaned data' sheet)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
else:
    st.warning("Please upload your data file to continue.")
    st.stop()

tabs = st.tabs([
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
])

# ---- Data Visualisation Tab ----
with tabs[0]:
    st.header("Data Visualisation")
    st.write("Explore your bank customer dataset with advanced visual insights. Use the filters below to drill down on segments of interest.")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.multiselect("Select Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
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
    st.success(f"{len(filtered_df)} records after filtering.")

    # 1. Churn Rate by Account Type
    st.subheader("1. Churn Rate by Account Type")
    churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean()
    st.bar_chart(churn_rate)
    st.caption("This chart shows the proportion of customers who have churned, segmented by account type.")

    # 2. Average Account Balance by Region
    st.subheader("2. Average Account Balance by Region")
    region_balance = filtered_df.groupby('Region')['Account_Balance'].mean().sort_values()
    st.bar_chart(region_balance)
    st.caption("Reveals which regions hold the highest average account balances.")

    # 3. Churn by Age Group
    st.subheader("3. Churn Rate by Age Group")
    bins = [17, 25, 35, 45, 55, 65, 80]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, include_lowest=True)
    churn_by_age = filtered_df.groupby('Age_Group')['Churn_Label'].mean()
    st.line_chart(churn_by_age)
    st.caption("Shows which age groups are most likely to churn.")

    # 4. Customer Satisfaction by Account Type
    st.subheader("4. Customer Satisfaction by Account Type")
    satisfaction = filtered_df.groupby('Account_Type')['Customer_Satisfaction_Score'].mean()
    st.bar_chart(satisfaction)
    st.caption("Visualizes average satisfaction scores by account type.")

    # 5. Loan Distribution by Type
    st.subheader("5. Loan Amount Distribution by Loan Type")
    loan_dist = filtered_df.groupby('Loan_Type')['Loan_Amount'].sum().sort_values(ascending=False)
    st.bar_chart(loan_dist)
    st.caption("Total loan amounts distributed across various loan types.")

    # 6. Churn vs. Credit Score (Boxplot)
    st.subheader("6. Credit Score Distribution: Churned vs. Non-Churned")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='Churn_Label', y='Credit_Score', ax=ax)
    ax.set_xticklabels(['Not Churned', 'Churned'])
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Credit Score")
    st.pyplot(fig)
    st.caption("Compares the credit score distributions for churned vs. non-churned customers.")

    # 7. Customer Count by Branch (Top 10)
    st.subheader("7. Top 10 Branches by Customer Count")
    top_branches = filtered_df['Branch'].value_counts().head(10)
    st.bar_chart(top_branches)
    st.caption("Displays the branches with the largest number of customers.")

    # 8. Transaction Type Distribution (Plotly Pie Chart)
    st.subheader("8. Transaction Type Distribution")
    trx_dist = filtered_df['Transaction_Type'].value_counts()
    trx_dist_df = trx_dist.reset_index()
    trx_dist_df.columns = ['Transaction Type', 'Count']
    import plotly.express as px
    fig3 = px.pie(
        trx_dist_df,
        names='Transaction Type',
        values='Count',
        title='Transaction Type Distribution',
        hole=0.3
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Shows how different types of transactions are distributed in the filtered dataset.")

    # 9. Monthly Transaction Amount Trend
    st.subheader("9. Monthly Transaction Amount Trend")
    filtered_df['Transaction_Month'] = filtered_df['Transaction_Date'].dt.to_period('M').astype(str)
    monthly_trx = filtered_df.groupby('Transaction_Month')['Transaction_Amount'].sum()
    st.line_chart(monthly_trx)
    st.caption("Total transaction amount per month, showing seasonality or trends.")

    # 10. Correlation Heatmap of Key Numeric Variables
    st.subheader("10. Correlation Heatmap")
    import numpy as np
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).drop(columns=['Churn_Label'], errors='ignore')
    corr = numeric_cols.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    st.pyplot(fig2)
    st.caption("Shows relationships between key numeric variables.")

    st.info("All charts update automatically based on the filters above. Use these insights to identify trends, risks, and opportunities in your customer base.")

# ---- Classification Tab ----
with tabs[1]:
    st.header("Classification Models: Churn Prediction")
    st.write(
        """
        Apply and compare classification algorithms to predict customer churn.
        Upload new data to generate predictions using the best-performing model.
        """
    )
    drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 
                 'Churn_Timeframe', 'Simulated_New_Churn_Label']
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
    st.caption("Accuracy, precision, recall, and F1-score are shown for each model.")
    st.subheader("Confusion Matrix")
    cm_option = st.selectbox("Select Model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, predictions[cm_option][1])
    cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Predicted Not Churned", "Predicted Churned"])
    st.write(cm_df)
    st.caption(f"Confusion matrix for {cm_option} on test data.")
    st.subheader("ROC Curves: All Models")
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
    st.caption("ROC curve compares classifier ability to distinguish churners/non-churners.")
    st.subheader("Upload New Data for Prediction")
    uploaded_pred_file = st.file_uploader("Upload new data (xlsx/csv, same columns as original features, *without* Churn_Label)", type=["xlsx", "csv"], key="predict")
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
        st.success(f"Prediction done using best model: {best_model_name}")
        st.write(results.head())
        st.download_button(
            label="Download Results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
        st.caption("Download predicted churn results for new uploaded data.")
    st.info("Select a model above to view its confusion matrix. ROC curves compare model performance. Upload new data to generate churn predictions.")

# ---- Clustering Tab ----
with tabs[2]:
    st.header("Customer Clustering")
    st.write("""
        Segment your customers using KMeans clustering.
        Use the slider to set number of clusters, view the elbow chart, and download persona-labeled data.
    """)
    numeric_exclude = [
        "Customer_ID", "Churn_Label", "Simulated_New_Churn_Label",
        "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date", "Churn_Timeframe"
    ]
    cluster_features = df.select_dtypes(include=['number']).drop(columns=numeric_exclude, errors="ignore").columns.tolist()
    st.write("Features used for clustering:", ", ".join(cluster_features))
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    k = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=3, step=1)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    st.subheader("Elbow Chart (Optimal K Detection)")
    inertias = []
    K_range = range(2, 11)
    for ki in K_range:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(list(K_range), inertias, marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Chart for KMeans")
    st.pyplot(fig)
    st.caption("The 'elbow' point helps choose the optimal number of clusters for your data.")
    st.subheader("Cluster Persona Table")
    persona = df_with_clusters.groupby('Cluster')[cluster_features].mean().round(2)
    st.dataframe(persona)
    st.caption("Average value of each feature per cluster. Use this to interpret customer personas.")
    st.subheader("Download Cluster-Labeled Data")
    st.download_button(
        label="Download Data with Cluster Labels (CSV)",
        data=df_with_clusters.to_csv(index=False).encode("utf-8"),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )
    st.caption("Download full dataset including assigned cluster labels for further analysis.")
    st.info("Cluster assignments update automatically with k. Use the persona table to guide marketing and segmentation strategies.")

# ---- Association Rule Mining Tab ----
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")
    st.write("""
        Discover hidden patterns between categorical variables using the Apriori algorithm.
        Select columns and parameters to find meaningful associations in your data.
    """)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    default_cols = ['Account_Type', 'Transaction_Type'] if 'Account_Type' in cat_cols and 'Transaction_Type' in cat_cols else cat_cols[:2]
    apriori_cols = st.multiselect(
        "Select 2 or more categorical columns for association rule mining:",
        options=cat_cols, default=default_cols
    )
    min_support = st.slider("Minimum Support:", 0.01, 0.2, 0.05, step=0.01)
    min_conf = st.slider("Minimum Confidence:", 0.01, 1.0, 0.3, step=0.01)
    min_lift = st.slider("Minimum Lift:", 1.0, 5.0, 1.2, step=0.1)
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
            st.caption("Top-10 association rules based on confidence. 'Support' is the proportion of records containing both antecedents and consequents. 'Lift' > 1 indicates positive association.")
        else:
            st.warning("No rules found with the selected columns and thresholds. Try lowering thresholds or choosing different columns.")
    else:
        st.info("Please select at least two categorical columns to run association rule mining.")
    st.info("Adjust parameters to find more or fewer associations. Use these rules to discover strong links between customer behaviors or attributes.")

# ---- Regression Tab ----
with tabs[4]:
    st.header("Regression Models & Insights")
    st.write("""
        Predict key financial outcomes (e.g., account balance, income, satisfaction) using regression.
        Review model results and uncover actionable relationships.
    """)
    regression_targets = ['Account_Balance', 'Annual_Income', 'Customer_Satisfaction_Score']
    target_reg = st.selectbox("Select regression target:", regression_targets)
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
    import numpy as np
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
            "RMSE": np.sqrt(mean_squared_error(yr_test, y_pred))
        })
    st.subheader("Model Comparison Table")
    st.dataframe(pd.DataFrame(results).set_index("Model").style.format("{:.2f}"))
    st.caption("Higher R² and lower errors indicate better predictive performance.")
    st.subheader("Feature Importance (Decision Tree)")
    dt_reg = regressors["Decision Tree"]
    feat_imp = pd.Series(dt_reg.feature_importances_, index=Xr_train.columns)
    feat_imp = feat_imp[feat_imp > 0].sort_values(ascending=False).head(10)
    st.bar_chart(feat_imp)
    st.caption("Top features driving predictions for the tree model.")
    st.subheader("Quick Regression Insights")
    st.markdown("""
    - Customers with higher credit scores tend to have higher account balances and incomes.
    - Recent complaints and low satisfaction scores are negatively associated with account balances.
    - Subscription to premium services is linked to higher annual income.
    - Younger customers (under 25) have significantly lower account balances.
    - Employment status and loan amounts are among the strongest predictors for both account balance and satisfaction.
    - Number of transactions positively impacts predicted account balance.
    - Churn risk (from previous tab) is inversely associated with both account balance and satisfaction.
    """)
    st.caption("Insights derived from regression coefficients and feature importances.")
    st.info("Use these models to forecast key business outcomes and identify actionable drivers of customer value.")

st.markdown("""
---
*All charts and tables include explanations below for your insights.*
""")
