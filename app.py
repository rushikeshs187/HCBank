import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config("🏦 Bank Analytics", layout="wide")

TABS = [
    "🎯 Objectives & Guide", "📊 Visual Explorer", "🤖 Churn Prediction", "🧩 Customer Segmentation",
    "📈 Value Regression", "🔗 Association Rules", "⏳ Monthly Trends"
]
tabs = st.tabs(TABS)

# ============ SIDEBAR: UPLOAD ===========
with st.sidebar:
    st.title("📥 Data Upload")
    up = st.file_uploader("Upload Excel (sheet 'Cleaned data')", type="xlsx")
    st.markdown("---")
    st.caption("Built with Plotly + Scikit-learn")
if up is None:
    st.warning("Upload your Excel file to start!")
    st.stop()

try:
    df = pd.read_excel(up, sheet_name="Cleaned data")
except Exception as e:
    st.error(f"❌ Problem reading 'Cleaned data' sheet: {e}")
    st.stop()

cols = df.columns

# ============ OBJECTIVES & GUIDE ==========
with tabs[0]:
    st.markdown("## 🎯 Dashboard Objectives")
    st.write(
        "- Predict and reduce customer churn\n"
        "- Forecast satisfaction & account value\n"
        "- Segment and profile up to 200 customer clusters\n"
        "- Mine cross-sell and loyalty rules (apriori)\n"
        "- Visualize monthly financial/sentiment flows\n"
        "- All charts/tabs adapt to available columns"
    )
    st.markdown("## 💡 Quick Guide")
    st.info(
        "Use filters at the top of each tab.\n"
        "Hover/zoom for interactive insight. "
        "Download results for campaign targeting or reporting."
    )

# ============ VISUAL EXPLORER =============
with tabs[1]:
    import plotly.express as px, matplotlib.pyplot as plt, seaborn as sns
    st.header("📊 Data Visualisation & Insights")

    # Filter bar
    c1,c2,c3 = st.columns(3)
    f_gender = c1.multiselect("Gender", df['Gender'].unique()) if 'Gender' in cols else []
    f_type   = c2.multiselect("Account Type", df['Account_Type'].unique()) if 'Account_Type' in cols else []
    f_region = c3.multiselect("Region", df['Region'].unique()) if 'Region' in cols else []
    v = df.copy()
    if f_gender: v = v[v.Gender.isin(f_gender)]
    if f_type:   v = v[v.Account_Type.isin(f_type)]
    if f_region: v = v[v.Region.isin(f_region)]
    st.success(f"Filtered rows: {len(v)}")

    # KPIs
    k1,k2,k3 = st.columns(3)
    k1.metric("Churn %", f"{v['Churn_Label'].mean()*100:.1f}%" if 'Churn_Label' in cols else "N/A")
    k2.metric("Avg. Satisfaction", f"{v['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in cols else "N/A")
    k3.metric("Avg. Balance", f"{v['Account_Balance'].mean():,.0f}" if 'Account_Balance' in cols else "N/A")

    # Churn by account type
    if {'Account_Type','Churn_Label'}<=set(cols):
        st.subheader("Churn Rate by Account Type")
        dt = v.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        st.plotly_chart(px.bar(dt, x='Account_Type', y='Churn_Label', color='Churn_Label', text_auto='.1%', color_continuous_scale='Reds'), use_container_width=True)
        st.caption("Insight: Some account types have double the churn risk. Target upgrades and retention offers accordingly.")

    # Sunburst: Region → Account Type → Churn
    if {'Region','Account_Type','Churn_Label'} <= set(cols):
        st.subheader("Customer Distribution: Region → Account → Churn")
        sb_data = v.dropna(subset=['Region', 'Account_Type', 'Churn_Label'])
        sb_args = dict(
            data_frame=sb_data,
            path=['Region','Account_Type','Churn_Label'],
            color='Churn_Label',
            color_continuous_scale='RdBu'
        )
        if 'Customer_ID' in cols:
            sb_args['values'] = 'Customer_ID'
        fig = px.sunburst(**sb_args)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    nums = v.select_dtypes('number')
    if len(nums.columns) >= 3:
        st.subheader("Correlation Heatmap (Key Numeric Columns)")
        fig,ax = plt.subplots(figsize=(7,4))
        sns.heatmap(nums.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Account balance vs age, colored by churn
    if {'Account_Balance','Age','Churn_Label'}<=set(cols):
        st.subheader("Balance vs Age (Churn Color)")
        fig = px.scatter(v, x='Age', y='Account_Balance', color='Churn_Label', size_max=10, opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Insight: Younger and older customers show lower balances—and higher churn.")

    # Loan amount by branch (treemap)
    if {'Loan_Type','Branch','Loan_Amount'}<=set(cols):
        st.subheader("Loan Amount by Type and Branch")
        fig = px.treemap(v, path=['Loan_Type','Branch'], values='Loan_Amount', color='Loan_Amount', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

    # Monthly trend lines
    if 'Transaction_Date' in cols and any(c in cols for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score']):
        st.subheader("Monthly Financial/Sentiment Trends")
        v['Month'] = pd.to_datetime(v['Transaction_Date']).dt.to_period('M').astype(str)
        metrics = [c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in v]
        gm = v.groupby('Month')[metrics].mean().reset_index()
        st.plotly_chart(px.line(gm, x='Month', y=metrics, markers=True), use_container_width=True)

# ============ CHURN CLASSIFICATION ==========
with tabs[2]:
    st.header("🤖 Churn Prediction (ML Classifier)")
    if 'Churn_Label' not in cols:
        st.warning("Column **Churn_Label** missing.")
    else:
        drop = ['Customer_ID','Transaction_Date','Account_Open_Date','Last_Transaction_Date','Churn_Timeframe']
        X = df.drop(columns=[c for c in drop if c in cols]+['Churn_Label'])
        y = df['Churn_Label']
        for c in X.select_dtypes('object'): X[c]=LabelEncoder().fit_transform(X[c].astype(str))
        X.fillna(0, inplace=True)
        Xt,Xs,yt,ys = train_test_split(X,y,stratify=y,test_size=.25,random_state=42)
        from sklearn.ensemble import GradientBoostingClassifier
        mdl = GradientBoostingClassifier(random_state=42).fit(Xt,yt)
        preds = mdl.predict(Xs)
        from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
        st.metric("Accuracy", f"{accuracy_score(ys,preds):.2%}")
        st.text(classification_report(ys,preds))
        # ROC
        prob = mdl.predict_proba(Xs)[:,1]
        fpr,tpr,_ = roc_curve(ys,prob); aucv=auc(fpr,tpr)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'k--')
        ax.set_title(f"ROC  AUC={aucv:.2f}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig)
        # Feature importances
        st.subheader("Top Feature Importances")
        fi = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        st.bar_chart(fi)
        st.caption("Insight: Credit score, income, satisfaction, and product mix are often the main churn predictors.")

# ============ SEGMENTATION (CLUSTERING) ==========
with tabs[3]:
    st.header("🧩 K-Means Segmentation")
    nums = df.select_dtypes('number').drop(columns=['Churn_Label'] if 'Churn_Label' in cols else [])
    if nums.shape[1] < 2:
        st.warning("Need at least 2 numeric features.")
    else:
        k = st.slider("Choose clusters (k)", 2, 200, 8)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42).fit(StandardScaler().fit_transform(nums))
        df['Cluster'] = kmeans.labels_
        st.write("Cluster counts", df['Cluster'].value_counts().sort_index())
        st.dataframe(df.groupby('Cluster')[nums.columns].mean().round(1))
        # Download
        st.download_button("Download data with Clusters", df.to_csv(index=False).encode(), "clusters.csv", "text/csv")
        # Visual
        if {'Account_Balance','Annual_Income'}<=set(nums.columns):
            st.subheader("Scatter: Clusters by Balance & Income")
            fig = px.scatter(df, x='Annual_Income', y='Account_Balance', color='Cluster', opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Insight: Outlier clusters often point to segments with unique value/risk profiles.")

# ============ REGRESSION ==========
with tabs[4]:
    st.header("📈 Ridge Regression (Value, Income, Satisfaction)")
    targets = [c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in cols]
    if not targets:
        st.warning("No numeric target columns.")
    else:
        tgt = st.selectbox("Target variable", targets)
        X = df.drop(columns=[tgt]).select_dtypes(exclude='datetime')
        for c in X.select_dtypes('object'): X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        y = df[tgt]
        X.fillna(0,inplace=True)
        Xt,Xs,yt,ys = train_test_split(X,y,test_size=.25,random_state=42)
        from sklearn.linear_model import Ridge
        reg = Ridge().fit(Xt,yt)
        yp = reg.predict(Xs)
        from sklearn.metrics import r2_score, mean_squared_error
        rmse = np.sqrt(mean_squared_error(ys, yp))
        st.metric("R²",f"{r2_score(ys, yp):.2f}")
        st.metric("RMSE",f"{rmse:,.0f}")
        st.subheader("Top Feature Effects (Ridge coef.)")
        imp = pd.Series(reg.coef_, index=X.columns).sort_values(ascending=False).head(10)
        st.bar_chart(imp)
        st.caption("Insight: Demographics and product usage are usually the strongest drivers of customer value.")

# ============ ASSOCIATION RULES ==========
with tabs[5]:
    st.header("🔗 Association Rule Mining (Apriori)")
    catcols = df.select_dtypes('object').columns
    if len(catcols) < 2:
        st.warning("Need at least 2 categorical columns.")
    else:
        from mlxtend.frequent_patterns import apriori, association_rules
        sel = st.multiselect("Columns to mine", catcols, default=list(catcols)[:3])
        min_sup = st.slider("Min Support", .01,.2,.05,.01)
        min_conf = st.slider("Min Confidence", .1,1.,.3,.05)
        min_lift = st.slider("Min Lift", 1.,5.,1.2,.1)
        enc = pd.get_dummies(df[sel].astype(str))
        freq = apriori(enc, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules.lift>=min_lift].sort_values('confidence',ascending=False).head(20)
        if rules.empty:
            st.info("No rules found at these thresholds.")
        else:
            rules['antecedents'] = rules['antecedents'].apply(lambda s:', '.join(list(s)))
            rules['consequents'] = rules['consequents'].apply(lambda s:', '.join(list(s)))
            st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
            # Visualize: Support vs Confidence
            st.plotly_chart(
                px.scatter(rules,x='support',y='confidence',size='lift',hover_data=['antecedents','consequents']),
                use_container_width=True
            )
            st.caption("Insight: High-confidence rules reveal natural cross-sell or risk clusters.")

# ============ MONTHLY TRENDS ==========
with tabs[6]:
    st.header("⏳ Monthly Trends")
    if 'Transaction_Date' not in cols:
        st.warning("Transaction_Date missing.")
    else:
        df['Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M').astype(str)
        mt_cols = [c for c in ['Transaction_Amount','Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in cols]
        if mt_cols:
            gm = df.groupby('Month')[mt_cols].mean().reset_index()
            st.plotly_chart(px.line(gm, x='Month', y=mt_cols, markers=True), use_container_width=True)
            st.caption("Insight: Monitor sudden changes for risk/opportunity signals.")

st.markdown("---\n_Built for insight. Missing a chart? Check your columns or data sheet._")
