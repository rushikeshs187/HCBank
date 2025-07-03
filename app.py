# ──────────────────────────────────────────────────────────
#  BANK CUSTOMER ANALYTICS – INSIGHT-RICH STREAMLIT APP
# ──────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config("🏦 Bank Customer Analytics", layout="wide")

# ──────────────────────────── TABS ─────────────────────────────
tabs = st.tabs([
    "🎯 Objectives",  "💡 Help", "📊 Visual Insights",
    "🤖 Classification", "🧩 Clustering",
    "📈 Regression",  "🔗 Association Rules",
    "⏳ Trends"
])

# ───────────────────────── SIDEBAR ─────────────────────────────
with st.sidebar:
    st.title("📥 Data Upload")
    f = st.file_uploader("Excel with sheet 'Cleaned data'", type="xlsx")
    st.markdown("---")
    st.caption("Built with Plotly + Scikit-learn")
if f is None:
    st.warning("Upload your Excel file to start.")
    st.stop()

try:
    df = pd.read_excel(f, sheet_name="Cleaned data")
except Exception as e:
    st.error(f"💥 Sheet 'Cleaned data' not found – {e}")
    st.stop()

cols = df.columns

# ─────────────────── OBJECTIVES ────────────────────────────────
with tabs[0]:
    st.markdown("## 🎯 Objectives")
    st.markdown("""
* Predict & reduce churn  
* Forecast satisfaction / value  
* Segment customers (up to 200 K-means clusters)  
* Mine cross-sell & loyalty patterns (Apriori rules)  
* Visualise monthly financial trends
    """)

# ────────────────────── HELP ───────────────────────────────────
with tabs[1]:
    st.markdown("## 💡 How to use")
    st.markdown("""
1. Apply filters at top of each tab  
2. Hover / click legends for detail  
3. Download CSV outputs for campaigns
    """)

# ────────────────── VISUAL INSIGHTS ────────────────────────────
with tabs[2]:
    import plotly.express as px, plotly.figure_factory as ff, matplotlib.pyplot as plt, seaborn as sns
    st.header("📊 Exploratory Visuals")

    # Basic filters
    c1,c2,c3 = st.columns(3)
    f_gender  = c1.multiselect("Gender",       df['Gender'].unique())        if 'Gender'       in cols else []
    f_type    = c1.multiselect("Account Type", df['Account_Type'].unique())  if 'Account_Type' in cols else []
    f_region  = c2.multiselect("Region",       df['Region'].unique())        if 'Region'       in cols else []
    view = df.copy()
    if f_gender: view  = view[view.Gender.isin(f_gender)]
    if f_type:   view  = view[view.Account_Type.isin(f_type)]
    if f_region: view  = view[view.Region.isin(f_region)]
    st.success(f"Filtered rows: {len(view)}")

    # KPI cards
    k1,k2,k3 = st.columns(3)
    if 'Churn_Label' in cols:
        k1.metric("Churn %", f"{view['Churn_Label'].mean()*100:.1f}%")
    if 'Customer_Satisfaction_Score' in cols:
        k2.metric("Avg Satisfaction", f"{view['Customer_Satisfaction_Score'].mean():.2f}")
    if 'Account_Balance' in cols:
        k3.metric("Avg Balance", f"{view['Account_Balance'].mean():,.0f}")

    # Sunburst (Region → Account Type → Churn)
    if {'Region','Account_Type','Churn_Label'}<=set(cols):
        st.subheader("Region • Account • Churn sunburst")
        sb = px.sunburst(view, path=['Region','Account_Type','Churn_Label'],
                         values='Customer_ID' if 'Customer_ID' in cols else None,
                         color='Churn_Label', color_continuous_scale='RdBu_r')
        st.plotly_chart(sb, use_container_width=True)

    # Treemap of loan amount by loan type & branch
    if {'Loan_Type','Branch','Loan_Amount'}<=set(cols):
        st.subheader("Treemap – Loan amount by Type & Branch")
        tm = px.treemap(view, path=['Loan_Type','Branch'], values='Loan_Amount',
                        color='Loan_Amount', color_continuous_scale='Viridis')
        st.plotly_chart(tm, use_container_width=True)

    # Scatter-matrix of key numeric columns
    nums = view.select_dtypes('number').drop(columns=['Churn_Label'] if 'Churn_Label'in view else [])
    if nums.shape[1]>=3:
        st.subheader("Scatter-matrix (sample 800)")
        smp = nums.sample(min(800,len(nums)), random_state=1)
        st.plotly_chart(px.scatter_matrix(smp, dimensions=smp.columns[:6]), use_container_width=True)

    # KDE density of credit score by churn
    if {'Credit_Score','Churn_Label'}<=set(cols):
        st.subheader("Credit-score density • churn vs stay")
        kde = ff.create_distplot(
            [view[view.Churn_Label==0]['Credit_Score'],
             view[view.Churn_Label==1]['Credit_Score']],
            group_labels=['Stay','Churn'], show_hist=False, show_rug=False)
        st.plotly_chart(kde, use_container_width=True)

    # Correlation heat-map
    numcorr = nums.corr()
    if numcorr.size>1:
        st.subheader("Correlation heat-map")
        fig,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numcorr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ─────────────── CLASSIFICATION ────────────────────────────────
with tabs[3]:
    st.header("🤖 Churn Prediction")
    if 'Churn_Label' not in cols:
        st.warning("Column **Churn_Label** missing.")
    else:
        drop = ['Customer_ID','Transaction_Date','Account_Open_Date','Last_Transaction_Date','Churn_Timeframe']
        X = df.drop(columns=[c for c in drop if c in cols]+['Churn_Label'])
        y = df['Churn_Label']
        for c in X.select_dtypes('object'): X[c]=LabelEncoder().fit_transform(X[c].astype(str))
        X.fillna(0, inplace=True)
        Xt,Xs,yt,ys=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)

        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42).fit(Xt,yt)
        preds = model.predict(Xs)
        from sklearn.metrics import accuracy_score,classification_report,roc_curve,auc
        st.metric("Accuracy", f"{accuracy_score(ys,preds):.2%}")
        st.text(classification_report(ys,preds))

        # ROC
        prob = model.predict_proba(Xs)[:,1]
        fpr,tpr,_=roc_curve(ys,prob); aucv=auc(fpr,tpr)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'k--')
        ax.set_title(f"ROC  AUC={aucv:.2f}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig)

# ───────────────── CLUSTERING ─────────────────────────────────
with tabs[4]:
    st.header("🧩 K-Means Clustering")
    num = df.select_dtypes('number').drop(columns=['Churn_Label'] if 'Churn_Label' in cols else [])
    if num.shape[1] < 2:
        st.warning("Need ≥2 numeric columns.")
    else:
        k = st.slider("Choose k", 2, 200, 8)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42).fit(StandardScaler().fit_transform(num))
        df['Cluster'] = kmeans.labels_
        st.write("Cluster size distribution", df['Cluster'].value_counts().sort_index())
        st.dataframe(df.groupby('Cluster')[num.columns].mean().round(1))
        st.download_button("Download cluster CSV", df.to_csv(index=False).encode(),
                           "clustered_data.csv","text/csv")

# ───────────────── REGRESSION ─────────────────────────────────
with tabs[5]:
    st.header("📈 Ridge Regression")
    targets = [c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in cols]
    if not targets:
        st.warning("Add numeric targets (balance/income/satisfaction).")
    else:
        tgt = st.selectbox("Target", targets)
        X = df.drop(columns=[tgt]).select_dtypes(exclude='datetime')
        for c in X.select_dtypes('object'): X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        X.fillna(0, inplace=True)
        y = df[tgt]
        Xt,Xs,yt,ys=train_test_split(X,y,test_size=.25,random_state=42)
        from sklearn.linear_model import Ridge
        reg=Ridge().fit(Xt,yt); yp=reg.predict(Xs)
        from sklearn.metrics import r2_score, mean_squared_error
        st.metric("R²",f"{r2_score(ys,yp):.2f}")
        rmse = np.sqrt(mean_squared_error(ys,yp))
        st.metric("RMSE",f"{rmse:,.0f}")

        st.subheader("Top positive drivers")
        imp = pd.Series(reg.coef_,index=X.columns).sort_values(ascending=False).head(10)
        st.bar_chart(imp)

# ───────────── ASSOCIATION RULES ──────────────────────────────
with tabs[6]:
    st.header("🔗 Association-Rule Mining")
    cat_cols = df.select_dtypes('object').columns
    if len(cat_cols) < 2:
        st.warning("Need ≥2 categorical columns.")
    else:
        from mlxtend.frequent_patterns import apriori, association_rules
        sel = st.multiselect("Columns to analyse", cat_cols, default=list(cat_cols)[:3])
        min_sup = st.slider("Min Support", .01,.2,.05,.01)
        min_conf = st.slider("Min Confidence", .1,1.,.3,.05)
        min_lift = st.slider("Min Lift", 1.,5.,1.2,.1)
        enc = pd.get_dummies(df[sel].astype(str))
        freq = apriori(enc, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules.lift>=min_lift].sort_values('confidence',ascending=False).head(20)
        if rules.empty:
            st.info("No rules – lower thresholds.")
        else:
            rules['antecedents'] = rules['antecedents'].apply(lambda s:', '.join(list(s)))
            rules['consequents'] = rules['consequents'].apply(lambda s:', '.join(list(s)))
            st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

            # scatter insight
            st.subheader("Support vs Confidence")
            st.plotly_chart(
                px.scatter(rules,x='support',y='confidence',size='lift',hover_data=['antecedents','consequents']),
                use_container_width=True
            )

# ────────────────── TRENDS ────────────────────────────────────
with tabs[7]:
    st.header("⏳ Monthly Trends")
    if 'Transaction_Date' not in cols:
        st.warning("Transaction_Date column missing.")
    else:
        df['Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M').astype(str)
        mt_cols = [c for c in ['Transaction_Amount','Account_Balance','Annual_Income'] if c in cols]
        if mt_cols:
            gm = df.groupby('Month')[mt_cols].mean().reset_index()
            st.plotly_chart(px.line(gm,x='Month',y=mt_cols,markers=True), use_container_width=True)
        else:
            st.info("No numeric monthly metrics found.")

st.markdown("---\n*All charts adapt to available columns. For deeper insights, ensure key fields exist.*")
