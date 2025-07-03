# ────────────────────────────────────────────────────────────────
#  Bank Customer Analytics  •  Streamlit Dashboard
# ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

# ───────────────── TABS ─────────────────
tab_names = [
    "🎯 Objectives", "💡 How to Use", "📊 Visual Insights",
    "🤖 Churn Classification", "🧩 Customer Clustering",
    "📈 Value Regression", "⏳ Time-Series"
]
tabs = st.tabs(tab_names)

# ─────────────── SIDEBAR ────────────────
with st.sidebar:
    st.title("🏦  Bank Analytics")
    up_file = st.file_uploader("Upload Excel (sheet **Cleaned data**)", type=["xlsx"])
    st.markdown("---")
    st.info("1) Upload data  2) Explore tabs  3) Download insights", icon="ℹ️")

# ───────────── LOAD DATA / STOP IF NONE ─────────────
if up_file is None:
    st.warning("Upload data to unlock the dashboard.", icon="⚠️")
    st.stop()

try:
    df = pd.read_excel(up_file, sheet_name="Cleaned data")
except Exception as e:
    st.error(f"❌ Could not read **Cleaned data** sheet:\n{e}")
    st.stop()

# ───────────── OBJECTIVES TAB ─────────────
with tabs[0]:
    st.markdown("## 🎯 Dashboard Objectives")
    st.markdown("""
- **Predict churn** and take proactive retention actions  
- **Estimate satisfaction / value** for smarter prioritisation  
- **Cluster customers** (up to 200 groups) for tailored offers  
- **Track monthly trends** in revenue & engagement  
    """)

# ───────────── HOW-TO TAB ────────────────
with tabs[1]:
    st.markdown("## 💡 How to Use")
    st.markdown("""
1. **Filter** cohorts in each tab  
2. **Hover** charts for details; click legend to isolate series  
3. **Download** CSVs for campaigns & reports  
    """)

# ───────────── VISUAL INSIGHTS TAB ───────
with tabs[2]:
    st.header("📊 Visual Insights")
    import plotly.express as px, matplotlib.pyplot as plt, seaborn as sns

    # Filters (only show if column exists)
    cols = df.columns
    c1,c2,c3 = st.columns(3)
    gender       = c1.multiselect("Gender", df['Gender'].unique())           if 'Gender'in cols else []
    acc_type     = c1.multiselect("Account Type", df['Account_Type'].unique())if 'Account_Type'in cols else []
    region       = c2.multiselect("Region", df['Region'].unique())           if 'Region'in cols else []

    view = df.copy()
    if gender   : view = view[view['Gender'].isin(gender)]
    if acc_type : view = view[view['Account_Type'].isin(acc_type)]
    if region   : view = view[view['Region'].isin(region)]

    st.success(f"Records after filter: {len(view)}")

    k1,k2,k3 = st.columns(3)
    k1.metric("Churn %", f"{view['Churn_Label'].mean()*100: .1f}%" if 'Churn_Label' in cols else "N/A")
    k2.metric("Avg Satisfaction", f"{view['Customer_Satisfaction_Score'].mean(): .2f}" if 'Customer_Satisfaction_Score'in cols else "N/A")
    k3.metric("Avg Balance", f"{view['Account_Balance'].mean():,.0f}" if 'Account_Balance'in cols else "N/A")

    # 1 Churn by account
    if {'Account_Type','Churn_Label'}<=set(cols):
        st.subheader("Churn rate by account type")
        ch = df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        st.plotly_chart(px.bar(ch,x='Account_Type',y='Churn_Label',color='Churn_Label',color_continuous_scale='Reds',
                               text_auto='.1%'), use_container_width=True)

    # 2 Monthly income / balance trend
    if {'Transaction_Date','Annual_Income','Account_Balance'}<=set(cols):
        st.subheader("Monthly income & balance trend")
        tmp = df.copy()
        tmp['Month'] = pd.to_datetime(tmp['Transaction_Date']).dt.to_period('M').astype(str)
        grp = tmp.groupby('Month')[['Annual_Income','Account_Balance']].mean().reset_index()
        st.plotly_chart(px.line(grp,x='Month',y=['Annual_Income','Account_Balance'],markers=True),
                        use_container_width=True)

# ───────────── CLASSIFICATION TAB ────────
with tabs[3]:
    st.header("🤖 Churn Classification")
    if 'Churn_Label' not in df:
        st.warning("Column **Churn_Label** missing.")
    else:
        drop_cols = ['Customer_ID','Transaction_Date','Account_Open_Date','Last_Transaction_Date','Churn_Timeframe']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]+['Churn_Label'])
        y = df['Churn_Label']
        for col in X.select_dtypes('object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X.fillna(0, inplace=True)
        Xtr,Xte,ytr,yte = train_test_split(X,y,stratify=y,test_size=.25,random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(random_state=42).fit(Xtr,ytr)
        preds = mdl.predict(Xte)

        from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
        st.metric("Accuracy", f"{accuracy_score(yte,preds):.2%}")

        cm = pd.DataFrame(confusion_matrix(yte,preds),
                          index=["Not Churn","Churn"], columns=["Pred No","Pred Yes"])
        st.write("Confusion matrix", cm)

        yprob = mdl.predict_proba(Xte)[:,1]
        fpr,tpr,_ = roc_curve(yte,yprob)
        aucv = auc(fpr,tpr)
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.plot(fpr,tpr,label=f"AUC={aucv:.2f}")
        ax.plot([0,1],[0,1],'k--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig)

# ───────────── CLUSTERING TAB ────────────
with tabs[4]:
    st.header("🧩 Customer Clustering")
    num = df.select_dtypes('number').drop(columns=['Churn_Label'] if 'Churn_Label' in df else [])
    if num.shape[1] < 2:
        st.warning("Need ≥2 numeric features for clustering.")
    else:
        k = st.slider("Choose k (2-200)", 2, 200, 5)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42).fit(StandardScaler().fit_transform(num))
        df['Cluster'] = kmeans.labels_

        # Elbow (k<=20 for clarity)
        inert = []
        for ki in range(2, min(21,len(df))):
            inert.append(KMeans(n_clusters=ki, random_state=42).fit(StandardScaler().fit_transform(num)).inertia_)
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(); ax.plot(range(2,len(inert)+2), inert, marker='o')
        ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow up to k=20")
        st.pyplot(fig)

        st.subheader("Cluster personas (feature means)")
        st.dataframe(df.groupby('Cluster')[num.columns].mean().round(2))

        st.download_button("Download full data + Cluster", df.to_csv(index=False).encode(),
                           "clustered_customers.csv", "text/csv")

# ───────────── REGRESSION TAB ────────────
with tabs[5]:
    st.header("📈 Value Regression")
    possible = [c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in df]
    if not possible:
        st.warning("No numeric target columns.")
    else:
        tgt = st.selectbox("Target to predict", possible)
        X = df.drop(columns=[tgt]).select_dtypes(exclude='datetime')
        for col in X.select_dtypes('object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        y = df[tgt]
        X.fillna(0,inplace=True)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,random_state=42)
        from sklearn.linear_model import Ridge
        reg = Ridge().fit(Xtr,ytr)
        yp = reg.predict(Xte)

        from sklearn.metrics import r2_score, mean_squared_error
        rmse = np.sqrt(mean_squared_error(yte, yp))   # <-- fixed (no squared=)
        st.metric("R²", f"{r2_score(yte, yp):.2f}")
        st.metric("RMSE", f"{rmse:,.0f}")

        st.subheader("Top drivers (Ridge coef.)")
        coef = pd.Series(reg.coef_, index=X.columns).abs().sort_values(ascending=False).head(10)
        st.bar_chart(coef)

# ───────────── TIME-SERIES TAB ───────────
with tabs[6]:
    st.header("⏳ Monthly Trends")
    if 'Transaction_Date' not in df:
        st.warning("Transaction_Date missing.")
    else:
        df['Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M').astype(str)
        metrics = [c for c in ['Transaction_Amount','Account_Balance','Annual_Income'] if c in df]
        if metrics:
            m = df.groupby('Month')[metrics].mean().reset_index()
            import plotly.express as px
            st.plotly_chart(px.line(m,x='Month',y=metrics,markers=True), use_container_width=True)
        else:
            st.info("No monetary metrics to plot.")

st.markdown("---\n*All set!  Missing a chart?  Check your column names.*")
