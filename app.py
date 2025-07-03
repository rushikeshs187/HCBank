import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

# ───────────────────────────── TABS ──────────────────────────────
tab_names = [
    "🎯 Objectives",
    "💡 How to Use",
    "📊 Visual Insights",
    "🤖 Churn Classification",
    "🧩 Customer Clustering",
    "📈 Value Regression",
    "⏳ Time-Series"
]
tabs = st.tabs(tab_names)

# ──────────────────────────── SIDEBAR ────────────────────────────
with st.sidebar:
    st.title("🏦 Bank Analytics")
    up_file = st.file_uploader("Upload Excel → sheet **Cleaned data**", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload data  2. Explore tabs  3. Download insights", icon="ℹ️")

# ─────────────────────────── LOAD DATA ───────────────────────────
if up_file is None:
    st.warning("Upload a file to unlock the dashboard.", icon="⚠️")
    st.stop()

try:
    df = pd.read_excel(up_file, sheet_name="Cleaned data")
except Exception as e:
    st.error(f"❌ Couldn’t read sheet **Cleaned data** – {e}")
    st.stop()

# ───────────────────────── OBJECTIVES TAB ────────────────────────
with tabs[0]:
    st.markdown("## 🎯 Dashboard Objectives")
    st.markdown("""
- **Predict churn** and act before customers leave.  
- **Estimate satisfaction / value** to prioritise service.  
- **Cluster customers** into actionable personas (up to 200 groups).  
- **Track monthly trends** in revenue & sentiment.  
    """)

# ───────────────────────── HOW-TO TAB ────────────────────────────
with tabs[1]:
    st.markdown("## 💡 How to Use")
    st.markdown("""
1. **Filter** in each tab to zoom into target cohorts.  
2. **Hover** charts for details; click legends to isolate series.  
3. **Download** predictions/segments for campaigns.  
    """)

# ─────────────────────── VISUAL INSIGHTS TAB ─────────────────────
with tabs[2]:
    st.header("📊 Visual Insights")
    import plotly.express as px, matplotlib.pyplot as plt, seaborn as sns

    # ── dynamic filters
    cols = df.columns
    filt_gender  = st.multiselect("Gender",      df['Gender'].unique()      if 'Gender'      in cols else [], default=None)
    filt_type    = st.multiselect("Account Type",df['Account_Type'].unique()if 'Account_Type'in cols else [], default=None)
    filt_region  = st.multiselect("Region",      df['Region'].unique()      if 'Region'      in cols else [], default=None)

    view = df.copy()
    if filt_gender : view = view[view['Gender'].isin(filt_gender)]
    if filt_type   : view = view[view['Account_Type'].isin(filt_type)]
    if filt_region : view = view[view['Region'].isin(filt_region)]

    st.success(f"Records after filter: {len(view)}")

    k1,k2,k3 = st.columns(3)
    k1.metric("Churn %", f"{view['Churn_Label'].mean()*100: .1f}%" if 'Churn_Label' in cols else "N/A")
    k2.metric("Avg Satisfaction", f"{view['Customer_Satisfaction_Score'].mean(): .2f}" if 'Customer_Satisfaction_Score' in cols else "N/A")
    k3.metric("Avg Balance",      f"{view['Account_Balance'].mean():,.0f}"  if 'Account_Balance' in cols else "N/A")

    # ── churn rate by account
    if {'Account_Type','Churn_Label'}<=set(cols):
        st.subheader("Churn rate by account type")
        ch=df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        st.plotly_chart(px.bar(ch,x='Account_Type',y='Churn_Label',text_auto='.1%',color='Churn_Label',color_continuous_scale='Reds'),use_container_width=True)

    # ── monthly income & balance trend
    if {'Transaction_Date','Account_Balance','Annual_Income'}<=set(cols):
        st.subheader("Monthly income vs balance trend")
        tmp=df.copy()
        tmp['Month']=pd.to_datetime(tmp['Transaction_Date']).dt.to_period('M').astype(str)
        m=tmp.groupby('Month')[['Annual_Income','Account_Balance']].mean().reset_index()
        st.plotly_chart(px.line(m,x='Month',y=['Annual_Income','Account_Balance'],markers=True),use_container_width=True)

# ────────────────────── CLASSIFICATION TAB ───────────────────────
with tabs[3]:
    st.header("🤖 Churn Classification")
    needed={'Churn_Label'}
    if not needed<=set(df.columns):
        st.warning("Churn_Label missing – upload full dataset.")
    else:
        drop=['Customer_ID','Transaction_Date','Account_Open_Date','Last_Transaction_Date','Churn_Timeframe']
        X=df.drop(columns=[c for c in drop if c in df.columns]+['Churn_Label'])
        y=df['Churn_Label']
        X=X.select_dtypes(exclude='datetime')
        for c in X.select_dtypes('object').columns:
            X[c]=LabelEncoder().fit_transform(X[c].astype(str))
        X=X.fillna(0)
        Xtr,Xte,ytr,yte=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        mdl=RandomForestClassifier(random_state=42).fit(Xtr,ytr)
        preds=mdl.predict(Xte)
        from sklearn.metrics import accuracy_score,confusion_matrix
        st.metric("Accuracy",f"{accuracy_score(yte,preds):.2%}")
        st.write("Confusion matrix",pd.DataFrame(confusion_matrix(yte,preds),
                  index=["Not Churned","Churned"],columns=["Pred NC","Pred C"]))

# ─────────────────────── CLUSTERING TAB ──────────────────────────
with tabs[4]:
    st.header("🧩 Customer Clustering")
    num=df.select_dtypes('number').drop(columns=['Churn_Label'] if 'Churn_Label'in df else [])
    if num.shape[1]<2:
        st.warning("Need at least 2 numeric features.")
    else:
        k=st.slider("Clusters (2-200)",2,200,5)
        from sklearn.cluster import KMeans
        kmeans=KMeans(n_clusters=k,random_state=42).fit(StandardScaler().fit_transform(num))
        df['Cluster']=kmeans.labels_
        st.write("Top-line persona (mean by cluster):")
        st.dataframe(df.groupby('Cluster')[num.columns].mean().round(2))
        st.download_button("Download full data with cluster labels",
                           df.to_csv(index=False).encode(), "clustered_data.csv","text/csv")

# ─────────────────────── REGRESSION TAB ──────────────────────────
with tabs[5]:
    st.header("📈 Value Regression")
    targets=[c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in df]
    if not targets:
        st.warning("No numeric target columns available.")
    else:
        tgt=st.selectbox("Predict target",targets)
        X=df.drop(columns=[tgt])
        X=X.select_dtypes(exclude='datetime')
        for c in X.select_dtypes('object').columns:
            X[c]=LabelEncoder().fit_transform(X[c].astype(str))
        X=X.fillna(0)
        y=df[tgt]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
        from sklearn.linear_model import Ridge
        reg=Ridge().fit(Xtr,ytr)
        yp=reg.predict(Xte)
        from sklearn.metrics import r2_score,mean_squared_error
        st.metric("R²",f"{r2_score(yte,yp):.2f}")
        st.metric("RMSE",f"{mean_squared_error(yte,yp,squared=False):,.0f}")
        st.bar_chart(pd.Series(reg.coef_,index=X.columns).nlargest(10))

# ───────────────────── TIME-SERIES TAB ───────────────────────────
with tabs[6]:
    st.header("⏳ Monthly Trends")
    if 'Transaction_Date' not in df:
        st.warning("Transaction_Date missing.")
    else:
        df['Month']=pd.to_datetime(df['Transaction_Date']).dt.to_period('M').astype(str)
        mets=[c for c in ['Transaction_Amount','Account_Balance','Annual_Income'] if c in df]
        if mets:
            m=df.groupby('Month')[mets].mean().reset_index()
            import plotly.express as px
            st.plotly_chart(px.line(m,x='Month',y=mets,markers=True),use_container_width=True)
        else:
            st.info("No monetary columns to plot.")

st.markdown("---\n*Dashboard: refined & insight-rich – missing features? ensure your columns exist.*")
