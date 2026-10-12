import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.inspection import permutation_importance
from pathlib import Path

st.set_page_config(page_title="TalentPulse AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.2rem;padding-bottom:2.2rem;max-width:1500px;}
.hero {padding:1.35rem 1.5rem;border-radius:16px;background:linear-gradient(115deg,#eaf4f4,#f4f7fa);border:1px solid #d9e7e8;margin-bottom:1.1rem;}
.hero h1 {color:#17324d;margin:0;font-size:2.25rem;line-height:1.2;}
.hero p {color:#526778;margin:.5rem 0 0;font-size:1rem;}
div[data-testid="stMetric"] {padding:14px 16px;border-radius:12px;border:1px solid rgba(128,128,128,.25);min-height:100px;}
div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {color:var(--text-color)!important;}
div[data-testid="stPlotlyChart"] {border-radius:12px;border:1px solid rgba(128,128,128,.15);padding:8px;}
</style>
<div class="hero"><h1>TalentPulse AI</h1>
<p>Explainable workforce intelligence • Internal talent mobility • Retention scenario planning</p></div>
""", unsafe_allow_html=True)

DATA_DIR=Path(__file__).parent/"data"
DATA_DIR.mkdir(exist_ok=True)
SYNTHETIC_PATH=DATA_DIR/"demo_hr_data.csv"

@st.cache_data
def make_demo_data(n=900,seed=17):
    rng=np.random.default_rng(seed)
    departments=np.array(["Engineering","Sales","Operations","Finance","Customer Support","HR"])
    roles={"Engineering":["Software Engineer","Data Analyst","QA Engineer"],"Sales":["Account Executive","Sales Analyst","Sales Associate"],"Operations":["Operations Analyst","Team Coordinator","Process Specialist"],"Finance":["Financial Analyst","Accountant","Risk Analyst"],"Customer Support":["Support Specialist","Customer Success Associate","Support Lead"],"HR":["HR Generalist","Recruiter","Learning Coordinator"]}
    dept=rng.choice(departments,n,p=[.25,.18,.2,.12,.15,.1])
    age=rng.integers(21,61,n); tenure=np.minimum(rng.gamma(2.2,2.1,n),20).round(1)
    income=np.maximum(18000,rng.normal(62000,23000,n)).round(0)
    overtime=rng.choice(["No","Yes"],n,p=[.68,.32]); satisfaction=rng.integers(1,5,n)
    worklife=rng.integers(1,5,n); promotion=np.minimum(rng.gamma(1.8,2,n),12).round(1)
    training=rng.integers(0,61,n); commute=np.minimum(rng.gamma(2,7,n),55).round(1)
    support=rng.integers(1,5,n); role=[rng.choice(roles[d]) for d in dept]
    logit=(-2+.95*(overtime=="Yes")+.42*(satisfaction<=2)+.32*(worklife<=2)+.09*promotion+
           .012*np.maximum(commute-15,0)-.055*tenure-.22*(support>=3)+.006*np.maximum(45000-income,0)/1000+.15*(training<8))
    prob=1/(1+np.exp(-logit)); attr=rng.binomial(1,prob)
    return pd.DataFrame({"EmployeeID":[f"DEMO-{i+1:04d}" for i in range(n)],"Department":dept,"Role":role,"Age":age,
      "TenureYears":tenure,"MonthlyIncome":income,"Overtime":overtime,"JobSatisfaction":satisfaction,
      "WorkLifeBalance":worklife,"YearsSincePromotion":promotion,"TrainingHours":training,"CommuteKm":commute,
      "ManagerSupport":support,"Attrition":np.where(attr==1,"Yes","No")})

if not SYNTHETIC_PATH.exists():
    make_demo_data().to_csv(SYNTHETIC_PATH,index=False)

@st.cache_data
def load_data():
    return pd.read_csv(SYNTHETIC_PATH)

def prep_data(df):
    data=df.copy()
    data.columns=[str(c).strip() for c in data.columns]
    if "Attrition" not in data:
        raise ValueError("Dataset must include an Attrition column with Yes/No labels.")
    data["Attrition"]=data["Attrition"].astype(str).str.strip().str.title()
    data=data[data["Attrition"].isin(["Yes","No"])].copy()
    if len(data)<30:
        raise ValueError("At least 30 valid rows are required.")
    y=(data["Attrition"]=="Yes").astype(int)
    X=data.drop(columns=[c for c in ["Attrition","EmployeeID","EmployeeNumber","ID","id"] if c in data.columns]).copy()
    # Explicitly normalize each feature type before determining pipeline columns.
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c]=X[c].astype("object")
        elif pd.api.types.is_numeric_dtype(X[c]):
            X[c]=pd.to_numeric(X[c],errors="coerce")
        else:
            X[c]=X[c].where(X[c].notna(),"Unknown").astype(str)
    cat_cols=X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num_cols=[c for c in X.columns if c not in cat_cols]
    return data,X,y,cat_cols,num_cols

@st.cache_resource
def train_model(df,model_name):
    data,X,y,cats,nums=prep_data(df)
    stratify=y if y.nunique()==2 and y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=stratify)
    # Determine columns once, based on normalized dtypes; never impute categorical columns with median.
    cats=Xtr.select_dtypes(include=["object","category","bool"]).columns.tolist()
    nums=[c for c in Xtr.columns if c not in cats]
    transformers=[]
    if nums:
        numpipe=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())])
        transformers.append(("num",numpipe,nums))
    if cats:
        catpipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("cat",catpipe,cats))
    pre=ColumnTransformer(transformers,remainder="drop")
    clf=(RandomForestClassifier(n_estimators=220,max_depth=7,min_samples_leaf=4,class_weight="balanced",random_state=42)
         if model_name=="Random Forest" else LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))
    model=Pipeline([("preprocess",pre),("model",clf)])
    model.fit(Xtr,ytr)
    probs=model.predict_proba(Xte)[:,1]
    preds=(probs>=.5).astype(int)
    metrics={"Accuracy":accuracy_score(yte,preds),"Precision":precision_score(yte,preds,zero_division=0),
             "Recall":recall_score(yte,preds,zero_division=0),"F1":f1_score(yte,preds,zero_division=0),
             "ROC-AUC":roc_auc_score(yte,probs) if yte.nunique()==2 else np.nan,
             "PR-AUC":average_precision_score(yte,probs) if yte.nunique()==2 else np.nan,
             "Brier score":brier_score_loss(yte,probs)}
    return model,Xtr,Xte,ytr,yte,probs,preds,metrics,confusion_matrix(yte,preds,labels=[0,1]),data,X

def explain_record(model,row,Xtrain):
    clf=model.named_steps["model"]
    try:
        transformed=model.named_steps["preprocess"].transform(row)
        names=model.named_steps["preprocess"].get_feature_names_out()
        if hasattr(clf,"coef_"):
            arr=transformed.toarray() if hasattr(transformed,"toarray") else np.asarray(transformed)
            vals=clf.coef_[0]*arr[0]
            out=pd.DataFrame({"Feature":names,"Contribution":vals})
            out["Magnitude"]=out["Contribution"].abs()
            return out.sort_values("Magnitude",ascending=False).head(8).sort_values("Contribution")
    except Exception:
        pass
    actual=float(model.predict_proba(row)[:,1][0]); rows=[]
    for col in row.columns:
        changed=row.copy()
        if Xtrain[col].dtype=="object":
            mode=Xtrain[col].mode(dropna=True)
            changed[col]=mode.iloc[0] if len(mode) else "Unknown"
        else:
            changed[col]=Xtrain[col].median()
        rows.append((col,actual-float(model.predict_proba(changed)[:,1][0])))
    return pd.DataFrame(rows,columns=["Feature","Contribution"]).assign(Magnitude=lambda d:d.Contribution.abs()).sort_values("Magnitude",ascending=False).head(8).sort_values("Contribution")

def default_roles():
    return pd.DataFrame([
      {"Role":"People Analytics Analyst","Department":"HR","RequiredSkills":"excel,statistics,python,sql,data visualization"},
      {"Role":"HR Business Partner","Department":"HR","RequiredSkills":"communication,conflict resolution,employee relations,planning"},
      {"Role":"Data Analyst","Department":"Analytics","RequiredSkills":"python,sql,statistics,data visualization,excel"},
      {"Role":"Operations Analyst","Department":"Operations","RequiredSkills":"excel,process mapping,sql,problem solving,planning"},
      {"Role":"Learning & Development Associate","Department":"HR","RequiredSkills":"communication,training design,planning,facilitation"},
      {"Role":"Talent Acquisition Specialist","Department":"HR","RequiredSkills":"interviewing,sourcing,communication,planning"},
      {"Role":"Project Coordinator","Department":"Operations","RequiredSkills":"planning,communication,risk tracking,excel"}])

def match_skills(skills,roles):
    have={x.strip().lower() for x in skills if x.strip()}; out=[]
    for _,r in roles.iterrows():
        need={x.strip().lower() for x in str(r["RequiredSkills"]).split(",") if x.strip()}
        out.append({"Role":r["Role"],"Department":r["Department"],"Match score (%)":round(100*len(have&need)/max(1,len(need)),1),
                    "Matched skills":", ".join(sorted(have&need)) or "None","Skill gaps":", ".join(sorted(need-have)) or "No listed gaps"})
    return pd.DataFrame(out).sort_values("Match score (%)",ascending=False)

with st.sidebar:
    st.header("Workspace")
    st.caption("Demo mode uses synthetic data.")
    page=st.radio("Navigate",["Overview","Attrition Intelligence","Talent Mobility","Intervention Simulator","Model Evaluation","Responsible AI & Data"])
    st.divider()
    model_name=st.selectbox("Model",["Logistic Regression","Random Forest"])
    uploaded=st.file_uploader("Optional HR CSV upload",type=["csv"],help="Only upload data you are authorized to process.")

if uploaded:
    try:
        df=pd.read_csv(uploaded); source="Uploaded CSV"
    except Exception as e:
        st.error(f"CSV could not be read: {e}"); df=load_data(); source="Synthetic demo dataset"
else:
    df=load_data(); source="Synthetic demo dataset"

try:
    model,Xtr,Xte,ytr,yte,probs,preds,metrics,cm,cleaned,Xall=train_model(df,model_name)
    model_error=None
except Exception as e:
    model_error=str(e); cleaned=df.copy()

if page=="Overview":
    st.subheader("Workforce overview")
    st.caption("A high-level view of the selected dataset and the platform workflow.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Records",f"{len(df):,}")
    c2.metric("Departments",df["Department"].nunique() if "Department" in df else "N/A")
    c3.metric("Recorded attrition",f"{(df['Attrition'].astype(str).str.lower()=='yes').mean()*100:.1f}%" if "Attrition" in df else "N/A")
    c4.metric("Data source","Synthetic" if source.startswith("Synthetic") else "Uploaded")
    st.info(f"**Data source:** {source}. Synthetic data is generated for demonstration and does not represent real employees or validated business patterns.")
    a,b=st.columns(2)
    with a:
        if "Department" in df:
            counts=df["Department"].value_counts().rename_axis("Department").reset_index(name="Records")
            st.plotly_chart(px.bar(counts,x="Department",y="Records",title="Records by department",color="Department"),width="stretch")
    with b:
        if "Attrition" in df:
            counts=df["Attrition"].value_counts().rename_axis("Attrition").reset_index(name="Count")
            st.plotly_chart(px.pie(counts,names="Attrition",values="Count",title="Recorded attrition labels",hole=.55),width="stretch")
    st.markdown("#### Platform workflow")
    st.markdown("Historical data → validation → risk estimate + explanation → skill-to-role match → human-reviewed action → outcome monitoring")
    st.markdown("#### Included modules")
    st.markdown("- Attrition prediction and approximate explanations\n- Skills-based internal role matching\n- Retention scenario calculator\n- Held-out model evaluation\n- Data quality and responsible-AI checklist")
    st.caption("Decision-support demonstration only; not a validated employment decision system.")

elif page=="Attrition Intelligence":
    st.subheader("Attrition intelligence")
    st.caption("Model estimates are signals for human review, not statements of employee intent.")
    if model_error:
        st.error(f"Model could not train: {model_error}"); st.stop()
    if "Attrition" not in cleaned:
        st.warning("This view requires an Attrition column with Yes/No labels."); st.stop()
    idx=st.selectbox("Select demo record",range(len(cleaned)),format_func=lambda i:str(cleaned.iloc[i].get("EmployeeID",f"Record {i+1}")))
    row=Xall.iloc[[idx]]
    risk=float(model.predict_proba(row)[:,1][0])
    st.metric("Model-estimated attrition probability",f"{risk*100:.1f}%")
    st.caption("This probability is not certainty and does not indicate an employee's actual intentions.")
    left,right=st.columns(2)
    with left:
        # Convert values to strings to avoid Arrow mixed-type conversion failures.
        details=cleaned.iloc[idx].drop(labels=["Attrition"],errors="ignore").astype(str).rename("Value").rename_axis("Field").reset_index()
        st.dataframe(details,width="stretch",hide_index=True)
    with right:
        explanation=explain_record(model,row,Xtr)
        if not explanation.empty:
            st.plotly_chart(px.bar(explanation,x="Contribution",y="Feature",orientation="h",title="Approximate model explanation"),width="stretch")
    st.markdown("**Human review prompts:** verify data; consider supportive career, workload, or learning options; never use this score alone for discipline or denial of opportunity.")

elif page=="Talent Mobility":
    st.subheader("Internal talent mobility")
    st.caption("Listed skill coverage is not a validated measure of job performance.")
    skills=st.text_input("Employee skills (comma-separated)",value="python, sql, statistics, excel, communication")
    rolefile=st.file_uploader("Optional role catalogue CSV",type=["csv"],key="rolefile")
    roles=default_roles()
    if rolefile:
        try:
            candidate=pd.read_csv(rolefile)
            if {"Role","Department","RequiredSkills"}.issubset(candidate.columns): roles=candidate
            else: st.error("Role CSV needs Role, Department, RequiredSkills columns.")
        except Exception:
            st.error("Could not read role CSV.")
    matches=match_skills(skills.split(","),roles)
    st.dataframe(matches,width="stretch",hide_index=True)
    st.plotly_chart(px.bar(matches.head(7).sort_values("Match score (%)"),x="Match score (%)",y="Role",orientation="h",range_x=[0,100],title="Skill coverage against listed requirements"),width="stretch")
    if len(matches):
        st.success(f"Highest listed skill coverage: {matches.iloc[0]['Role']} ({matches.iloc[0]['Match score (%)']}%).")
        st.write("Skill gaps:",matches.iloc[0]["Skill gaps"])

elif page=="Intervention Simulator":
    st.subheader("Retention intervention simulator")
    st.caption("Scenario arithmetic only. Avoided exits are user-entered assumptions, not causal predictions.")
    a,b=st.columns(2)
    with a:
        exits=st.number_input("Annual voluntary exits",min_value=0,value=40,step=1)
        cost=st.number_input("Estimated replacement cost per exit (₹)",min_value=0,value=180000,step=10000)
        pct=st.slider("Hypothetical exits avoided (%)",0,50,10)
    with b:
        intervention=st.number_input("Intervention cost (₹)",min_value=0,value=250000,step=10000)
        platform=st.number_input("Platform / implementation cost (₹)",min_value=0,value=100000,step=10000)
        productivity=st.number_input("Other validated productivity benefit (₹)",min_value=0,value=0,step=10000)
    avoided=exits*pct/100; gross=avoided*cost+productivity; net=gross-intervention-platform
    a,b,c=st.columns(3)
    a.metric("Hypothetical exits avoided",f"{avoided:.1f}"); b.metric("Gross estimated benefit",f"₹{gross:,.0f}"); c.metric("Net scenario value",f"₹{net:,.0f}")
    scenario=pd.DataFrame({"Scenario":["Conservative","Selected","Optimistic"],"Avoided exits %":[max(0,pct-5),pct,min(50,pct+5)]})
    scenario["Net scenario value"]=scenario["Avoided exits %"]/100*exits*cost+productivity-intervention-platform
    st.plotly_chart(px.bar(scenario,x="Scenario",y="Net scenario value",title="Sensitivity to avoided-exit assumption",text_auto=",.0f"),width="stretch")
    st.dataframe(scenario,width="stretch",hide_index=True)
    st.warning("Replace assumptions with finance/HR inputs and validate outcomes before making business claims.")

elif page=="Model Evaluation":
    st.subheader("Model evaluation")
    if model_error:
        st.error(f"Evaluation unavailable: {model_error}"); st.stop()
    st.info(f"Model: {model_name} · Source: {source}. Synthetic data metrics only test the demo pipeline.")
    a=st.columns(4)
    for i,k in enumerate(["Precision","Recall","F1","ROC-AUC"]): a[i].metric(k,"N/A" if pd.isna(metrics[k]) else f"{metrics[k]:.3f}")
    a=st.columns(3)
    for i,k in enumerate(["Accuracy","PR-AUC","Brier score"]): a[i].metric(k,"N/A" if pd.isna(metrics[k]) else f"{metrics[k]:.3f}")
    st.markdown("#### Confusion matrix")
    st.dataframe(pd.DataFrame(cm,index=["Actual No","Actual Yes"],columns=["Predicted No","Predicted Yes"]),width="stretch")
    st.plotly_chart(px.imshow(cm,text_auto=True,aspect="auto",labels={"x":"Predicted","y":"Actual","color":"Count"},title="Confusion matrix"),width="stretch")
    try:
        importance=permutation_importance(model,Xte,yte,n_repeats=5,random_state=42,scoring="f1")
        impdf=pd.DataFrame({"Feature":Xte.columns,"Importance":importance.importances_mean}).sort_values("Importance",ascending=False).head(12)
        st.plotly_chart(px.bar(impdf.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Permutation importance (F1)"),width="stretch")
    except Exception as e:
        st.caption(f"Permutation importance unavailable: {e}")
    st.caption("A single random split is a demonstration, not robust evidence of real-world performance.")

else:
    st.subheader("Responsible AI & data")
    st.write(f"**Current source:** {source} · **Rows:** {len(df):,}")
    st.warning("Default records are synthetic. Do not treat their patterns as evidence about real employees.")
    st.markdown("#### Responsible-use checklist")
    for i,item in enumerate(["Process HR data only with authorization and a defined purpose.","Use synthetic/de-identified demo data.","Never make employment decisions solely from model output.","Provide human review and correction processes.","Evaluate subgroup performance where lawful and statistically meaningful.","Document features, versions, limitations, and evaluation period.","Revalidate when workforce patterns change.","Association is not causation."]):
        st.checkbox(item,key=f"rai_{i}")
    dq=pd.DataFrame({"Column":df.columns,"Missing values":df.isna().sum().values,"Missing %":(df.isna().mean().values*100).round(1),"Unique values":[df[c].nunique(dropna=True) for c in df.columns]})
    st.markdown("#### Data quality scan")
    st.dataframe(dq,width="stretch",hide_index=True)
    st.download_button("Download current dataset CSV",df.to_csv(index=False).encode(),"talentpulse_current_dataset.csv","text/csv")
    st.download_button("Download data quality report",dq.to_csv(index=False).encode(),"talentpulse_data_quality_report.csv","text/csv")
