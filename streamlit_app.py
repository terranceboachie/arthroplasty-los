import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Arthroplasty LOS Explorer",
    page_icon="🦴",
    layout="wide",
)

# ---------------------------------------------------------
# MODERN CSS (unchanged)
# ---------------------------------------------------------
modern_css = """
<style>
    body { background-color: #f7f9fc; }
    .main-title {
        font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(90deg,#0048ff,#6a00ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { font-size: 1.05rem; color: #5e6572; margin-bottom: 1.6rem; }
    .card {
        padding: 1.2rem 1.4rem; border-radius: 1rem; background: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: 0.2s ease-in-out;
    }
    .card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    }
    .section-label {
        font-size: 0.82rem; font-weight: 600; color: #7b7f87;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .risk-pill {
        padding: 0.35rem 0.9rem; border-radius: 999px;
        font-weight: 600; font-size: 0.95rem; display: inline-block;
    }
    .risk-low { background-color: #e6f4ea; color: #137333; }
    .risk-moderate { background-color: #fff4ce; color: #9c6700; }
    .risk-high { background-color: #fde8e7; color: #d93025; }
</style>
"""
st.markdown(modern_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Model Overview")
    st.write(
        "This dashboard estimates the risk of **prolonged postoperative stay** "
        "after elective hip/knee arthroplasty using an explainable XGBoost model."
    )
    st.markdown("### 📘 Quick Steps")
    st.markdown("""
1. Go to **Prediction**
2. Enter basic patient info
3. Click **Predict Risk**
4. Open **Explainability** tab
""")
    st.markdown("---")
    st.caption("⚠️ Research prototype only — not for clinical use.")

# ---------------------------------------------------------
# CACHE MODEL (MAJOR SPEED BOOST)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    pipeline = joblib.load("los_xgb_pipeline.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    return pipeline, feature_cols, preprocessor, model

pipeline, feature_cols, preprocessor, model = load_model()

numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def risk_label(prob):
    if prob < 0.20:
        return "Low", "🟢", "risk-low"
    elif prob < 0.50:
        return "Moderate", "🟠", "risk-moderate"
    else:
        return "High", "🔴", "risk-high"


def to_timestamp_int(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d%H%M%S"))


def build_feature_row(age, bmi, height_cm, asa, sex_label, race_label,
                      anaesthetic_label, op_duration_min, icd10_label):

    height_m = height_cm / 100
    weight = bmi * (height_m ** 2)

    emop = 0
    department = "OS"

    base_start = datetime(2024, 1, 1, 9)
    opstart_dt = base_start
    opend_dt = opstart_dt + timedelta(minutes=op_duration_min)
    discharge_dt = opend_dt + timedelta(days=2)

    opstart_time = to_timestamp_int(opstart_dt)
    opend_time = to_timestamp_int(opend_dt)
    discharge_time = to_timestamp_int(discharge_dt)

    row = {col: np.nan for col in feature_cols}

    mapping = {
        "age": age, "bmi": bmi, "height": height_cm, "weight": weight,
        "asa": asa, "emop": emop,
        "opstart_time": opstart_time, "opend_time": opend_time,
        "discharge_time": discharge_time,
        "op_duration": op_duration_min,
        "sex": sex_label, "race": race_label,
        "department": department, "antype": anaesthetic_label,
        "icd10_pcs": icd10_label,
    }

    for col, val in mapping.items():
        if col in row:
            row[col] = val

    return row

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<div class="main-title">Arthroplasty LOS Risk Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A modern, explainable ML tool for postoperative length-of-stay risk.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_pred, tab_explain, tab_about = st.tabs(
    ["🔍 Prediction", "📊 Explainability", "ℹ️ About"]
)

# ---------------------------------------------------------
# TAB 1 — PREDICTION
# ---------------------------------------------------------
with tab_pred:
    st.markdown('<div class="section-label">Patient Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Enter perioperative details below. Additional model features are computed automatically.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, 65)
        height_cm = c2.number_input("Height (cm)", 140.0, 210.0, 170.0, step=0.5)

        c3, c4 = st.columns(2)
        bmi = c3.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, step=0.1)
        asa = c4.selectbox("ASA Class", [1, 2, 3, 4, 5], index=1)

        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        sex_label = c5.selectbox("Sex", ["M", "F"])
        race_label = c6.selectbox("Race", ["White", "Black", "Asian", "Other"])

        st.markdown('<div class="section-label">Procedure & Anaesthesia</div>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        procedure_type = c7.selectbox("Procedure Type", ["Total Hip Arthroplasty", "Total Knee Arthroplasty"])
        anaesthetic_label = c8.selectbox("Anaesthetic Type", ["General", "Spinal", "Combined"])

        st.markdown('<div class="section-label">Operative Parameters</div>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        op_duration_min = c9.number_input("Operative duration (minutes)", 30, 300, 90, step=5)
        icd10_label = c10.selectbox("ICD-10-PCS Code", ["0WJG0", "0DHS0", "0QB90"])

    with col_right:
        st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

        if st.button("🚀 Predict Risk", use_container_width=True):
            row = build_feature_row(
                age, bmi, height_cm, asa,
                sex_label, race_label, anaesthetic_label,
                op_duration_min, icd10_label
            )
            input_df = pd.DataFrame([row], columns=feature_cols)

            proba = float(pipeline.predict_proba(input_df)[0, 1])
            st.session_state["input_df"] = input_df
            st.session_state["proba"] = proba

            label, icon, css = risk_label(proba)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {icon} Predicted Long-Stay Risk")
            st.markdown(f"<h2>{proba:.1%}</h2>", unsafe_allow_html=True)
            st.markdown(f"<span class='risk-pill {css}'>{label} risk</span>", unsafe_allow_html=True)
            st.progress(proba)
            st.caption("Risk thresholds: <20% Low • 20–50% Moderate • >50% High")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Enter patient details and click **Predict Risk**.")

# ---------------------------------------------------------
# TAB 2 — EXPLAINABILITY (LAZY LOADED FOR SPEED)
# ---------------------------------------------------------
with tab_explain:
    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">This section explains the latest prediction using SHAP values.</div>', unsafe_allow_html=True)

    if "input_df" not in st.session_state:
        st.info("Please make a prediction first in the **Prediction** tab.")
    else:
        # 🔥 Lazy import SHAP only here
        with st.spinner("Computing SHAP values..."):
            import shap
            import matplotlib.pyplot as plt

            input_df = st.session_state["input_df"]

            X_trans = preprocessor.transform(input_df)
            X_trans = np.array(X_trans, dtype=np.float64)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)

            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features)
            feature_names = np.concatenate([numeric_features, cat_names])

            shap_exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_trans[0],
                feature_names=feature_names,
            )

            st.write("")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔎 SHAP Waterfall Plot")

            fig, ax = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3 — ABOUT
# ---------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>📘 Model Summary</h3>
<ul>
<li>Data source: INSPIRE perioperative dataset</li>
<li>Target: prolonged LOS (>7 days, binary)</li>
<li>Models: Logistic Regression baseline + XGBoost</li>
<li>Pipeline: Imputation, scaling, One-Hot Encoding, XGBoost</li>
<li>Explainability: TreeSHAP (global + local)</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.write("")
    st.markdown("""
<div class="card">
<h3>⚠️ Important Notice</h3>
This dashboard is a <strong>research prototype</strong> created for a postgraduate
dissertation project. It is not validated for clinical use.
</div>
""", unsafe_allow_html=True)

