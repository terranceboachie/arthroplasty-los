import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Arthroplasty LOS Predictor",
    page_icon="🦴",
    layout="wide",
)

# -------------------------------------------------------------------
# PREMIUM MEDICAL CSS
# -------------------------------------------------------------------
medical_css = """
<style>

body {
    background-color: #f4f7fb;
    font-family: 'Inter', sans-serif;
}

/* Main Title */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    margin-top: 0.3rem;
    color: #0a2a66;
}

/* Subtitle */
.sub-title {
    font-size: 1.1rem;
    text-align: center;
    color: #3f5468;
    margin-bottom: 1.8rem;
}

/* Clinical Card */
.card {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    border: 1px solid #e6ecf5;
    box-shadow: 0px 4px 14px rgba(10, 42, 102, 0.05);
    margin-bottom: 1.4rem;
    transition: all 0.25s ease-in-out;
}

.card:hover {
    box-shadow: 0px 6px 20px rgba(10, 42, 102, 0.08);
}

/* Section Label */
.section-label {
    color: #0a2a66;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* Risk Pills */
.risk-pill {
    padding: 0.38rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}
.risk-low {
    background-color: #e8f6ef;
    color: #10793F;
}
.risk-moderate {
    background-color: #fff5d9;
    color: #C27A00;
}
.risk-high {
    background-color: #fde6e6;
    color: #B91C1C;
}

</style>
"""
st.markdown(medical_css, unsafe_allow_html=True)

# -------------------------------------------------------------------
# CACHED MODEL LOADING
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_meta():
    pipeline = joblib.load("los_xgb_pipeline.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return pipeline, feature_cols

pipeline, feature_cols = load_model_and_meta()

preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def risk_label(prob):
    if prob < 0.20:
        return "Low", "🟢", "risk-low"
    elif prob < 0.50:
        return "Moderate", "🟠", "risk-moderate"
    else:
        return "High", "🔴", "risk-high"

def to_timestamp_int(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d%H%M%S"))

def build_feature_row(
    age, bmi, height_cm, asa, sex_label, race_label,
    anaesthetic_label, op_duration_min, icd10_label
):
    height_m = height_cm / 100
    weight = bmi * (height_m ** 2)

    base_start = datetime(2024, 1, 1, 9, 0)
    opstart_dt = base_start
    opend_dt = base_start + timedelta(minutes=op_duration_min)
    discharge_dt = opend_dt + timedelta(days=2)

    row = {
        "age": age,
        "weight": weight,
        "height": height_cm,
        "asa": asa,
        "emop": 0,
        "opstart_time": to_timestamp_int(opstart_dt),
        "opend_time": to_timestamp_int(opend_dt),
        "discharge_time": to_timestamp_int(discharge_dt),
        "bmi": bmi,
        "op_duration": op_duration_min,
        "sex": sex_label,
        "race": race_label,
        "department": "OS",
        "antype": anaesthetic_label,
        "icd10_pcs": icd10_label,
    }

    return {col: row.get(col, np.nan) for col in feature_cols}

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown('<div class="main-title">Arthroplasty LOS Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A modern clinical tool using explainable machine learning</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab_pred, tab_explain, tab_about = st.tabs(["🔍 Prediction", "📊 Explainability", "ℹ️ About"])

# -------------------------------------------------------------------
# TAB 1 — PREDICTION
# -------------------------------------------------------------------
with tab_pred:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient Inputs</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    # PATIENT INPUTS
    with col_left:
        st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, 65)
        height = c2.number_input("Height (cm)", 140.0, 210.0, 170.0)

        c3, c4 = st.columns(2)
        bmi = c3.number_input("BMI", 15.0, 60.0, 28.0)
        asa = c4.selectbox("ASA Class", [1, 2, 3, 4])

        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        sex = st.selectbox("Sex", ["M", "F"])
        race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])

        st.markdown('<div class="section-label">Surgery Details</div>', unsafe_allow_html=True)
        anaesthetic = st.selectbox("Anaesthetic Type", ["General", "Spinal", "Combined"])
        op_duration = st.number_input("Operative Duration (minutes)", 30, 300, 90)
        icd10 = st.selectbox("ICD-10-PCS Code", ["0WJG0", "0DHS0", "0QB90"])

    # OUTPUT PANEL
    with col_right:
        st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

        if st.button("🚀 Predict LOS Risk", use_container_width=True):
            row = build_feature_row(age, bmi, height, asa, sex, race, anaesthetic, op_duration, icd10)
            df = pd.DataFrame([row])
            st.session_state["input_df"] = df

            proba = float(pipeline.predict_proba(df)[0, 1])
            st.session_state["proba"] = proba

            label, icon, css = risk_label(proba)

            st.markdown(f"""
            <div class="card">
                <h3>{icon} Predicted Prolonged LOS Risk</h3>
                <h1>{proba:.1%}</h1>
                <span class="risk-pill {css}">{label} Risk</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("Enter patient details and click **Predict LOS Risk**.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 2 — EXPLAINABILITY
# -------------------------------------------------------------------
with tab_explain:

    if "input_df" not in st.session_state:
        st.info("Make a prediction first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔎 SHAP Waterfall Explanation")

        # Lazy import for speed
        import shap

        X_trans = preprocessor.transform(st.session_state["input_df"])
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

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# TAB 3 — ABOUT
# -------------------------------------------------------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📘 About This Tool

    This dashboard predicts the risk of **prolonged postoperative length of stay**  
    after elective hip and knee arthroplasty using an XGBoost model trained on  
    perioperative data from the INSPIRE dataset.

    - Target: LOS > 7 days (binary)
    - Model: XGBoost (tuned)
    - Baseline: Logistic Regression
    - Explainability: TreeSHAP
    """)
    st.markdown('</div>', unsafe_allow_html=True)
