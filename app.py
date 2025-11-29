import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Arthroplasty LOS Explorer",
    page_icon="🦴",
    layout="wide",
)

# =========================================================
# MODERN CSS
# =========================================================
modern_css = """
<style>
    body {
        background-color: #f7f9fc;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(90deg,#0048ff,#6a00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 400;
        color: #5e6572;
        margin-bottom: 1.6rem;
    }
    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 1rem;
        background: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: 0.2s ease-in-out;
    }
    .card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    }
    .section-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #7b7f87;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .risk-pill {
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .risk-low {
        background-color: #e6f4ea;
        color: #137333;
    }
    .risk-moderate {
        background-color: #fff4ce;
        color: #9c6700;
    }
    .risk-high {
        background-color: #fde8e7;
        color: #d93025;
    }
</style>
"""
st.markdown(modern_css, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 Model Overview")
    st.write(
        "This dashboard estimates the risk of **prolonged postoperative stay** "
        "after elective hip/knee arthroplasty using an explainable XGBoost model."
    )
    st.markdown("### 📘 Quick Steps")
    st.markdown(
        """
1. Go to **Prediction**
2. Enter basic patient info
3. Click **Predict Risk**
4. Open **Explainability** tab
"""
    )
    st.markdown("---")
    st.caption("⚠️ Research prototype only — not for clinical use.")

# =========================================================
# LOAD MODEL & METADATA
# =========================================================
@st.cache_resource
def load_model_and_meta():
    pipeline = joblib.load("los_xgb_pipeline.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return pipeline, feature_cols

pipeline, feature_cols = load_model_and_meta()

# Your pipeline uses 'preprocess' and 'model'
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# Get numeric / categorical feature names from the ColumnTransformer
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# =========================================================
# HELPERS
# =========================================================
def risk_label(prob):
    if prob < 0.20:
        return "Low", "🟢", "risk-low"
    elif prob < 0.50:
        return "Moderate", "🟠", "risk-moderate"
    else:
        return "High", "🔴", "risk-high"


def to_timestamp_int(dt: datetime) -> int:
    """Convert datetime to 14-digit integer YYYYMMDDHHMMSS."""
    return int(dt.strftime("%Y%m%d%H%M%S"))


def build_feature_row(
    age,
    bmi,
    height_cm,
    asa,
    sex_label,
    race_label,
    anaesthetic_label,
    op_duration_min,
    icd10_label,
):
    """
    Build a single row (dict) matching feature_cols expected by the pipeline.
    We auto-generate weight, timestamps, department, emop, discharge_time.
    """

    # Weight from BMI and height
    height_m = height_cm / 100.0
    weight = bmi * (height_m ** 2)

    # Simple fixed elective setup
    emop = 0  # elective only
    department = "OS"  # orthopaedic surgery default

    # Generate a synthetic timeline for opstart, opend, discharge
    base_start = datetime(2024, 1, 1, 9, 0, 0)  # arbitrary but consistent
    opstart_dt = base_start
    opend_dt = opstart_dt + timedelta(minutes=op_duration_min)
    discharge_dt = opend_dt + timedelta(days=2)  # default LOS ~2 days

    opstart_time = to_timestamp_int(opstart_dt)
    opend_time = to_timestamp_int(opend_dt)
    discharge_time = to_timestamp_int(discharge_dt)

    # Map UI labels → training labels
    sex = sex_label  # already "M" or "F"
    race = race_label  # should match training ["Asian","White","Black","Other"]
    antype = anaesthetic_label  # should match your training values
    icd10_pcs = icd10_label

    # Now build the row for ALL features
    row = {}
    for col in feature_cols:
        if col == "age":
            row[col] = age
        elif col == "weight":
            row[col] = weight
        elif col == "height":
            row[col] = height_cm
        elif col == "asa":
            row[col] = asa
        elif col == "emop":
            row[col] = emop
        elif col == "opstart_time":
            row[col] = opstart_time
        elif col == "opend_time":
            row[col] = opend_time
        elif col == "discharge_time":
            row[col] = discharge_time
        elif col == "bmi":
            row[col] = bmi
        elif col == "op_duration":
            row[col] = op_duration_min
        elif col == "sex":
            row[col] = sex
        elif col == "race":
            row[col] = race
        elif col == "department":
            row[col] = department
        elif col == "antype":
            row[col] = antype
        elif col == "icd10_pcs":
            row[col] = icd10_pcs
        else:
            # If there are any unexpected features, default to NaN
            row[col] = np.nan

    return row

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Arthroplasty LOS Risk Explorer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">A modern, explainable ML tool for postoperative length-of-stay risk.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
tab_pred, tab_explain, tab_about = st.tabs(
    ["🔍 Prediction", "📊 Explainability", "ℹ️ About"]
)

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab_pred:
    st.markdown(
        '<div class="section-label">Patient Inputs</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="card">Enter basic perioperative details below. '
        'Additional model features are computed automatically.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([1.3, 1])

    # LEFT: USER INPUTS
    with col_left:
        st.markdown(
            '<div class="section-label">Patient Profile</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, 65, step=1)
        height_cm = c2.number_input("Height (cm)", 140.0, 210.0, 170.0, step=0.5)

        c3, c4 = st.columns(2)
        bmi = c3.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, step=0.1)
        asa = c4.selectbox("ASA Class", [1, 2, 3, 4, 5], index=1)

        st.markdown(
            '<div class="section-label">Demographics</div>', unsafe_allow_html=True
        )
        c5, c6 = st.columns(2)
        sex_label = c5.selectbox("Sex", ["M", "F"])
        race_label = c6.selectbox("Race", ["White", "Black", "Asian", "Other"])

        st.markdown(
            '<div class="section-label">Procedure & Anaesthesia</div>',
            unsafe_allow_html=True,
        )
        c7, c8 = st.columns(2)
        procedure_type = c7.selectbox(
            "Procedure Type (for info only)",
            ["Total Hip Arthroplasty", "Total Knee Arthroplasty"],
        )
        anaesthetic_label = c8.selectbox(
            "Anaesthetic Type", ["General", "Spinal", "Combined"]
        )

        st.markdown(
            '<div class="section-label">Operative Parameters</div>',
            unsafe_allow_html=True,
        )
        c9, c10 = st.columns(2)
        op_duration_min = c9.number_input(
            "Operative duration (minutes)", 30, 300, 90, step=5
        )
        icd10_label = c10.selectbox(
            "ICD-10-PCS Code",
            ["0WJG0", "0DHS0", "0QB90"],
        )

    # RIGHT: PREDICTION
    with col_right:
        st.markdown(
            '<div class="section-label">Prediction Output</div>',
            unsafe_allow_html=True,
        )

        if st.button("🚀 Predict Risk", use_container_width=True):
            # Build features
            row = build_feature_row(
                age=age,
                bmi=bmi,
                height_cm=height_cm,
                asa=asa,
                sex_label=sex_label,
                race_label=race_label,
                anaesthetic_label=anaesthetic_label,
                op_duration_min=op_duration_min,
                icd10_label=icd10_label,
            )
            input_df = pd.DataFrame([row], columns=feature_cols)

            # Store in session for SHAP
            st.session_state["input_df"] = input_df

            # Predict
            proba = pipeline.predict_proba(input_df)[0, 1]
            st.session_state["proba"] = float(proba)

            label, icon, css = risk_label(proba)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {icon} Predicted Long-Stay Risk")
            st.markdown(f"<h2>{proba:.1%}</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='risk-pill {css}'>{label} risk</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(proba, 0), 1))
            st.caption("Risk thresholds: <20% Low • 20–50% Moderate • >50% High")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Enter patient details and click **Predict Risk**.")

# =========================================================
# TAB 2 — EXPLAINABILITY
# =========================================================
with tab_explain:
    st.markdown(
        '<div class="section-label">Explainability</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="card">This section explains the most recent prediction using SHAP values.</div>',
        unsafe_allow_html=True,
    )

    if "input_df" not in st.session_state:
        st.info("Please make a prediction first in the **Prediction** tab.")
    else:
        input_df = st.session_state["input_df"]

        with st.spinner("Computing SHAP values..."):
            # Transform using the same preprocessor
            X_trans = preprocessor.transform(input_df)
            X_trans = np.array(X_trans, dtype=np.float64)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)

            # Get OHE feature names
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features)
            feature_names = np.concatenate([numeric_features, cat_names])

            # Build SHAP Explanation for first (and only) sample
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

# =========================================================
# TAB 3 — ABOUT
# =========================================================
with tab_about:
    st.markdown(
        '<div class="section-label">About</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
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
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
<div class="card">
<h3>⚠️ Important Notice</h3>
This dashboard is a <strong>research prototype</strong> created for a postgraduate
dissertation on postoperative recovery prediction. It is not validated for
clinical use and must not guide real medical decisions.
</div>
""",
        unsafe_allow_html=True,
    )

!streamlit run app.py --server.port 8501 --server.headless true
from pyngrok import ngrok
public_url = ngrok.connect(8501)
public_url
