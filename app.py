"""
app.py  -  AI-Based Diabetes Prediction System
Student : Nafize Ali  |  AI & Data Science  |  Minor Project
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from src.prediction import predict_patient_risk

# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System | Nafize Ali",
    page_icon="stethoscope",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# INJECT CSS  -  uses unsafe_allow_html=True so the browser reads it as CSS,
#               NOT as visible text.
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #f5f7fa;
    color: #1a202c;
}

/* ---------------------------------------------------------------
   SIDEBAR  -  scoped tightly so it does NOT bleed into main area
--------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
}
/* Only target direct text nodes inside the sidebar */
[data-testid="stSidebar"] > div p,
[data-testid="stSidebar"] > div span,
[data-testid="stSidebar"] > div small,
[data-testid="stSidebar"] > div label {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] > div h1,
[data-testid="stSidebar"] > div h2,
[data-testid="stSidebar"] > div h3 {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #334155;
}

/* Main content padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Page heading */
.pg-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.pg-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 24px;
}

/* White card */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94a3b8;
    margin-bottom: 12px;
}

/* Stat metric boxes */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
}
.stat-box {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.stat-val {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1e40af;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #94a3b8;
    margin-top: 5px;
}

/* Workflow steps */
.wf-container {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 18px 0;
}
.wf-step {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 10px;
    text-align: center;
}
.wf-num {
    font-size: 0.72rem;
    font-weight: 700;
    color: #1e40af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: block;
    margin-bottom: 5px;
}
.wf-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #0f172a;
}
.wf-arrow {
    align-self: center;
    color: #cbd5e1;
    font-size: 1rem;
    padding: 0 8px;
    flex-shrink: 0;
}

/* Prediction result cards */
.res-pos {
    background: #fff5f5;
    border: 1px solid #fca5a5;
    border-left: 5px solid #dc2626;
    border-radius: 8px;
    padding: 22px 24px;
    margin-top: 16px;
}
.res-neg {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 22px 24px;
    margin-top: 16px;
}
.res-tag {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94a3b8;
    margin-bottom: 2px;
}
.res-class-pos {
    font-size: 1.5rem;
    font-weight: 700;
    color: #b91c1c;
    margin: 4px 0 14px;
}
.res-class-neg {
    font-size: 1.5rem;
    font-weight: 700;
    color: #15803d;
    margin: 4px 0 14px;
}
.res-prob {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
}
.res-note {
    font-size: 0.83rem;
    color: #64748b;
    margin-top: 10px;
    line-height: 1.6;
}

/* Input group label */
.inp-group-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    padding-bottom: 6px;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 4px;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.55;
    border-top: 1px solid #f1f5f9;
    padding-top: 14px;
    margin-top: 18px;
}

/* Tech badges */
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 4px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}

/* About list */
.ab-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f8fafc;
    font-size: 0.87rem;
}
.ab-key { color: #64748b; }
.ab-val { font-weight: 600; color: #0f172a; }

/* Model performance note */
.perf-note {
    font-size: 0.86rem;
    color: #475569;
    line-height: 1.65;
}

/* Section heading */
.section-heading {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #475569;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 18px;
}

/* ---------------------------------------------------------------
   WIDGET LABELS  -  dark charcoal (#1F2937), medium/bold weight, 14-16px
--------------------------------------------------------------- */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] div,
[data-testid="stWidgetLabel"] *,
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] *,
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label *,
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label *,
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label *,
.stNumberInput label,
.stNumberInput label *,
.stSelectbox label,
.stSelectbox label * {
    color: #1f2937 !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stWidgetLabel"] {
    margin-bottom: 6px !important;
}

/* ---------------------------------------------------------------
   INPUT BOXES  -  white background, dark text, subtle grey border
--------------------------------------------------------------- */
/* BaseWeb input wrapper & containers */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-testid="stNumberInputContainer"],
div[data-testid="stNumberInput"] > div > div,
div[data-testid="stTextInput"] > div > div {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

/* Focus outline */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within,
div[data-testid="stNumberInputContainer"]:focus-within,
div[data-testid="stNumberInput"] > div > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

/* Text inside number & text inputs */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
.stNumberInput input,
.stTextInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border: none !important;
    outline: none !important;
}

/* Number input increment & decrement buttons (+ / -) */
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInputContainer"] button {
    background-color: #f9fafb !important;
    color: #374151 !important;
    border: none !important;
    border-left: 1px solid #e5e7eb !important;
}
div[data-testid="stNumberInput"] button:hover,
div[data-testid="stNumberInputContainer"] button:hover {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
}
div[data-testid="stNumberInput"] button svg,
div[data-testid="stNumberInputContainer"] button svg {
    fill: #374151 !important;
}

/* Selectbox styling */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    color: #111827 !important;
}
div[data-baseweb="select"]:focus-within > div {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #111827 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] svg {
    fill: #4b5563 !important;
}

/* Dropdown popover menu */
ul[data-baseweb="menu"],
div[data-baseweb="popover"] ul {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 6px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
}
ul[data-baseweb="menu"] li,
div[data-baseweb="popover"] li {
    color: #1f2937 !important;
    background-color: #ffffff !important;
}
ul[data-baseweb="menu"] li:hover,
div[data-baseweb="popover"] li:hover {
    background-color: #f3f4f6 !important;
    color: #1e40af !important;
}

/* ---------------------------------------------------------------
   BUTTON
--------------------------------------------------------------- */
.stButton > button {
    background-color: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
}

/* Form border reset */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_data():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "model", "diabetes_model.pkl"),
        os.path.join(base_dir, "diabetes_model.pkl"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    viz_dir    = os.path.join(base_dir, "visualizations")
    if not os.path.exists(model_path):
        return None, viz_dir
    try:
        return joblib.load(model_path), viz_dir
    except Exception:
        return None, viz_dir


model_data, viz_dir = load_model_data()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Nafize Ali")
    st.caption("Artificial Intelligence & Data Science")
    st.caption("B.Tech Minor Project")
    st.markdown("---")
    nav_options = ["Home", "Prediction", "Data Insights", "Model Performance", "About"]
    default_idx = 0
    if "_nav" in st.session_state and st.session_state["_nav"] in nav_options:
        default_idx = nav_options.index(st.session_state["_nav"])

    page = st.radio(
        "Navigate",
        nav_options,
        index=default_idx,
        label_visibility="collapsed",
        key="main_nav_radio",
    )
    st.markdown("---")
    st.markdown("**Model**")
    if model_data:
        st.caption(f"Algorithm: **{model_data.get('model_name', 'Random Forest')}**")
        acc = model_data.get("best_metrics", {}).get("Accuracy", None)
        auc = model_data.get("best_metrics", {}).get("ROC-AUC", None)
        if acc:
            st.caption(f"Test Accuracy: **{round(float(acc)*100, 1)}%**")
        if auc:
            st.caption(f"ROC-AUC: **{auc}**")
    else:
        st.caption("Random Forest Classifier")
    st.markdown("---")
    st.caption("For academic purposes only. Not a clinical diagnostic tool.")


# ===========================================================================
# PAGE: HOME
# ===========================================================================
if page == "Home":

    # Hero
    st.markdown('<p class="pg-title">AI-Based Diabetes Prediction System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub">Supervised Machine Learning Based Diabetes Risk Prediction</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This project analyses selected health-related attributes and uses supervised machine "
        "learning to estimate the likelihood of diabetes."
    )

    st.markdown("---")

    # About the project
    st.markdown("#### About the Project")
    st.markdown(
        "The AI-Based Diabetes Prediction System is a minor project developed as part of the "
        "B.Tech Artificial Intelligence & Data Science curriculum. "
        "It uses the Pima Indians Diabetes Dataset to train and evaluate machine learning models "
        "that can classify whether a patient is likely to have diabetes based on health indicators "
        "such as glucose level, BMI, blood pressure, and age. "
        "The project demonstrates a complete ML pipeline — from data preprocessing and model training "
        "to an interactive web application built with Streamlit."
    )

    st.markdown("---")

    # How it works
    st.markdown("#### How the System Works")
    st.markdown("""
<div class="wf-container">
  <div class="wf-step">
    <span class="wf-num">01</span>
    <span class="wf-label">Patient Information</span>
  </div>
  <div class="wf-arrow">&rarr;</div>
  <div class="wf-step">
    <span class="wf-num">02</span>
    <span class="wf-label">Data Preprocessing</span>
  </div>
  <div class="wf-arrow">&rarr;</div>
  <div class="wf-step">
    <span class="wf-num">03</span>
    <span class="wf-label">Machine Learning</span>
  </div>
  <div class="wf-arrow">&rarr;</div>
  <div class="wf-step">
    <span class="wf-num">04</span>
    <span class="wf-label">Risk Prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Objectives
    col_obj, col_cta = st.columns([3, 1], gap="large")

    with col_obj:
        st.markdown("#### Project Objectives")
        st.markdown(
            "- Predict diabetes classification using machine learning\n"
            "- Analyse health-related factors that influence diabetes risk\n"
            "- Evaluate classification performance across multiple models\n"
            "- Provide an interactive prediction interface for patient data"
        )

    with col_cta:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Prediction \u2192", use_container_width=True):
            st.session_state["_nav"] = "Prediction"
            st.rerun()


# ===========================================================================
# PAGE: PREDICTION
# ===========================================================================
elif page == "Prediction":

    st.markdown('<p class="pg-title">Diabetes Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub">Enter the patient\'s health information to estimate '
        'diabetes risk using the trained machine-learning model.</p>',
        unsafe_allow_html=True,
    )

    col_form, col_guide = st.columns([2.3, 1], gap="large")

    with col_form:
        st.markdown('<p class="section-heading">PATIENT INFORMATION</p>', unsafe_allow_html=True)

        with st.form("pred_form"):
            fc1, fc2 = st.columns(2, gap="medium")

            with fc1:
                age_val = st.number_input(
                    "Age (years)",
                    min_value=18, max_value=100, value=35, step=1,
                )
                bmi_val = st.number_input(
                    "BMI (kg/m\u00b2)",
                    min_value=12.0, max_value=65.0, value=26.4, step=0.1, format="%.1f",
                )
                act_val = st.selectbox(
                    "Physical Activity Level",
                    ["Moderate", "Low", "High"], index=0,
                )

            with fc2:
                bp_val = st.number_input(
                    "Blood Pressure (mm Hg)",
                    min_value=40, max_value=160, value=75, step=1,
                )
                glu_val = st.number_input(
                    "Glucose Level (mg/dL)",
                    min_value=50, max_value=300, value=110, step=1,
                )
                chol_val = st.number_input(
                    "Total Cholesterol (mg/dL)",
                    min_value=100, max_value=400, value=195, step=1,
                )

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Predict Diabetes Risk",
                use_container_width=True,
                type="primary",
            )

    with col_guide:
        st.markdown('<p class="section-heading">PREDICTION GUIDE</p>', unsafe_allow_html=True)
        st.markdown("""
<div class="card" style="padding:18px 20px;">
  <p style="font-size:0.88rem; color:#1f2937; line-height:1.6; margin-bottom:12px;">
    Enter the six health parameters and click <strong>"Predict Diabetes Risk"</strong>.
  </p>
  <p style="font-size:0.84rem; color:#4b5563; line-height:1.55; margin-bottom:0;">
    The trained model will process the values and generate an estimated probability.
  </p>
</div>
""", unsafe_allow_html=True)

    # Prediction result
    if submitted:
        st.markdown("---")
        try:
            result   = predict_patient_risk(
                age=age_val, bmi=bmi_val, physical_activity=act_val,
                blood_pressure=bp_val, cholesterol=chol_val, glucose=glu_val,
                model_data=model_data,
            )
            is_pos   = result["predicted_class"] == 1
            risk_pct = result["risk_percentage"]
            base_pct = result["base_model_probability"]

            st.markdown('<p class="section-heading">PREDICTION RESULT</p>', unsafe_allow_html=True)
            res_col, gap_col = st.columns([2, 1], gap="large")
            with res_col:
                if is_pos:
                    st.markdown(f"""
<div class="res-pos">
  <div class="res-tag">Classification Outcome</div>
  <div class="res-class-pos">DIABETIC (High Risk)</div>
  <div class="res-tag">Estimated Risk Probability</div>
  <div class="res-prob">{risk_pct}%</div>
  <div class="res-note">
    The model indicates an <strong>elevated likelihood</strong> of diabetes
    based on the entered clinical metrics.
    Base classifier output: <strong>{base_pct}%</strong>.
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="res-neg">
  <div class="res-tag">Classification Outcome</div>
  <div class="res-class-neg">NON-DIABETIC (Low Risk)</div>
  <div class="res-tag">Estimated Risk Probability</div>
  <div class="res-prob">{risk_pct}%</div>
  <div class="res-note">
    The model indicates a <strong>lower likelihood</strong> of diabetes
    based on the entered clinical metrics.
    Base classifier output: <strong>{base_pct}%</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                st.progress(float(risk_pct / 100.0))
                st.caption(f"Calculated risk probability: {risk_pct}%")

            with gap_col:
                st.markdown("**Submitted Parameters**")
                inp = result["inputs_received"]
                for k, v in inp.items():
                    st.markdown(
                        f"<div class='ab-row'><span class='ab-key'>{k}</span>"
                        f"<span class='ab-val'>{v}</span></div>",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("""
<div class="disclaimer">
  <strong>Academic Disclaimer:</strong> This system is developed solely for educational and academic demonstration as part of a B.Tech minor project. It is not intended for diagnostic or clinical medical decision-making. Always consult certified healthcare professionals.
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE: DATA INSIGHTS
# ===========================================================================
elif page == "Data Insights":

    st.markdown('<p class="pg-title">Dataset & Data Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub">Exploratory analysis, class distribution, feature correlations, '
        'and preprocessing methodology.</p>',
        unsafe_allow_html=True,
    )

    # Stat boxes
    st.markdown("""
<div class="stat-row">
  <div class="stat-box"><div class="stat-val">284</div><div class="stat-lbl">Total Records</div></div>
  <div class="stat-box"><div class="stat-val">8</div><div class="stat-lbl">Features</div></div>
  <div class="stat-box"><div class="stat-val">177</div><div class="stat-lbl">Non-Diabetic &nbsp;(62.3%)</div></div>
  <div class="stat-box"><div class="stat-val">107</div><div class="stat-lbl">Diabetic &nbsp;(37.7%)</div></div>
</div>
""", unsafe_allow_html=True)

    # Preprocessing methodology
    st.markdown("#### Data Cleaning & Preprocessing")
    st.markdown("""
**Biological Zero Treatment:**  
A living person cannot have a Glucose reading of 0 or a BMI of 0. In the Pima dataset these
represent unrecorded measurements encoded as zero. The columns affected are Glucose (2 zeros),
BloodPressure (12 zeros), BMI (5 zeros), SkinThickness (67 zeros), and Insulin (218 zeros).
All zeros in these columns were replaced with `NaN` before any further processing.

**Zero Data Leakage:**  
The `SimpleImputer` (median strategy) and `StandardScaler` were fitted on the **training
partition only** (80% of data) and then applied to the test set. This ensures the model never
learns anything from test data during training.

**Feature Selection:**  
Four core clinically validated features were selected for the ML pipeline: *Age, BMI,
BloodPressure, Glucose*. SkinThickness and Insulin were excluded because more than 70% of
their values were missing.
""")

    st.markdown("---")
    st.markdown("#### Exploratory Visualisations")

    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Correlation Matrix", "Feature Distributions"])

    with tab1:
        img = os.path.join(viz_dir, "class_distribution.png")
        if os.path.exists(img):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(img, caption="Target Class Distribution", use_container_width=True)
            with c2:
                st.markdown("**Non-Diabetic (0):** 177 records — 62.3%")
                st.markdown("**Diabetic (1):** 107 records — 37.7%")
                st.markdown(
                    "There is a moderate class imbalance. The Random Forest was trained with "
                    "`class_weight='balanced'` to prevent under-predicting the minority (diabetic) class."
                )
        else:
            st.info("Run `python src/train_model.py` to generate charts.")

    with tab2:
        img = os.path.join(viz_dir, "correlation_heatmap.png")
        if os.path.exists(img):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(img, caption="Pearson Correlation Matrix", use_container_width=True)
            with c2:
                st.markdown("**Key correlations with Outcome:**")
                st.markdown("- Glucose → Outcome: **0.49** (strongest)")
                st.markdown("- BMI → Outcome: **0.30**")
                st.markdown("- Age → Outcome: **0.24**")
                st.markdown("- BloodPressure → Outcome: **0.16**")
        else:
            st.info("Run `python src/train_model.py` to generate charts.")

    with tab3:
        img = os.path.join(viz_dir, "key_features_distribution.png")
        if os.path.exists(img):
            st.image(img, caption="KDE Distributions — Glucose, BMI, Age (by Outcome)",
                     use_container_width=True)
            st.markdown(
                "Diabetic patients (red) show a notably higher glucose distribution peak, "
                "a heavier BMI tail, and are slightly older on average. "
                "These separations confirm meaningful predictive signal in all three features."
            )
        else:
            st.info("Run `python src/train_model.py` to generate charts.")


# ===========================================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":

    st.markdown('<p class="pg-title">Model Performance & Evaluation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub">Comparative evaluation on the unseen 20% hold-out test set '
        '(57 patients, stratified split).</p>',
        unsafe_allow_html=True,
    )

    if model_data and "all_metrics" in model_data:
        df_m = model_data["all_metrics"].copy()

        # Format metrics as percentages where applicable
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            if col in df_m.columns:
                df_m[col] = df_m[col].apply(lambda x: f"{float(x)*100:.2f}%")

        st.markdown("#### Performance Comparison Table")
        st.dataframe(df_m, use_container_width=True, hide_index=True)

        st.markdown("#### Model Selection Rationale")
        st.markdown("""
**Random Forest Classifier** was selected as the final deployed model.

Compared to Logistic Regression (a linear baseline) and a single Decision Tree, the Random Forest
combines 100 decorrelated decision trees through *bagging*, reducing variance and improving
generalisation on unseen data. It achieved the highest **test accuracy (71.93%)**, a balanced
**recall of 71.43%** (critical for not missing diabetic patients), and a strong **ROC-AUC of 0.7817**.

In clinical risk screening, *recall is prioritised* over precision — it is more important to
correctly flag a diabetic patient (reduce false negatives) than to avoid false alarms.
The Random Forest with `class_weight='balanced'` handles this trade-off effectively.
""")

        st.markdown("---")
        st.markdown("#### Diagnostic Plots")

        col_cm, col_roc = st.columns(2, gap="medium")
        with col_cm:
            img = os.path.join(viz_dir, "confusion_matrices.png")
            if os.path.exists(img):
                st.image(img, caption="Confusion Matrices — all three models", use_container_width=True)
                st.caption("TP = correctly predicted diabetic | FN = missed diabetics (high clinical cost)")
        with col_roc:
            img = os.path.join(viz_dir, "roc_curves.png")
            if os.path.exists(img):
                st.image(img, caption="ROC Curves — area under curve comparison", use_container_width=True)
                st.caption("A curve closer to the top-left corner is better. AUC of 0.78 indicates good discrimination.")

        st.markdown("---")
        st.markdown("#### Feature Importance (Random Forest — Gini Impurity)")
        c1, c2 = st.columns([2, 1])
        with c1:
            img = os.path.join(viz_dir, "feature_importance.png")
            if os.path.exists(img):
                st.image(img, caption="Mean Decrease in Impurity per Feature", use_container_width=True)
        with c2:
            st.markdown("""
1. **Glucose** — most influential. High fasting glucose is the primary clinical marker for diabetes.
2. **BMI** — second most important. Obesity drives insulin resistance.
3. **Age** — moderate importance. Risk increases with age due to metabolic changes.
4. **Blood Pressure** — least among the four, but contributes to metabolic syndrome risk.
""")

    else:
        st.warning("Model data not found. Please run `python src/train_model.py` first.")


# ===========================================================================
# PAGE: ABOUT
# ===========================================================================
elif page == "About":

    st.markdown('<p class="pg-title">About the Project</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub">Project information, methodology, and documentation.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2], gap="medium")

    with c1:
        st.markdown("#### Project Details")
        details = [
            ("Title",          "AI-Based Diabetes Prediction System"),
            ("Student",        "Nafize Ali"),
            ("Branch",         "Artificial Intelligence & Data Science"),
            ("Type",           "B.Tech Minor Project"),
            ("Dataset",        "Pima Indians Diabetes Dataset"),
            ("Algorithm",      "Random Forest Classifier"),
            ("ML Features",    "Age, BMI, BloodPressure, Glucose"),
            ("UI Inputs",      "Age, BMI, Activity, BP, Cholesterol, Glucose"),
            ("Train / Test",   "80% / 20% stratified split"),
            ("Best Accuracy",  "71.93%"),
            ("Best ROC-AUC",   "0.7817"),
        ]
        for key, val in details:
            st.markdown(
                f"<div class='ab-row'><span class='ab-key'>{key}</span>"
                f"<span class='ab-val'>{val}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Methodology Summary")
        st.markdown("""
1. Loaded the Pima Indians Diabetes CSV dataset (284 records, 8 features).
2. Detected and replaced physiological zeros with `NaN` (biological zero treatment).
3. Applied a stratified 80/20 train-test split before any imputation — no data leakage.
4. Built a Scikit-Learn `Pipeline`: `SimpleImputer(median)` → `StandardScaler` → Classifier.
5. Trained and compared Logistic Regression, Decision Tree, and Random Forest.
6. Evaluated each on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
7. Selected Random Forest as the best model and serialised it with `joblib`.
8. Deployed the model in this interactive Streamlit web application.
""")

    with c2:
        st.markdown("#### Libraries Used")
        libs = [
            ("Python 3.12",  "Core language"),
            ("Pandas",       "Data manipulation"),
            ("NumPy",        "Numerical operations"),
            ("Scikit-Learn", "ML pipeline & models"),
            ("Matplotlib",   "Static charts"),
            ("Seaborn",      "Statistical visualisations"),
            ("Joblib",       "Model serialisation"),
            ("Streamlit",    "Web application"),
        ]
        for lib, desc in libs:
            st.markdown(
                f"<div class='ab-row'><span class='ab-key'>{lib}</span>"
                f"<span class='ab-val' style='color:#64748b;font-weight:400;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Project Files")
        files = [
            ("app.py",                        "Streamlit application"),
            ("src/data_preprocessing.py",     "Data loading & cleaning"),
            ("src/train_model.py",            "Model training & evaluation"),
            ("src/prediction.py",             "Inference pipeline"),
            ("notebooks/diabetes_analysis.ipynb", "EDA notebook"),
            ("README.md",                     "Project documentation"),
            ("VIVA_QUESTIONS.md",             "Viva preparation"),
        ]
        for fname, fdesc in files:
            st.markdown(
                f"<div class='ab-row'><span class='ab-key' style='font-family:monospace;font-size:0.8rem;'>{fname}</span>"
                f"<span class='ab-val' style='color:#64748b;font-weight:400;'>{fdesc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("""
<div class="disclaimer" style="margin-top:18px;">
  <strong>Disclaimer:</strong> This application is developed solely for academic and educational
  purposes as part of a B.Tech Minor Project. It is not a medical diagnostic tool and must not
  replace professional clinical advice.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Handle "Start Prediction" button navigation from Home page
# ---------------------------------------------------------------------------
if "_nav" in st.session_state and st.session_state["_nav"] == "Prediction":
    del st.session_state["_nav"]
