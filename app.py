"""Streamlit interface for the educational diabetes prediction project."""
from pathlib import Path
import subprocess
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
METRICS_PATH = ROOT / "model_metrics.txt"

st.set_page_config(page_title="Medical Disease Prediction", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#246b8e); color: white;}
.small-note {color: #52606d; font-size: .95rem;}
div[data-testid="stMetric"] {background: white; border: 1px solid #e4eaf0; padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🩺 Medical Disease Prediction</h1><p>Machine Learning ISE Activity · Diabetes risk classification demonstration</p></div>', unsafe_allow_html=True)
st.write("")
st.info("Educational demonstration only. This tool does not diagnose diabetes and must not be used for medical decisions.")

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        result = subprocess.run([sys.executable, str(ROOT / "train_model.py")], cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Model training failed.")
    return joblib.load(MODEL_PATH)

try:
    bundle = load_bundle()
    model = bundle["model"]
except Exception as exc:
    st.error("The model could not be prepared. Run `python train_model.py` in the project folder and check your internet connection.")
    st.code(str(exc))
    st.stop()

st.subheader("Enter the sample health parameters")
st.caption("Enter values from a dataset or exercise scenario. Do not use this page to make health decisions.")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood pressure (mm Hg)", min_value=1.0, max_value=200.0, value=70.0, step=1.0)
    with col2:
        skin_thickness = st.number_input("Skin thickness (mm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        insulin = st.number_input("Insulin (µU/mL)", min_value=1.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=1.0, max_value=80.0, value=28.0, step=0.1)
    with col3:
        pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=21, max_value=100, value=30, step=1)
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age
    }], columns=bundle["features"])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    positive_index = list(model.named_steps["classifier"].classes_).index(1)
    probability = float(probabilities[positive_index])
    st.divider()
    st.subheader("Model output")
    if prediction == 1:
        st.warning("The model classified this example in the positive class.")
    else:
        st.success("The model classified this example in the negative class.")
    st.metric("Model-estimated positive-class probability", f"{probability:.1%}")
    st.caption("This probability is not a calibrated personal medical risk estimate. Model output can be wrong; consult a qualified healthcare professional for health concerns.")

with st.expander("About the project"):
    st.write("The app uses a Random Forest classifier trained on a public benchmark dataset. Missing-value handling and model evaluation are performed in train_model.py.")
    st.write("The dataset describes a specific population and is not representative of all people. Performance on the held-out test set does not establish clinical validity.")
with st.expander("Model evaluation (generated locally)"):
    if METRICS_PATH.exists():
        st.code(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        st.write("Run `python train_model.py` to generate the evaluation report.")
st.markdown("---")
st.markdown('<p class="small-note">Machine Learning ISE Activity · Python · Scikit-learn · Streamlit</p>', unsafe_allow_html=True)
