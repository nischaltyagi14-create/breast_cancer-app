import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page title and styling
st.set_page_config(page_title="Breast Cancer 10-Yr Mortality Predictor", layout="centered")

st.title("🩺 10-Year Mortality Risk Predictor")
st.write("""
This clinical tool utilizes a tuned Logistic Regression model to assess the 10-year mortality risk for patients with breast cancer.
""")

# Load model and features
@st.cache_resource
def load_model_assets():
    model = joblib.load('breast_cancer_model.joblib')
    features = joblib.load('feature_names.joblib')
    return model, features

try:
    model, feature_names = load_model_assets()
except Exception as e:
    st.error(f"Error loading model. Make sure 'breast_cancer_model.joblib' and 'feature_names.joblib' are in the same directory. Detail: {e}")
    st.stop()

# Collect clinical user inputs
st.header("Patient Clinical Profile")

# Create form container for structured inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age at Diagnosis", min_value=18.0, max_value=100.0, value=60.0, step=1.0)
        tumor_size = st.number_input("Tumor Size (mm)", min_value=1.0, max_value=200.0, value=25.0, step=1.0)
        tumor_stage = st.selectbox("Tumor Stage", [0.0, 1.0, 2.0, 3.0, 4.0], index=2)

    with col2:
        nodes = st.number_input("Positive Lymph Nodes Examined", min_value=0, max_value=50, value=0, step=1)
        npi = st.number_input("Nottingham Prognostic Index (NPI)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
        mastectomy = st.selectbox("Underwent Mastectomy Surgery?", ["Yes", "No"])

    # Risk threshold selection for clinical customization
    threshold = st.slider("Decision Risk Threshold (Lower threshold emphasizes higher recall/sensitivity)",
                          min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    submit_btn = st.form_submit_button()

if submit_btn:
    # Initialize full feature space with zeros
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Assign numerical fields directly
    if 'Age at Diagnosis' in feature_names: input_data['Age at Diagnosis'] = age
    if 'Tumor Size' in feature_names: input_data['Tumor Size'] = tumor_size
    if 'Tumor Stage' in feature_names: input_data['Tumor Stage'] = tumor_stage
    if 'Lymph nodes examined positive' in feature_names: input_data['Lymph nodes examined positive'] = nodes
    if 'Nottingham prognostic index' in feature_names: input_data['Nottingham prognostic index'] = npi

    # Encode categorical surgery choice
    if mastectomy == "Yes" and 'Type of Breast Surgery_Mastectomy' in feature_names:
        input_data['Type of Breast Surgery_Mastectomy'] = 1.0
    elif mastectomy == "No" and 'Type of Breast Surgery_Breast Conserving' in feature_names:
        input_data['Type of Breast Surgery_Breast Conserving'] = 1.0

    # Generate prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    risk_probability = probabilities[1]

    st.subheader("Prediction Analysis")
    st.write(f"**Estimated Patient 10-Year Mortality Risk:** `{risk_probability * 100:.2f}%`")

    if risk_probability >= threshold:
        st.error(f"⚠️ **Classified Status:** High Risk (At or above customized risk threshold of {threshold * 100:.0f}%)")
    else:
        st.success(f"✅ **Classified Status:** Low Risk (Below risk threshold of {threshold * 100:.0f}%)")
