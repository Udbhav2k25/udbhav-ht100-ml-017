# app.py
# Streamlit inference app for The Cardio Predictor
# - Loads stacked ensemble models once (cached)
# - Cool pink/black gradient UI
# - Collects raw data inputs from the user
# - CORRECTLY uses the predict_single function from infer.py for all preprocessing and inference

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Import the prediction function from your infer.py script
from predict import predict_single

# ======================
# 1. PAGE CONFIG + CSS
# ======================
st.set_page_config(
    page_title="The Cardio Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Use more of the page horizontally */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1400px;
    }

    /* Pinkish–black gradient background */
    .stApp {
        background: radial-gradient(circle at top left, #ff66cc 0%, #1b0024 35%, #000000 100%);
        color: #f5f5f5;
    }

    /* Card-style containers */
    .card {
        background: rgba(10, 10, 25, 0.92);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 105, 180, 0.22);
        backdrop-filter: blur(6px);
        margin-bottom: 2rem; /* Add space between cards */
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff99dd, #ffea7a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.97rem;
        color: #f5e9ff;
        opacity: 0.9;
    }

    /* Fancy Predict button */
    .stButton>button {
        background: linear-gradient(90deg, #ff5fa2, #ff9966);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.65rem 2.4rem;
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 12px 30px rgba(255, 105, 180, 0.4);
        transition: all 0.18s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 18px 45px rgba(255, 105, 180, 0.65);
        filter: brightness(1.1);
        cursor: pointer;
    }

    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.85);
    }

    .risk-score {
        font-size: 2.0rem;
        font-weight: 800;
    }

    .risk-label {
        font-size: 1.05rem;
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================
# 2. LOAD MODELS (CACHED)
# ======================
@st.cache_resource
def load_models():
    """Loads the stacked ensemble models and caches them."""
    model_dir = "artifacts_s1_boosted"
    lgb_path = os.path.join(model_dir, "lgb_base.pkl")
    xgb_path = os.path.join(model_dir, "xgb_base.pkl")
    meta_path = os.path.join(model_dir, "meta_model.pkl")
    features_path = os.path.join(model_dir, "features.pkl")

    if not all(os.path.exists(p) for p in [lgb_path, xgb_path, meta_path, features_path]):
        st.error(f"Model files not found in '{model_dir}'. Please ensure all .pkl files exist.")
        return None

    try:
        lgb_model = joblib.load(lgb_path)
        xgb_model = joblib.load(xgb_path)
        meta_model = joblib.load(meta_path)
        feature_names = joblib.load(features_path)
        return lgb_model, xgb_model, meta_model, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Make loaded models available globally in the script for the predict function to use
lgb_model, xgb_model, meta_model, feature_names = load_models()
if lgb_model is None:
    st.stop() # Stop the app if models can't be loaded

# ======================
# 3. UI: PATIENT INPUTS
# ======================
def get_user_inputs():
    """
    Creates the UI for user input.
    These fields directly correspond to the raw data columns expected by predict_single.
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Patient Details")

    # Create columns for a cleaner layout
    c1, c2, c3 = st.columns(3)

    with c1:
        # Corresponds to 'age' column (input in years)
        age_years = st.slider("Age (years)", 18, 100, 50)
        
        # Corresponds to 'ap_hi' column
        ap_hi = st.slider("Systolic Blood Pressure (ap_hi)", 80, 250, 120)
        
        # Corresponds to 'ap_lo' column
        ap_lo = st.slider("Diastolic Blood Pressure (ap_lo)", 40, 150, 80)

    with c2:
        # Corresponds to 'height' column
        height = st.slider("Height (cm)", 100, 220, 170)
        
        # Corresponds to 'weight' column
        weight = st.slider("Weight (kg)", 30, 200, 80)
        
        # Corresponds to 'cholesterol' column (1=normal, 2=above normal, 3=well above normal)
        cholesterol = st.selectbox(
            "Cholesterol", 
            [1, 2, 3], 
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
        )

    with c3:
        # Corresponds to 'gender' column (1=female, 2=male)
        gender = st.selectbox(
            "Gender", 
            [1, 2], 
            format_func=lambda x: {1: "Female", 2: "Male"}[x]
        )
        
        # Corresponds to 'gluc' column (1=normal, 2=above normal, 3=well above normal)
        gluc = st.selectbox(
            "Glucose", 
            [1, 2, 3], 
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
        )

        # Corresponds to 'smoke', 'alco', 'active' columns (0=no, 1=yes)
        smoke = st.selectbox("Smoker", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
        alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
        active = st.selectbox("Physically Active", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

    st.markdown('</div>', unsafe_allow_html=True)

    # Return a dictionary that matches the predict_single function's expected input
    return {
        "age": int(age_years),  # Pass age directly in years
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
    }


# ======================
# 4. MAIN APP
# ======================

def main():
    # Header card
    st.markdown(
        """
        <div class="card">
          <div class="main-title">The Cardio Predictor</div>
          <div class="subtitle">
            Enter patient data to get an instant prediction of cardiovascular risk.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Get user inputs from the UI
    user_data = get_user_inputs()

    st.write("")
    c_btn, c_space = st.columns([1, 3])

    with c_btn:
        clicked = st.button("Predict Risk")

    if clicked:
        with st.spinner("Analyzing data..."):
            # Call the prediction function from infer.py
            result = predict_single(user_data)

        # Check for errors returned from the prediction function
        if "error" in result:
            st.error(result["error"])
        else:
            risk_pct = result["heart_attack_risk_percent"]
            prediction_label = result["prediction"]

            # Determine risk level and color for display
            if risk_pct < 20:
                level = "Low"
                color = "#4CAF50"
                desc = "Currently looks low, but maintaining healthy habits is still important."
            elif risk_pct < 50:
                level = "Moderate"
                color = "#FFC107"
                desc = "There may be some risk factors present. Consider monitoring and lifestyle changes."
            else:
                level = "High"
                color = "#FF5252"
                desc = "Higher risk indicated. Please consult a healthcare professional."

            # Display the final result
            st.markdown(
                f"""
                <div class="card" style="margin-top: 1.2rem;">
                  <div class="risk-label">Estimated probability of heart risk:</div>
                  <div class="risk-score" style="color: {color};">
                    {risk_pct:.2f}% &nbsp; ({level} risk)
                  </div>
                  <div style="margin-top: 0.4rem; font-size: 0.95rem; opacity: 0.9;">
                    {desc}
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()