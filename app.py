import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
MODEL_PATH = "./sklearn_lr_credit_risk_pipeline.joblib" # Path to the saved scikit-learn pipeline
try:
    pipeline = joblib.load(MODEL_PATH)
    st.sidebar.success("Credit Risk Model Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- App Title and Introduction ---
st.title("🏦 Real-Time Credit Risk Scoring Simulation")
st.markdown("""
This application simulates a real-world scenario where a bank uses a machine learning model
to assess the credit risk of new loan applications in real-time.

**Behind the Scenes:**
*   A sophisticated model (represented here by a Scikit-learn Logistic Regression pipeline) was **originally trained offline on a large historical dataset (potentially millions of loans) using distributed computing frameworks like Apache Spark** to handle the scale.
*   This offline process involves complex feature engineering and hyperparameter tuning.
*   The pipeline saved from that process includes both data preprocessing steps and the trained model.
*   This app loads the *equivalent* pre-trained pipeline to score incoming applications instantly.

**Enter the details of a new loan application below to get a real-time risk assessment.**
""")
st.info("ℹ️ Disclaimer: This application uses simulated data and a representative model for demonstration purposes only. Predictions are illustrative.")

# --- Real-Time Application Input ---
st.header("📝 New Loan Application Input")

# Use columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1, help="Enter the applicant's credit score (e.g., 300-850).")
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f", help="Applicant's Debt-to-Income ratio (e.g., 0.35 for 35%).")
    collateral_flag = st.selectbox("Is the loan secured by collateral?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Select 1 for Yes, 0 for No.")

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=100, help="Requested loan amount.")
    employment_years = st.number_input("Employment Length (Years)", min_value=0, max_value=50, value=5, step=1, help="Number of years the applicant has been employed.")
    monitoring_intensity = st.selectbox("Bank's Monitoring Intensity Plan", options=['Low', 'Medium', 'High'], index=1, help="Planned monitoring level for this loan if approved.")

with col3:
    policy_score = st.slider("Policy Adherence Score", min_value=1, max_value=5, value=3, step=1, help="Internal score representing adherence to lending policy (1=Low, 5=High).")
    analysis_score = st.slider("Analysis Depth Score", min_value=1, max_value=5, value=3, step=1, help="Internal score for depth of underwriting analysis (1=Shallow, 5=Deep).")


# --- Scoring Button and Logic ---
st.header("📊 Risk Assessment")
submit_button = st.button("Assess Credit Risk", type="primary")

if submit_button:
    # 1. Prepare Input Data for the model pipeline
    input_data = pd.DataFrame({
        # Ensure column order and names match EXACTLY what the pipeline was trained on
        'Credit_Score': [credit_score],
        'DTI_Ratio': [dti_ratio],
        'Loan_Amount': [loan_amount],
        'Employment_Length_Years': [employment_years],
        'Policy_Adherence_Score': [policy_score],
        'Monitoring_Intensity': [monitoring_intensity],
        'Analysis_Depth_Score': [analysis_score],
        'Collateral_Secured_Flag': [collateral_flag]
    })

    st.write("---")
    st.subheader("Application Data Received:")
    st.dataframe(input_data)

    # 2. Make Prediction
    try:
        with st.spinner("🧠 Analyzing risk..."):
            start_pred_time = time.time()
            prediction_proba = pipeline.predict_proba(input_data)[:, 1] # Probability of Default (class 1)
            prediction = pipeline.predict(input_data)[0] # 0 or 1 prediction
            end_pred_time = time.time()

        prob_default = prediction_proba[0]
        processing_time = end_pred_time - start_pred_time

        # 3. Display Results
        st.subheader("✅ Assessment Complete!")
        st.write(f"Processing Time: {processing_time:.4f} seconds")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(label="Predicted Probability of Default", value=f"{prob_default:.2%}")
            if prob_default < 0.10: # Example threshold
                st.success("Risk Level: Low")
                prediction_text = "Likely Approved"
                pred_icon = "✅"
            elif prob_default < 0.35:
                st.warning("Risk Level: Medium")
                prediction_text = "Further Review Recommended"
                pred_icon = "⚠️"
            else:
                st.error("Risk Level: High")
                prediction_text = "Likely Rejected"
                pred_icon = "❌"

        with col_res2:
            st.metric(label="Decision Indication", value=prediction_text)
            # Add progress bar visualization
            st.progress(prob_default)
            st.caption("Progress bar indicates probability of default.")

        # Optional: Explain features (requires model introspection - harder with pipelines)
        # st.subheader("Key Factors (Illustrative)")
        # st.write("*(In a real system, SHAP values or LIME could provide feature importance for this specific prediction)*")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Conceptual Monitoring Section ---
st.divider()
st.header("🔬 Model Monitoring (Conceptual)")
st.markdown("""
In a production environment, the model's performance and the incoming data would be continuously monitored:

*   **Data Drift:** Are the characteristics of new applicants (e.g., average credit score, loan amounts) changing significantly compared to the training data? This could invalidate the model. *Techniques: Statistical tests (KS test), distribution comparisons.*
*   **Concept Drift (Performance):** Is the model's predictive accuracy degrading over time? Does the relationship between features and default change? *Requires joining predictions with actual loan outcomes (after months/years) and tracking metrics like AUC, Gini, F1-score.*
*   **Operational Health:** Is the scoring service fast and reliable? Are there errors?

Monitoring triggers alerts and potentially model retraining cycles to ensure the risk assessment remains accurate and reliable.
""")
# You could add static images of example monitoring dashboards here
# st.image("path/to/drift_dashboard.png")