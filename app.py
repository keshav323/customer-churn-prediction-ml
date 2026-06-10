import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/gui_model.pkl")

# Page settings
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Customer Churn Analytics Dashboard")
st.markdown(
    "Predict whether a customer is likely to leave the company using Machine Learning."
)

st.divider()

# Input Section
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0.0,
        value=12.0
    )

with col2:
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col3:
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )

# Prediction Button
if st.button("Predict Churn"):

    # Create DataFrame with correct feature names
    data = pd.DataFrame(
        [[tenure, monthly, total]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    )

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.divider()

    # Metrics Section
    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="Churn Probability",
            value=f"{probability * 100:.2f}%"
        )

    with metric2:
        st.metric(
            label="Model Accuracy",
            value="77%"
        )

    st.divider()

    # Result Section
    if prediction == 1:

        st.error(
            f"⚠ HIGH RISK CUSTOMER ({probability * 100:.2f}%)"
        )

        st.warning(
            "Recommendation: Offer discounts, retention benefits, or personalized support."
        )

    else:

        st.success(
            f"✅ LOW RISK CUSTOMER ({probability * 100:.2f}%)"
        )

        st.info(
            "Recommendation: Customer is likely to stay with the company."
        )

# Footer
st.divider()

st.caption(
    "Built using Python, Scikit-Learn, Streamlit, Pandas, and Random Forest Classification."
)
