import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/gui_model.pkl")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to leave the company."
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0.0,
    value=12.0
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

if st.button("Predict Churn"):

    data = np.array([[tenure, monthly, total]])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    probability_percent = round(
        probability * 100,
        2
    )

    if prediction == 1:

        st.error(
            f"High Risk Customer ({probability_percent}%)"
        )

    else:

        st.success(
            f"Low Risk Customer ({probability_percent}%)"
        )