import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/gui_model.pkl")

# Page Config
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("📋 Project Information")

st.sidebar.markdown("""
### Model Details

- Algorithm: Random Forest
- Accuracy: 77%
- Features Used:
  - Tenure
  - Monthly Charges
  - Total Charges

### Author
Keshav Saini
""")

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Logo
try:
    st.image("customer_churn_logo.png", width=120)
except:
    pass

# Title
st.title("📊 Customer Churn Analytics Dashboard")

st.success(
    "🚀 Live Machine Learning Dashboard Deployed on Streamlit Cloud"
)

st.markdown(
    "Predict whether a customer is likely to leave the company using Machine Learning."
)

# Top Metrics
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Records", "7043")

with m2:
    st.metric("Features", "20")

with m3:
    st.metric("Algorithm", "Random Forest")

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

# Prediction Section
if st.button("🚀 Predict Churn"):

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

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with k2:
        st.metric(
            "Model Accuracy",
            "77%"
        )

    with k3:
        st.metric(
            "Risk Level",
            "HIGH" if prediction == 1 else "LOW"
        )

    st.divider()

    st.subheader("📈 Risk Visualization")

    st.progress(float(probability))

    if prediction == 1:

        st.error(
            f"⚠ High Risk Customer ({probability * 100:.2f}%)"
        )

        st.warning(
            "Recommendation: Offer discounts, loyalty rewards, or personalized support."
        )

    else:

        st.success(
            f"✅ Low Risk Customer ({probability * 100:.2f}%)"
        )

        st.info(
            "Recommendation: Customer is likely to remain with the company."
        )

    # Download Report
    st.divider()

    report = pd.DataFrame(
        {
            "Tenure": [tenure],
            "Monthly Charges": [monthly],
            "Total Charges": [total],
            "Prediction": [
                "High Risk Customer"
                if prediction == 1
                else "Low Risk Customer"
            ],
            "Churn Probability (%)": [
                round(probability * 100, 2)
            ]
        }
    )

    csv = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="customer_churn_report.csv",
        mime="text/csv"
    )

# Analytics Section
st.divider()

st.subheader("📊 Dataset Insights")

tab1, tab2, tab3 = st.tabs(
    [
        "Churn Distribution",
        "Feature Importance",
        "Confusion Matrix"
    ]
)

with tab1:
    st.image(
        "screenshots/churn_distribution.png",
        use_container_width=True
    )

with tab2:
    st.image(
        "screenshots/feature_importance.png",
        use_container_width=True
    )

with tab3:
    st.image(
        "screenshots/confusion_matrix.png",
        use_container_width=True
    )

st.divider()

st.caption(
    "Built using Python, Streamlit, Scikit-Learn, Pandas, Matplotlib, Seaborn, and Random Forest Classification."
)