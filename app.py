import streamlit as st
import pandas as pd
import joblib
import os 

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("📊 Churn Prediction")

st.sidebar.markdown(
    """
    ### Navigation

    Use the main panel to:

    - Enter customer information
    - Predict churn probability
    - View customer risk
    - Understand key churn factors
    - Review model performance
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Model: Tuned Random Forest\n\n"
    "ROC-AUC: 84.21%"
)


st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a customer is likely to churn "
    "using a machine learning model."
)


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(
        "models/churn_prediction_pipeline.pkl"
    )


model = load_model()

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/telco_churn.csv"
    )

df = load_data()

# -----------------------------
# Customer Information
# -----------------------------

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# -----------------------------
# Service Information
# -----------------------------

st.subheader("📞 Service Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


# -----------------------------
# Security & Support
# -----------------------------

st.subheader("🔐 Security & Support")

col1, col2 = st.columns(2)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


# -----------------------------
# Streaming Services
# -----------------------------

st.subheader("📺 Streaming Services")

col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# -----------------------------
# Billing Information
# -----------------------------

st.subheader("💳 Billing Information")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=150.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )

st.divider()

st.subheader("🔮 Churn Prediction")

if st.button("Predict Churn", type="primary"):

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]

    probability_percentage = probability * 100

    st.subheader("👤 Customer Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric(
            "Tenure",
            f"{tenure} months"
        )
    
    with summary_col2:
        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )
    
    with summary_col3:
        st.metric(
            "Contract",
            contract
        )

    # Determine risk level
    if probability < 0.30:
        risk_level = "Low Risk"
        risk_message = "Customer has a relatively low predicted churn risk."
    elif probability < 0.60:
        risk_level = "Medium Risk"
        risk_message = "Customer has a moderate predicted churn risk."
    else:
        risk_level = "High Risk"
        risk_message = "Customer has a high predicted churn risk."

    # Display result
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is unlikely to churn")

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )

    st.progress(float(probability))

    st.write(f"**Risk Level:** {risk_level}")

    st.info(risk_message)
    
    st.markdown("### 🔎 Important Churn Indicators")

indicators = []

if contract == "Month-to-month":
    indicators.append("Month-to-month contract")

if tenure < 12:
    indicators.append("Short customer tenure")

if monthly_charges > 70:
    indicators.append("Higher monthly charges")

if internet_service == "Fiber optic":
    indicators.append("Fiber optic internet service")

if online_security == "No":
    indicators.append("No online security")

if tech_support == "No":
    indicators.append("No technical support")

if payment_method == "Electronic check":
    indicators.append("Electronic check payment")

if indicators:
    for indicator in indicators:
        st.write(f"• {indicator}")
else:
    st.write("No major high-risk indicators identified.")

    # Business recommendation
    if probability >= 0.60:
        st.warning(
            "💡 Recommendation: Consider prioritizing this customer "
            "for a retention campaign or personalized offer."
        )
    elif probability >= 0.30:
        st.info(
            "💡 Recommendation: Monitor this customer and consider "
            "a proactive engagement or satisfaction check."
        )
    else:
        st.success(
            "💡 Recommendation: No immediate retention action is "
            "required based on the current prediction."
        )


st.divider()

st.subheader("📊 Key Business Insights")

st.caption(
    "These insights are based on observed patterns in the "
    "telecom customer dataset and do not imply causation."
)


col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Contract Risk")
    st.write(
        "Month-to-month customers show substantially higher "
        "churn than customers on one-year or two-year contracts."
    )

    st.markdown("### 🌐 Internet Service")
    st.write(
        "Fiber optic customers have a higher observed churn rate "
        "than DSL and customers without internet service."
    )

    st.markdown("### 💳 Payment Method")
    st.write(
        "Customers using electronic check have a noticeably "
        "higher observed churn rate."
    )

with col2:
    st.markdown("### 🛠️ Technical Support")
    st.write(
        "Customers without technical support show higher "
        "churn rates than customers with technical support."
    )

    st.markdown("### 🔐 Online Security")
    st.write(
        "Customers without online security have considerably "
        "higher observed churn."
    )

    st.markdown("### ⏳ Tenure")
    st.write(
        "Customers with shorter tenure are generally more "
        "likely to churn than long-term customers."
    )

st.subheader("📊 Churn Analysis")
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(
    contract_churn
)

st.markdown("### 🌐 Churn by Internet Service")

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(
    internet_churn
)

st.markdown("### 💳 Churn by Payment Method")

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(
    payment_churn
)

st.markdown("### 🛠️ Churn by Tech Support")

support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(
    support_churn
)

st.markdown("### 🔐 Churn by Online Security")

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(
    security_churn
)



st.divider()

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "75.94%")

with col2:
    st.metric("Precision", "53.23%")

with col3:
    st.metric("Recall", "77.01%")

with col4:
    st.metric("ROC-AUC", "84.21%")

st.subheader("🎯 Risk Level Guide")

risk_data = pd.DataFrame({
    "Risk Level": [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ],
    "Churn Probability": [
        "0% – 29%",
        "30% – 59%",
        "60% – 100%"
    ],
    "Recommended Action": [
        "No immediate action",
        "Monitor and engage",
        "Prioritize retention"
    ]
})

st.dataframe(
    risk_data,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    """
    This Customer Churn Prediction System uses machine learning
    to identify customers who may be at risk of leaving a
    telecommunications service.

    The system uses customer demographics, service usage,
    contract information, and billing information to generate
    churn predictions and estimated churn probabilities.

    The final model is a tuned Random Forest integrated with
    preprocessing through a Scikit-learn Pipeline.
    """
)

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Python • Scikit-learn • Random Forest • Streamlit"
)
