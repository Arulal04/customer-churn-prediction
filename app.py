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

    if probability >= 0.60:
        st.caption(
            "The model estimates a relatively high likelihood of churn "
            "for this customer based on the information provided."
        )

    elif probability >= 0.30:
        st.caption(
            "The model estimates a moderate likelihood of churn "
            "for this customer based on the information provided."
        )

    else:
        st.caption(
            "The model estimates a relatively low likelihood of churn "
            "for this customer based on the information provided."
        )

    st.progress(float(probability))

    if probability >= 0.60:
        st.error(
            "🔴 High probability range — this customer should be prioritized "
            "for retention attention."
        )

    elif probability >= 0.30:
        st.warning(
            "🟡 Moderate probability range — this customer should be monitored."
        )

    else:
        st.success(
            "🟢 Low probability range — no immediate retention action is indicated."
        )

    st.write(f"**Risk Level:** {risk_level}")

    st.info(risk_message)
    
    if probability >= 0.60:
        st.markdown("### 🔎 Important Churn Indicators")
    elif probability >= 0.30:
        st.markdown("### 🔎 Factors to Monitor")
    else:
        st.markdown("### 🔎 Customer Risk Assessment")

    st.caption(
        "ℹ️ These indicators are based on observed patterns in the "
        "training dataset and are not causal explanations of churn."
    )

    risk_factors = []
    positive_factors = []

    # Risk factors
    if contract == "Month-to-month":
        risk_factors.append(
            "Month-to-month contract"
        )

    if tenure < 12:
        risk_factors.append(
            "Short customer tenure"
        )

    if monthly_charges > 70:
        risk_factors.append(
            "Higher monthly charges"
        )

    if internet_service == "Fiber optic":
        risk_factors.append(
            "Fiber optic internet service"
        )

    if online_security == "No":
        risk_factors.append(
            "No online security"
        )

    if tech_support == "No":
        risk_factors.append(
            "No technical support"
        )

    if payment_method == "Electronic check":
        risk_factors.append(
            "Electronic check payment"
        )

    # Positive factors
    if contract == "Two year":
        positive_factors.append(
            "Two-year contract"
        )

    if contract == "One year":
        positive_factors.append(
            "One-year contract"
        )

    if online_security == "Yes":
        positive_factors.append(
            "Online security enabled"
        )

    if tech_support == "Yes":
        positive_factors.append(
            "Technical support enabled"
        )

    if tenure >= 24:
        positive_factors.append(
            "Long customer tenure"
        )

    # Display risk factors
    if risk_factors:

        st.markdown("**⚠️ Factors that may increase churn risk**")

        for factor in risk_factors:
            st.write(f"• {factor}")

    # Display positive factors
    if positive_factors:

        st.markdown("**✅ Factors associated with lower churn**")

        for factor in positive_factors:
            st.write(f"• {factor}")

    if not risk_factors and not positive_factors:

        st.write(
            "No major risk or positive indicators identified "
            "from the selected customer characteristics."
        )


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

st.subheader("🤖 Model Information")

st.write(
    "The application uses a tuned Random Forest classification "
    "pipeline to predict customer churn."
)

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        "ROC-AUC",
        "84.21%"
    )

with model_col2:
    st.metric(
        "Recall",
        "77.01%"
    )

with model_col3:
    st.metric(
        "F1 Score",
        "62.95%"
    )

st.caption(
    "Evaluation metrics are based on the held-out test dataset. "
    "Performance may vary on new customer data."
)

top_features = pd.DataFrame({
    "Feature": [
        "Month-to-month contract",
        "Customer tenure",
        "Total charges",
        "Two-year contract",
        "No online security",
        "No technical support",
        "Monthly charges",
        "Fiber optic internet",
        "Electronic check payment"
    ],
    "Importance": [
        0.137940,
        0.126997,
        0.087457,
        0.075408,
        0.073328,
        0.060508,
        0.058937,
        0.055759,
        0.037716
    ]
})

st.bar_chart(
    top_features.set_index("Feature")
)

st.markdown("### 💡 What These Features Mean")

st.write(
    """
    The model found that contract type, customer tenure, charges,
    online security, technical support, internet service, and payment
    method were among the most useful features for distinguishing
    customers who churn from those who do not.
    """
)

st.caption(
    "These feature-importance scores describe the model's overall "
    "behavior across the dataset. They should not be interpreted as "
    "causal effects or as the probability contribution of an individual feature."
)

st.caption(
    "Feature importance indicates how useful each feature was to the "
    "Random Forest model across the dataset. It does not imply causation "
    "or represent an individual customer's probability of churn."
)

st.divider()

st.subheader("⚠️ Limitations & Responsible Use")

st.markdown(
    """
    - **Predictions are estimates:** The model predicts churn risk based
      on patterns learned from historical customer data.
    
    - **No causal conclusions:** A feature being important to the model
      does not mean it causes customer churn.
    
    - **Performance may vary:** Model performance on new customer data
      may differ from the evaluation results obtained on the test dataset.
    
    - **Risk thresholds are business-defined:** Low, Medium, and High
      Risk categories are based on selected probability thresholds and
      can be adjusted according to business requirements.
    
    - **Human judgment is important:** Predictions should support,
      rather than replace, customer-service and business decisions.
    
    - **Data quality matters:** Missing, incorrect, or significantly
      different customer information can affect prediction quality.
    """
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

st.subheader("📁 Batch CSV Prediction")

st.write(
    "Upload a CSV file containing customer information "
    "to generate churn predictions for multiple customers."
)
st.write("Don't have a CSV template? Download one below.")

template_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

template_data = pd.DataFrame(
    columns=template_columns
)

template_csv = template_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📄 Download CSV Template",
    data=template_csv,
    file_name="customer_churn_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_data = pd.read_csv(uploaded_file)

    st.success("CSV uploaded successfully!")

    st.write(
        f"Number of customers: {len(uploaded_data)}"
    )

    st.write(
        f"Number of columns: {len(uploaded_data.columns)}"
    )

    required_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    missing_columns = [
        column
        for column in required_features
        if column not in uploaded_data.columns
    ]

    extra_columns = [
        column
        for column in uploaded_data.columns
        if column not in required_features
        and column != "customerID"
    ]

    if missing_columns:

        st.error(
            "❌ Invalid CSV file. Missing required columns:"
        )

        st.write(missing_columns)

    else:

        st.success(
            "✅ CSV contains all required customer features."
        )
        if extra_columns:
            st.info(
                "ℹ️ Extra columns detected. "
                "They will be ignored during prediction."
            )

            st.write(
                "Extra columns:",
                extra_columns
            )

        st.dataframe(
            uploaded_data.head(10),
            use_container_width=True
        )

        st.divider()

        st.subheader("🔮 Batch Prediction")

        if st.button("Predict Churn for All Customers"):

            prediction_data = uploaded_data[
                required_features
            ].copy()

            numerical_features = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            for column in numerical_features:
                prediction_data[column] = pd.to_numeric(
                    prediction_data[column],
                    errors="coerce"
                )

            prediction_data[numerical_features] = (
                prediction_data[numerical_features]
                .fillna(0)
            )

            predictions = model.predict(
                prediction_data
            )

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1]

            result_data = uploaded_data.copy()

            result_data["ChurnPrediction"] = predictions

            result_data["ChurnProbability"] = probabilities

            result_data["ChurnPrediction"] = (
                result_data["ChurnPrediction"]
                .map({
                    0: "No",
                    1: "Yes"
                })
            )

            result_data["RiskLevel"] = pd.cut(
                result_data["ChurnProbability"],
                bins=[-0.01, 0.30, 0.60, 1.00],
                labels=[
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )

            st.success(
                "✅ Predictions generated successfully!"
            )

            total_customers = len(result_data)

            predicted_churn = (
                result_data["ChurnPrediction"] == "Yes"
            ).sum()

            predicted_no_churn = (
                result_data["ChurnPrediction"] == "No"
            ).sum()

            high_risk_customers = (
                result_data["RiskLevel"] == "High Risk"
            ).sum()

            average_probability = (
                result_data["ChurnProbability"].mean() * 100
            )


            st.subheader("📊 Prediction Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Customers",
                total_customers
            )

            col2.metric(
                "Predicted Churn",
                predicted_churn
            )

            col3.metric(
                "Predicted No Churn",
                predicted_no_churn
            )

            col4.metric(
                "High Risk",
                high_risk_customers
            )

            col5, col6 = st.columns(2)

            col5.metric(
                "Average Churn Probability",
                f"{average_probability:.2f}%"
            )

            medium_risk_customers = (
                result_data["RiskLevel"] == "Medium Risk"
            ).sum()

            col6.metric(
                "Medium Risk",
                medium_risk_customers
            )

            st.subheader("📊 Churn Prediction Overview")

            st.info(
                f"""
                **Business Interpretation:** Out of {total_customers:,} customers,
                the model predicts {predicted_churn:,} customers may churn.
                There are {high_risk_customers:,} customers classified as high risk
                who could be prioritized for retention campaigns.
                """

            )
            chart_data = pd.DataFrame({
                "Prediction": [
                    "Predicted Churn",
                    "Predicted No Churn"
                ],
                "Customers": [
                    predicted_churn,
                    predicted_no_churn
                ]
            })

            st.bar_chart(
                chart_data.set_index("Prediction")
            )

            st.subheader("⚠️ Customer Risk Distribution")

            risk_data = (
                result_data["RiskLevel"]
                .value_counts()
                .reindex([
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ])
                .fillna(0)
                .astype(int)
            )

            st.bar_chart(risk_data)
            st.subheader("📌 Churn Risk Levels")

            st.info(
                """
                **Low Risk:** Churn probability below 30%  
                **Medium Risk:** Churn probability between 30% and 60%  
                **High Risk:** Churn probability above 60%
    
                These risk levels are based on the model's predicted churn probability
                and are intended to help prioritize customer retention efforts.
                """

            )
            st.subheader("🚨 High-Risk Customers")

            high_risk_data = result_data[
                result_data["RiskLevel"] == "High Risk"
            ].copy()

            high_risk_data = high_risk_data.sort_values(
                by="ChurnProbability",
                ascending=False
            )

            st.write(
                f"Showing {len(high_risk_data)} high-risk customers."
            )

            st.dataframe(
                high_risk_data[
                    [
                        "customerID",
                        "tenure",
                        "MonthlyCharges",
                        "Contract",
                        "InternetService",
                        "PaymentMethod",
                        "ChurnProbability",
                        "RiskLevel"
                    ]
                ].head(20),
                use_container_width=True
            )

            high_risk_csv = high_risk_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="🚨 Download High-Risk Customers",
                data=high_risk_csv,
                file_name="high_risk_customers.csv",
                mime="text/csv"
            )

            st.subheader("📋 Prediction Results")

            st.dataframe(
                result_data.head(20),
                use_container_width=True
            )

            csv_data = result_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_data,
                file_name="churn_predictions.csv",
                mime="text/csv"
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
