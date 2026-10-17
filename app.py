
import streamlit as st
import pandas as pd

from src.predict import (
    predict_loan_status,
    predict_loan_probability
)


# Page configuration
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="centered"
)


# App heading
st.title("Loan Prediction System")

st.write(
    "Enter the applicant details below to predict the loan status."
)

st.info(
    "Educational project: Loan_Status was synthetically generated. "
    "Predictions are not real loan approval decisions."
)


# Input form
with st.form("loan_prediction_form"):

    st.subheader("Applicant Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0.0,
        value=150.0
    )

    loan_term = st.selectbox(
        "Loan Amount Term (months)",
        [360, 180, 120, 84, 60, 36, 12]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button(
        "Predict Loan Status"
    )


# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    try:
        prediction = predict_loan_status(input_data)

        st.subheader("Prediction Result")

        if prediction == "Y":
            st.success("Predicted class: Y")
        else:
            st.error("Predicted class: N")

        probabilities = predict_loan_probability(input_data)

        st.subheader("Class Probabilities")

        probability_df = pd.DataFrame({
            "Class": list(probabilities.keys()),
            "Probability": [
                round(float(value) * 100, 2)
                for value in probabilities.values()
            ]
        })

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These probabilities reflect the model's estimates "
            "for synthetic training labels, not actual loan "
            "approval probabilities."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
