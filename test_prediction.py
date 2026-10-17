import pandas as pd

from src.predict import (
    predict_loan_status,
    predict_loan_probability
)

input_data = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}])

prediction = predict_loan_status(input_data)
probabilities = predict_loan_probability(input_data)

print("Predicted Loan Status:", prediction)
print("Class Probabilities:", probabilities)