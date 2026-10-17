
import joblib
import pandas as pd


# Load the saved model
model = joblib.load("models/loan_model.pkl")


def predict_loan_status(input_data):
    """
    Predict loan status using applicant details.

    input_data should be a pandas DataFrame
    containing the same feature columns used during training.
    """

    prediction = model.predict(input_data)

    return prediction[0]


def predict_loan_probability(input_data):
    """
    Return class probabilities if the model supports them.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The selected model does not support predict_proba."
        )

    probabilities = model.predict_proba(input_data)

    return dict(zip(model.classes_, probabilities[0]))
