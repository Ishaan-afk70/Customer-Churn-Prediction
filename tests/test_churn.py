import pytest
import pandas as pd
import pickle


@pytest.fixture(scope="module")
def load_model_and_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def test_prediction(load_model_and_encoders):
    model, encoders = load_model_and_encoders

    input_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 351.75
    }

    df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    prediction = model.predict(df)
    assert prediction[0] in [0, 1]
