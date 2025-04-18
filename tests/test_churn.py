import os
import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from churn import load_model  # This should load your trained model and encoders

# -----------------------------
# Utility: Load dataset locally
# -----------------------------
def load_local_data():
    csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found in project root.")
    df = pd.read_csv(csv_file)

    # Fix invalid TotalCharges: convert blanks to NaN and drop them
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    return df

# -----------------------------
# Mock inputs for testing
# -----------------------------
@pytest.fixture
def mock_inputs():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
    }

# -----------------------------
# Test 1: Dataset loading
# -----------------------------
def test_load_dataset():
    df = load_local_data()
    assert isinstance(df, pd.DataFrame), "Dataset should be a pandas DataFrame"
    assert not df.empty, "Dataset should not be empty"

# -----------------------------
# Test 2: Model loading
# -----------------------------
def test_load_model():
    model_data, encoders = load_model()
    assert model_data is not None, "Model data should be loaded"
    assert encoders is not None, "Encoders should be loaded"

# -----------------------------
# Test 3: Model predictions on mock input
# -----------------------------
def test_model_predictions(mock_inputs):
    model_data, encoders = load_model()
    loaded_model = model_data["model"]

    input_df = pd.DataFrame([mock_inputs])

    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    prediction = loaded_model.predict(input_df)[0]
    prediction_prob = loaded_model.predict_proba(input_df)[0][1]

    assert prediction in [0, 1], "Prediction should be 0 or 1"
    assert 0 <= prediction_prob <= 1, "Probability must be between 0 and 1"

# -----------------------------
# Test 4: Accuracy check on real dataset
# -----------------------------
def test_model_accuracy():
    df = load_local_data()

    # Drop customerID and prepare features and target
    df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})  # Encode target labels

    # Load model and encoders
    model_data, encoders = load_model()
    loaded_model = model_data["model"]

    # Apply encoding to features
    for col, encoder in encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col])

    # Predict and evaluate
    y_pred = loaded_model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    assert accuracy > 0.75, f"Model accuracy too low: {accuracy:.2%}"
