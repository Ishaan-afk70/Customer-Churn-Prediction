import os
import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from churn import load_model  # Import your actual app code here

def download_data():
    dataset_name = "blastchar/telco-customer-churn"
    zip_file = "telco-customer-churn.zip"
    csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Download and unzip dataset
    os.system(f"kaggle datasets download -d {dataset_name} -f {csv_file} --force")
    os.system(f"unzip -o {zip_file}")

    # Load into DataFrame
    df = pd.read_csv(csv_file)
    return df

# Mock inputs for prediction test
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

# ✅ Test 1: Dataset loading
def test_load_dataset():
    df = download_data()
    assert isinstance(df, pd.DataFrame), "Dataset should be a pandas DataFrame"
    assert not df.empty, "Dataset should not be empty"

# ✅ Test 2: Model loading
def test_load_model():
    model_data, encoders = load_model()
    assert model_data is not None, "Model data should be loaded"
    assert encoders is not None, "Encoders should be loaded"

# ✅ Test 3: Model prediction with mock inputs
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
    assert 0 <= prediction_prob <= 1, "Probability should be between 0 and 1"

# ✅ Test 4: Accuracy on full dataset (optional)
def test_model_accuracy():
    df = download_data()

    # Drop ID and prepare features and target
    df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})  # Encode labels

    # Load model
    model_data, encoders = load_model()
    loaded_model = model_data["model"]

    # Encode features using saved encoders
    for col, encoder in encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col])

    # Predict and check accuracy
    y_pred = loaded_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.75, f"Accuracy should be > 75%, got {accuracy:.2%}"
