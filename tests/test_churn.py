import os
import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score
import streamlit as st
from your_streamlit_app_code import load_model  # Import your actual app code here

def download_data():
    dataset_name = "blastchar/telco-customer-churn"
    zip_file = "telco-customer-churn.zip"
    csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Download the dataset
    os.system(f"kaggle datasets download -d {dataset_name} -f {csv_file} --force")

    # Unzip the dataset
    os.system(f"unzip -o {zip_file}")

    # Read dataset into a DataFrame
    df = pd.read_csv(csv_file)
    return df
# Mock inputs
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

# Test case 1: Test dataset loading
def test_load_dataset():
    # Simulate downloading and loading dataset
    df = download_data()
    
    # Assertion: Ensure the dataset is loaded correctly
    assert isinstance(df, pd.DataFrame), "Dataset should be a pandas DataFrame"
    assert not df.empty, "Dataset should not be empty"

# Test case 2: Test model loading
def test_load_model():
    model_data, encoders = load_model()  # Replace with actual method to load the model
    assert model_data is not None, "Model data should be loaded"
    assert encoders is not None, "Encoders should be loaded"

# Test case 3: Test model predictions on mock inputs
def test_model_predictions(mock_inputs):
    # Load the model
    model_data, encoders = load_model()
    loaded_model = model_data["model"]

    # Prepare mock input data for prediction
    input_df = pd.DataFrame([mock_inputs])
    
    # Apply encoding
    encoded_input = input_df.copy()
    for col, encoder in encoders.items():
        if col in encoded_input.columns:
            encoded_input[col] = encoder.transform(encoded_input[col])

    # Make prediction
    prediction = loaded_model.predict(encoded_input)[0]
    prediction_prob = loaded_model.predict_proba(encoded_input)[0][1]

    # Assertions
    assert prediction in [0, 1], "Prediction should be binary (0 or 1)"
    assert 0 <= prediction_prob <= 1, "Prediction probability should be between 0 and 1"

# Test case 4: Test accuracy of predictions on a test set (Optional)
def test_model_accuracy():
    # Load dataset
    df = download_data()
    
    # Preprocessing steps similar to your actual application
    X = df.drop(columns=["Churn"])  # Features
    y = df["Churn"]  # Target variable

    # Load the model
    model_data, encoders = load_model()
    loaded_model = model_data["model"]
    
    # Encode the data
    for col, encoder in encoders.items():
        if col in X.columns:
            X[col] = encoder.transform(X[col])
    
    # Make predictions
    y_pred = loaded_model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    assert accuracy > 0.75, f"Model accuracy should be greater than 75%, but got {accuracy:.2f}"

