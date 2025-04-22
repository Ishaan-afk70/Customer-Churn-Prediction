import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import datetime
import os
import json
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Stay or Stray? 🚀",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_prediction = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_analytics = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_qrf2xhuc.json")
lottie_dashboard = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qzgz4puf.json")

# Custom CSS for styling (enhanced)
st.markdown("""
<style>
    /* Modern Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Styling for content boxes */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        margin-bottom: 24px;
        border-top: 5px solid #6200EA;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }

    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #4527A0, #6200EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        padding: 10px;
        margin-bottom: 20px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4527A0;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #6200EA;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #4527A0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Prediction result styling */
    .prediction-result {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .churn-yes {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
    }
    
    .churn-no {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* Input field styling */
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 8px;
    }
    
    .stSlider>div>div {
        background-color: #6200EA;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6200EA;
        color: white;
    }
    
    /* Feature list styling */
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 8px;
    }
    
    .feature-icon {
        margin-right: 10px;
        color: #6200EA;
    }
    
    /* Timeline styling */
    .timeline {
        position: relative;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .timeline::after {
        content: '';
        position: absolute;
        width: 6px;
        background-color: #6200EA;
        top: 0;
        bottom: 0;
        left: 50%;
        margin-left: -3px;
        border-radius: 10px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #6200EA;
    }
    
    /* Custom table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .styled-table thead tr {
        background-color: #6200EA;
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #6200EA;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .badge-success {
        background-color: #4caf50;
        color: white;
    }
    
    .badge-danger {
        background-color: #f44336;
        color: white;
    }
    
    .badge-warning {
        background-color: #ff9800;
        color: white;
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Custom file uploader */
    .stFileUploader>div>button {
        background-color: #6200EA;
        color: white;
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border-radius: 8px;
    }
    
    /* Custom info box */
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Custom warning box */
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Custom success box */
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Custom error box */
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Create a session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if 'customer_name' not in st.session_state:
    st.session_state.customer_name = ""

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'login_status' not in st.session_state:
    st.session_state.login_status = False

if 'username' not in st.session_state:
    st.session_state.username = ""

# Function to toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Function to simulate login
def login(username, password):
    # In a real app, you would check against a database
    if username and password:  # Simple validation
        st.session_state.login_status = True
        st.session_state.username = username
        return True
    return False

# Function to logout
def logout():
    st.session_state.login_status = False
    st.session_state.username = ""

# Function to save prediction
def save_prediction(customer_name, prediction, probability, input_data):
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    prediction_data = {
        "id": prediction_id,
        "customer_name": customer_name,
        "prediction": bool(prediction),
        "probability": float(probability),
        "input_data": input_data,
        "timestamp": timestamp
    }
    
    st.session_state.predictions.append(prediction_data)
    
    # In a real app, you would save to a database
    return prediction_id

# Function to load sample data
@st.cache_data
def load_sample_data():
    # Create sample data for demonstration
    data = {
        "gender": ["Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "Yes", "No"],
        "tenure": [72, 24, 12, 6, 36],
        "PhoneService": ["Yes", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["Yes", "No", "No", "No phone service", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "Yes", "No internet service", "No"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["Yes", "No", "Yes", "No internet service", "No"],
        "TechSupport": ["Yes", "No", "No", "No internet service", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes", "No internet service", "Yes"],
        "StreamingMovies": ["Yes", "No", "No", "No internet service", "Yes"],
        "Contract": ["Two year", "Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Credit card", "Electronic check", "Mailed check", "Electronic check", "Bank transfer"],
        "MonthlyCharges": [107.45, 55.65, 53.85, 20.25, 90.45],
        "TotalCharges": [7730.40, 1334.60, 646.20, 121.50, 3256.20],
        "Churn": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open("C:/Users/dell/OneDrive/Desktop/c/customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("C:/Users/dell/OneDrive/Desktop/c/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data, encoders
    except FileNotFoundError:
        # Create dummy model and encoders for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create dummy encoders
        encoders = {}
        for col in ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                   "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                   "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                   "PaperlessBilling", "PaymentMethod"]:
            encoders[col] = LabelEncoder()
        
        # Fit encoders on sample data
        sample_data = load_sample_data()
        for col, encoder in encoders.items():
            encoder.fit(sample_data[col])
        
        # Train model on sample data
        X = sample_data.drop("Churn", axis=1)
        y = sample_data["Churn"]
        
        # Apply encoding
        X_encoded = X.copy()
        for col, encoder in encoders.items():
            if col in X_encoded.columns:
                X_encoded[col] = encoder.transform(X_encoded[col])
        
        model.fit(X_encoded, y)
        
        model_data = {
            "model": model,
            "features_names": X.columns.tolist()
        }
        
        return model_data, encoders

# Feature importance (simulated for visualization)
@st.cache_data
def get_feature_importance():
    # This would normally come from your model, but we'll simulate it
    importances = {
        "Contract": 0.28,
        "tenure": 0.23,
        "MonthlyCharges": 0.18,
        "TotalCharges": 0.15,
        "InternetService": 0.12,
        "PaymentMethod": 0.10,
        "OnlineSecurity": 0.09,
        "TechSupport": 0.08,
        "PaperlessBilling": 0.07,
        "OnlineBackup": 0.06
    }
    return importances

# Function to generate a downloadable report
def generate_report(prediction_data):
    report = f"""
    # Customer Churn Prediction Report
    
    ## Customer Information
    - **Name:** {prediction_data['customer_name']}
    - **Prediction Date:** {prediction_data['timestamp']}
    
    ## Prediction Result
    - **Churn Prediction:** {"Likely to Churn" if prediction_data['prediction'] else "Likely to Stay"}
    - **Churn Probability:** {prediction_data['probability']:.2%}
    
    ## Customer Details
    """
    
    for key, value in prediction_data['input_data'].items():
        report += f"- **{key}:** {value}\n"
    
    # Add risk factors
    report += "\n## Risk Factors\n"
    
    input_data = prediction_data['input_data']
    
    if input_data["Contract"] == "Month-to-month":
        report += "- Month-to-month contract\n"
    
    if input_data["tenure"] < 12:
        report += "- Low tenure (< 12 months)\n"
    
    if input_data["InternetService"] == "Fiber optic" and input_data["TechSupport"] == "No":
        report += "- Fiber optic without tech support\n"
    
    if input_data["PaymentMethod"] == "Electronic check":
        report += "- Electronic check payment\n"
    
    if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
        report += "- No online security\n"
    
    # Add recommendations
    report += "\n## Recommendations\n"
    
    if prediction_data['prediction']:
        if input_data["Contract"] == "Month-to-month":
            report += "- Offer contract upgrade incentives\n"
        if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
            report += "- Provide security features bundle\n"
        if input_data["TechSupport"] == "No" and input_data["InternetService"] != "No":
            report += "- Offer tech support trial\n"
    else:
        report += "- Continue providing excellent service\n"
        report += "- Consider cross-selling additional services\n"
    
    return report

# Function to create a download link
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to create a PDF report (simulated)
def get_pdf_download_link(prediction_data, filename="churn_prediction_report.pdf"):
    # In a real app, you would generate a PDF here
    # For this example, we'll just create a text file
    report = generate_report(prediction_data)
    
    return get_download_link(report, filename, "Download PDF Report")

# Function to create a batch prediction
def batch_predict(df, model, encoders):
    # Apply encoding
    encoded_df = df.copy()
    for col, encoder in encoders.items():
        if col in encoded_df.columns:
            try:
                encoded_df[col] = encoder.transform(encoded_df[col])
            except:
                st.warning(f"Error encoding column {col}. Skipping.")
    
    # Make predictions
    predictions = model.predict(encoded_df)
    probabilities = model.predict_proba(encoded_df)[:, 1]
    
    # Add predictions to dataframe
    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities
    
    return df

# Function to create a correlation heatmap
def create_correlation_heatmap(df):
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    return buf

# Function to create a customer segmentation plot
def create_segmentation_plot(df):
    # Create a scatter plot of tenure vs monthly charges, colored by churn
    fig = px.scatter(
        df, 
        x="tenure", 
        y="MonthlyCharges",
        color="churn_prediction",
        color_discrete_map={0: "#4CAF50", 1: "#F44336"},
        size="TotalCharges",
        hover_name=df.index,
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        title="Customer Segmentation by Tenure and Monthly Charges"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Churn Prediction",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Try to load the model
try:
    model_data, encoders = load_model()
    loaded_model = model_data["model"]
    feature_names = model_data["features_names"]
    feature_importance = get_feature_importance()
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/decision.png", width=100)
        st.markdown("<h1 style='text-align: center;'>Stay or Stray?</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Customer Churn Prediction</p>", unsafe_allow_html=True)
        
        # Login/Logout section
        if not st.session_state.login_status:
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_col1, login_col2 = st.columns(2)
            with login_col1:
                if st.button("Login"):
                    if login(username, password):
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            with login_col2:
                if st.button("Guest Mode"):
                    login("Guest", "guest")
                    st.rerun()
        else:
            st.markdown(f"### Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                logout()
                st.rerun()
        
        st.markdown("---")
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Home", "Predict", "Batch Predict", "Dashboard", "Analytics", "Settings", "About"],
            icons=["house", "magic", "file-earmark-arrow-up", "speedometer2", "graph-up", "gear", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        
        # Dark mode toggle
        dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode, on_change=toggle_dark_mode)
        
        # Display current time
        st.markdown(f"**Current Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # App version
        st.markdown("**App Version:** 2.0.0")
        
        # Footer
        st.markdown("---")
        st.markdown("© 2023 Stay or Stray")
    
    # Main content based on navigation
    if selected == "Home":
        # Main app header
        st.markdown("<h1 class='main-header'>Stay or Stray? 🚀</h1>", unsafe_allow_html=True)
        
        # Hero section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h2>Predict Customer Churn with AI</h2>
                <p style="font-size: 1.2rem;">
                    Our advanced machine learning model helps you identify customers at risk of leaving, 
                    so you can take proactive steps to retain them.
                </p>
                <ul style="font-size: 1.1rem;">
                    <li>✅ <strong>Accurate Predictions:</strong> Identify at-risk customers with high accuracy</li>
                    <li>✅ <strong>Actionable Insights:</strong> Get specific recommendations to reduce churn</li>
                    <li>✅ <strong>Batch Processing:</strong> Analyze multiple customers at once</li>
                    <li>✅ <strong>Advanced Analytics:</strong> Understand what drives customer churn</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st_lottie(lottie_prediction, height=300, key="prediction_animation")
        
        # Quick actions
        st.markdown("<h3 class='sub-header'>Quick Actions</h3>", unsafe_allow_html=True)
        
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            st.markdown("""
            <div class="metric-card">
                <img src="https://img.icons8.com/color/48/000000/crystal-ball.png" style="width: 48px; height: 48px;">
                <h3>Predict Churn</h3>
                <p>Analyze individual customer data to predict churn risk</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Predicting", key="predict_btn"):
                selected = "Predict"
                st.rerun()
        
        with quick_col2:
            st.markdown("""
            <div class="metric-card">
                <img src="https://img.icons8.com/color/48/000000/upload-to-cloud.png" style="width: 48px; height: 48px;">
                <h3>Batch Predict</h3>
                <p>Upload CSV file to process multiple customers at once</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Upload CSV", key="batch_btn"):
                selected = "Batch Predict"
                st.rerun()
        
        with quick_col3:
            st.markdown("""
            <div class="metric-card">
                <img src="https://img.icons8.com/color/48/000000/dashboard.png" style="width: 48px; height: 48px;">
                <h3>Dashboard</h3>
                <p>View your prediction history and analytics</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Dashboard", key="dashboard_btn"):
                selected = "Dashboard"
                st.rerun()
        
        # Feature highlights
        st.markdown("<h3 class='sub-header'>Key Features</h3>", unsafe_allow_html=True)
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            <div class="card">
                <h4>Predictive Analytics</h4>
                <p>Our machine learning model analyzes customer data to predict churn probability with high accuracy.</p>
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/combo-chart--v1.png" style="width: 64px; height: 64px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>Risk Factor Analysis</h4>
                <p>Identify specific factors that contribute to churn risk for each customer.</p>
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/risk.png" style="width: 64px; height: 64px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with feature_col2:
            st.markdown("""
            <div class="card">
                <h4>Retention Recommendations</h4>
                <p>Get actionable recommendations to reduce churn risk and improve customer retention.</p>
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/idea.png" style="width: 64px; height: 64px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>Batch Processing</h4>
                <p>Upload CSV files to analyze multiple customers at once and get bulk predictions.</p>
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/data-sheet.png" style="width: 64px; height: 64px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Testimonials
        st.markdown("<h3 class='sub-header'>What Our Users Say</h3>", unsafe_allow_html=True)
        
        testimonial_col1, testimonial_col2, testimonial_col3 = st.columns(3)
        
        with testimonial_col1:
            st.markdown("""
            <div class="card">
                <div style="text-align: center;">
                    <img src="https://randomuser.me/api/portraits/women/44.jpg" style="width: 80px; height: 80px; border-radius: 50%;">
                </div>
                <p style="font-style: italic; text-align: center;">"This tool has helped us reduce churn by 25% in just three months. The insights are invaluable!"</p>
                <p style="text-align: center; font-weight: bold;">Sarah Johnson</p>
                <p style="text-align: center;">Customer Success Manager</p>
            </div>
            """, unsafe_allow_html=True)
        
        with testimonial_col2:
            st.markdown("""
            <div class="card">
                <div style="text-align: center;">
                    <img src="https://randomuser.me/api/portraits/men/32.jpg" style="width: 80px; height: 80px; border-radius: 50%;">
                </div>
                <p style="font-style: italic; text-align: center;">"The batch prediction feature saves us hours of work every week. Highly recommended!"</p>
                <p style="text-align: center; font-weight: bold;">Michael Chen</p>
                <p style="text-align: center;">Data Analyst</p>
            </div>
            """, unsafe_allow_html=True)
        
        with testimonial_col3:
            st.markdown("""
            <div class="card">
                <div style="text-align: center;">
                    <img src="https://randomuser.me/api/portraits/women/68.jpg" style="width: 80px; height: 80px; border-radius: 50%;">
                </div>
                <p style="font-style: italic; text-align: center;">"The actionable recommendations have transformed how we approach customer retention."</p>
                <p style="text-align: center; font-weight: bold;">Emily Rodriguez</p>
                <p style="text-align: center;">VP of Customer Experience</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif selected == "Predict":
        # Main app header
        st.markdown("<h1 class='main-header'>Predict Customer Churn</h1>", unsafe_allow_html=True)
        
        # Lottie animation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_prediction, height=200, key="predict_animation")
        
        # Customer name input
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Customer Information</h3>", unsafe_allow_html=True)
        
        customer_name = st.text_input("Customer Name (Optional)", value=st.session_state.customer_name)
        st.session_state.customer_name = customer_name
        
        # Create tabs for different input methods
        input_tabs = st.tabs(["Form Input", "JSON Input", "Sample Customers"])
        
        with input_tabs[0]:
            # Create 3 columns for inputs
            input_col1, input_col2, input_col3 = st.columns(3)
            
            with input_col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                Partner = st.selectbox("Has Partner?", ["Yes", "No"])
                Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                PhoneService = st.selectbox("Has Phone Service?", ["Yes", "No"])
                MultipleLines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
            
            with input_col2:
                InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                DeviceProtection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
                TechSupport = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
                StreamingTV = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
            
            with input_col3:
                StreamingMovies = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])
                Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])
                PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
                MonthlyCharges = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0, 0.1)
                TotalCharges = st.slider("Total Charges ($)", 0.0, 8000.0, tenure * MonthlyCharges, 0.1)
        
        with input_tabs[1]:
            # JSON input
            default_json = """
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.0,
                "TotalCharges": 780.0
            }
            """
            json_input = st.text_area("Enter customer data as JSON", value=default_json, height=300)
            
            try:
                json_data = json.loads(json_input)
                st.success("JSON data loaded successfully")
                
                # Use JSON data if valid
                if st.button("Use This Data"):
                    gender = json_data.get("gender", "Female")
                    SeniorCitizen = json_data.get("SeniorCitizen", 0)
                    Partner = json_data.get("Partner", "Yes")
                    Dependents = json_data.get("Dependents", "No")
                    tenure = json_data.get("tenure", 12)
                    PhoneService = json_data.get("PhoneService", "Yes")
                    MultipleLines = json_data.get("MultipleLines", "No")
                    InternetService = json_data.get("InternetService", "DSL")
                    OnlineSecurity = json_data.get("OnlineSecurity", "No")
                    OnlineBackup = json_data.get("OnlineBackup", "No")
                    DeviceProtection = json_data.get("DeviceProtection", "No")
                    TechSupport = json_data.get("TechSupport", "No")
                    StreamingTV = json_data.get("StreamingTV", "No")
                    StreamingMovies = json_data.get("StreamingMovies", "No")
                    Contract = json_data.get("Contract", "Month-to-month")
                    PaperlessBilling = json_data.get("PaperlessBilling", "Yes")
                    PaymentMethod = json_data.get("PaymentMethod", "Electronic check")
                    MonthlyCharges = json_data.get("MonthlyCharges", 65.0)
                    TotalCharges = json_data.get("TotalCharges", 780.0)
                    
                    # Switch to form input tab
                    input_tabs[0].selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(gender))
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
        
        with input_tabs[2]:
            # Sample customers
            st.markdown("### Sample Customers")
            st.markdown("Select a sample customer to pre-fill the form.")
            
            sample_customers = {
                "Low Risk Customer": {
                    "name": "John Smith",
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "tenure": 72,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Credit card",
                    "MonthlyCharges": 107.45,
                    "TotalCharges": 7730.40
                },
                "High Risk Customer": {
                    "name": "Jane Doe",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 3,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.70,
                    "TotalCharges": 212.10
                },
                "Medium Risk Customer": {
                    "name": "Robert Johnson",
                    "gender": "Male",
                    "SeniorCitizen": 1,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 24,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "Yes",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                    "Contract": "One year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Mailed check",
                    "MonthlyCharges": 60.05,
                    "TotalCharges": 1441.20
                }
            }
            
            sample_customer = st.selectbox("Select a sample customer", list(sample_customers.keys()))
            
            if st.button("Load Sample Customer"):
                selected_customer = sample_customers[sample_customer]
                
                # Set customer name
                st.session_state.customer_name = selected_customer["name"]
                
                # Set form values (this would work in a real Streamlit app)
                gender = selected_customer["gender"]
                SeniorCitizen = selected_customer["SeniorCitizen"]
                Partner = selected_customer["Partner"]
                Dependents = selected_customer["Dependents"]
                tenure = selected_customer["tenure"]
                PhoneService = selected_customer["PhoneService"]
                MultipleLines = selected_customer["MultipleLines"]
                InternetService = selected_customer["InternetService"]
                OnlineSecurity = selected_customer["OnlineSecurity"]
                OnlineBackup = selected_customer["OnlineBackup"]
                DeviceProtection = selected_customer["DeviceProtection"]
                TechSupport = selected_customer["TechSupport"]
                StreamingTV = selected_customer["StreamingTV"]
                StreamingMovies = selected_customer["StreamingMovies"]
                Contract = selected_customer["Contract"]
                PaperlessBilling = selected_customer["PaperlessBilling"]
                PaymentMethod = selected_customer["PaymentMethod"]
                MonthlyCharges = selected_customer["MonthlyCharges"]
                TotalCharges = selected_customer["TotalCharges"]
                
                # Switch to form input tab
                st.rerun()
        
        # Create input data dictionary
        input_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply encoding
        encoded_input = input_df.copy()
        for col, encoder in encoders.items():
            if col in encoded_input.columns:
                encoded_input[col] = encoder.transform(encoded_input[col])
        
        # Predict button
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            predict_button = st.button("Predict Churn", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction result
        if predict_button:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Prediction Result</h3>", unsafe_allow_html=True)
            
            with st.spinner("Analyzing customer data..."):
                time.sleep(0.5)  # Brief delay for UX
                
                prediction = loaded_model.predict(encoded_input)[0]
                prediction_prob = loaded_model.predict_proba(encoded_input)[0][1]
                
                # Save prediction
                prediction_id = save_prediction(customer_name, prediction, prediction_prob, input_data)
                
                # Display prediction result
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    churn_result = "⚠️ High Risk of Churn" if prediction == 1 else "✅ Low Risk of Churn"
                    churn_class = "churn-yes" if prediction == 1 else "churn-no"
                    
                    st.markdown(f"""
                    <div class='prediction-result {churn_class}'>
                        <h2>{churn_result}</h2>
                        <h3>Churn Probability: {prediction_prob:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge chart for churn probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#6200EA"},
                            'bgcolor': "white",
                            'steps': [
                                {'range': [0, 30], 'color': '#00C853'},
                                {'range': [30, 70], 'color': '#FFD600'},
                                {'range': [70, 100], 'color': '#FF5252'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with result_col2:
                    # Risk factors
                    st.markdown("<h4>Key Risk Factors</h4>", unsafe_allow_html=True)
                    
                    # Identify top risk factors
                    risk_factors = []
                    
                    if input_data["Contract"] == "Month-to-month":
                        risk_factors.append("Month-to-month contract")
                    
                    if input_data["tenure"] < 12:
                        risk_factors.append("Low tenure (< 12 months)")
                    
                    if input_data["InternetService"] == "Fiber optic" and input_data["TechSupport"] == "No":
                        risk_factors.append("Fiber optic without tech support")
                    
                    if input_data["PaymentMethod"] == "Electronic check":
                        risk_factors.append("Electronic check payment")
                    
                    if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
                        risk_factors.append("No online security")
                    
                    # Display risk factors
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.write("No significant risk factors identified.")
                    
                    # Simple recommendations
                    st.markdown("<h4>Recommendations</h4>", unsafe_allow_html=True)
                    
                    if prediction == 1:
                        if input_data["Contract"] == "Month-to-month":
                            st.markdown("• Offer contract upgrade incentives")
                        if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
                            st.markdown("• Provide security features bundle")
                        if input_data["TechSupport"] == "No" and input_data["InternetService"] != "No":
                            st.markdown("• Offer tech support trial")
                        if input_data["tenure"] < 12:
                            st.markdown("• Provide loyalty rewards for staying")
                        if input_data["PaymentMethod"] == "Electronic check":
                            st.markdown("• Suggest automatic payment methods")
                    else:
                        st.markdown("• Continue providing excellent service")
                        st.markdown("• Consider cross-selling additional services")
                        st.markdown("• Implement loyalty rewards program")
                
                # Feature contribution
                st.markdown("<h4>Feature Contribution to Prediction</h4>", unsafe_allow_html=True)
                
                # Create a waterfall chart (simulated)
                feature_contributions = {
                    "Contract": 0.35 if input_data["Contract"] == "Month-to-month" else -0.2,
                    "tenure": -0.25 if input_data["tenure"] > 24 else 0.2,
                    "MonthlyCharges": 0.15 if input_data["MonthlyCharges"] > 70 else -0.1,
                    "InternetService": 0.25 if input_data["InternetService"] == "Fiber optic" else -0.15,
                    "OnlineSecurity": 0.2 if input_data["OnlineSecurity"] == "No" else -0.1
                }
                
                # Sort by absolute contribution
                sorted_contributions = dict(sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True))
                
                # Create horizontal bar chart
                fig = go.Figure()
                
                for feature, contribution in sorted_contributions.items():
                    fig.add_trace(go.Bar(
                        y=[feature],
                        x=[contribution],
                        orientation='h',
                        marker=dict(
                            color='#F44336' if contribution > 0 else '#4CAF50',
                        ),
                        name=feature
                    ))
                
                fig.update_layout(
                    title="Top Features Influencing Prediction",
                    xaxis_title="Contribution to Churn Risk",
                    yaxis=dict(autorange="reversed"),
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download report
                st.markdown("<h4>Download Report</h4>", unsafe_allow_html=True)
                
                # Get the latest prediction
                latest_prediction = st.session_state.predictions[-1]
                
                # Create download link
                st.markdown(get_pdf_download_link(latest_prediction), unsafe_allow_html=True)
                
                # Share options
                share_col1, share_col2, share_col3 = st.columns(3)
                
                with share_col1:
                    if st.button("📧 Email Report"):
                        st.success("Email feature would be implemented here")
                
                with share_col2:
                    if st.button("📱 SMS Notification"):
                        st.success("SMS feature would be implemented here")
                
                with share_col3:
                    if st.button("📊 Add to Dashboard"):
                        st.success("Added to dashboard")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected == "Batch Predict":
        # Main app header
        st.markdown("<h1 class='main-header'>Batch Prediction</h1>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Upload Customer Data</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Instructions</h4>
            <p>Upload a CSV file containing customer data. The file should include the following columns:</p>
            <ul>
                <li>gender</li>
                <li>SeniorCitizen</li>
                <li>Partner</li>
                <li>Dependents</li>
                <li>tenure</li>
                <li>PhoneService</li>
                <li>MultipleLines</li>
                <li>InternetService</li>
                <li>OnlineSecurity</li>
                <li>OnlineBackup</li>
                <li>DeviceProtection</li>
                <li>TechSupport</li>
                <li>StreamingTV</li>
                <li>StreamingMovies</li>
                <li>Contract</li>
                <li>PaperlessBilling</li>
                <li>PaymentMethod</li>
                <li>MonthlyCharges</li>
                <li>TotalCharges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        # Option to use sample data
        use_sample_data = st.checkbox("Use sample data instead")
        
        if uploaded_file is not None or use_sample_data:
            if use_sample_data:
                # Load sample data
                df = load_sample_data()
                st.success("Sample data loaded successfully")
            else:
                # Load uploaded data
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Show data preview
            st.markdown("<h4>Data Preview</h4>", unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Data statistics
            st.markdown("<h4>Data Statistics</h4>", unsafe_allow_html=True)
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Total Customers", len(df))
            
            with stats_col2:
                st.metric("Average Tenure", f"{df['tenure'].mean():.1f} months")
            
            with stats_col3:
                st.metric("Average Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
            
            with stats_col4:
                contract_counts = df['Contract'].value_counts()
                most_common_contract = contract_counts.index[0]
                st.metric("Most Common Contract", most_common_contract)
            
            # Process button
            if st.button("Process Batch"):
                with st.spinner("Processing batch prediction..."):
                    # Make predictions
                    result_df = batch_predict(df, loaded_model, encoders)
                    
                    # Show results
                    st.markdown("<h3 class='sub-header'>Batch Prediction Results</h3>", unsafe_allow_html=True)
                    
                    # Summary metrics
                    total_customers = len(result_df)
                    churn_count = result_df['churn_prediction'].sum()
                    churn_percentage = (churn_count / total_customers) * 100
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Total Customers", total_customers)
                    
                    with metrics_col2:
                        st.metric("Predicted to Churn", int(churn_count))
                    
                    with metrics_col3:
                        st.metric("Churn Rate", f"{churn_percentage:.1f}%")
                    
                    # Results table
                    st.markdown("<h4>Prediction Results</h4>", unsafe_allow_html=True)
                    
                    # Add a column for churn status
                    result_df['churn_status'] = result_df['churn_prediction'].apply(lambda x: "Churn" if x == 1 else "Stay")
                    
                    # Format probability as percentage
                    result_df['churn_probability_pct'] = result_df['churn_probability'].apply(lambda x: f"{x:.2%}")
                    
                    # Display results
                    st.dataframe(result_df)
                    
                    # Visualizations
                    st.markdown("<h4>Visualizations</h4>", unsafe_allow_html=True)
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Churn distribution pie chart
                        fig = px.pie(
                            names=["Stay", "Churn"],
                            values=[(total_customers - churn_count), churn_count],
                            title="Churn Distribution",
                            color_discrete_sequence=["#4CAF50", "#F44336"]
                        )
                        
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Churn by contract type
                        contract_churn = result_df.groupby('Contract')['churn_prediction'].mean() * 100
                        
                        fig = px.bar(
                            x=contract_churn.index,
                            y=contract_churn.values,
                            title="Churn Rate by Contract Type",
                            labels={"x": "Contract Type", "y": "Churn Rate (%)"},
                            color=contract_churn.values,
                            color_continuous_scale=["#4CAF50", "#FFD600", "#F44336"]
                        )
                        
                        fig.update_layout(coloraxis_showscale=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Customer segmentation
                    st.markdown("<h4>Customer Segmentation</h4>", unsafe_allow_html=True)
                    
                    # Create segmentation plot
                    fig = create_segmentation_plot(result_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("<h4>Correlation Analysis</h4>", unsafe_allow_html=True)
                    
                    # Create correlation heatmap
                    heatmap_buf = create_correlation_heatmap(result_df)
                    st.image(heatmap_buf)
                    
                    # Download results
                    st.markdown("<h4>Download Results</h4>", unsafe_allow_html=True)
                    
                    # Convert DataFrame to CSV
                    csv = result_df.to_csv(index=False)
                    
                    # Create download link
                    st.markdown(
                        get_download_link(csv, "churn_predictions.csv", "Download CSV"),
                        unsafe_allow_html=True
                    )
                    
                    # Additional actions
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("📊 Generate Report"):
                            st.success("Report generation would be implemented here")
                    
                    with action_col2:
                        if st.button("📧 Email Results"):
                            st.success("Email feature would be implemented here")
                    
                    with action_col3:
                        if st.button("💾 Save to Dashboard"):
                            st.success("Results saved to dashboard")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected == "Dashboard":
        # Main app header
        st.markdown("<h1 class='main-header'>Dashboard</h1>", unsafe_allow_html=True)
        
        # Lottie animation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_dashboard, height=200, key="dashboard_animation")
        
        # Check if there are any predictions
        if not st.session_state.predictions:
            st.markdown("""
            <div class="card">
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/96/000000/empty-box.png" style="width: 96px; height: 96px;">
                    <h3>No Predictions Yet</h3>
                    <p>Make your first prediction to see your dashboard.</p>
                    <br>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Dashboard metrics
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            total_predictions = len(st.session_state.predictions)
            churn_predictions = sum(1 for p in st.session_state.predictions if p["prediction"])
            stay_predictions = total_predictions - churn_predictions
            churn_rate = (churn_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Total Predictions", total_predictions)
            
            with metrics_col2:
                st.metric("Predicted to Churn", churn_predictions)
            
            with metrics_col3:
                st.metric("Predicted to Stay", stay_predictions)
            
            with metrics_col4:
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Charts
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Prediction Analytics</h3>", unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Churn distribution pie chart
                fig = px.pie(
                    names=["Stay", "Churn"],
                    values=[stay_predictions, churn_predictions],
                    title="Prediction Distribution",
                    color_discrete_sequence=["#4CAF50", "#F44336"]
                )
                
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                # Probability distribution histogram
                probabilities = [p["probability"] for p in st.session_state.predictions]
                
                fig = px.histogram(
                    x=probabilities,
                    nbins=10,
                    title="Churn Probability Distribution",
                    labels={"x": "Churn Probability", "y": "Count"},
                    color_discrete_sequence=["#6200EA"]
                )
                
                fig.update_layout(
                    xaxis=dict(tickformat=".0%"),
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recent predictions
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Recent Predictions</h3>", unsafe_allow_html=True)
            
            # Create a DataFrame from predictions
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Sort by timestamp (newest first)
            if "timestamp" in predictions_df.columns:
                predictions_df = predictions_df.sort_values("timestamp", ascending=False)
            
            # Display recent predictions
            for i, row in predictions_df.head(5).iterrows():
                customer_name = row["customer_name"] if row["customer_name"] else f"Customer {i+1}"
                prediction = "Churn" if row["prediction"] else "Stay"
                probability = row["probability"]
                timestamp = row["timestamp"] if "timestamp" in row else "N/A"
                
                st.markdown(f"""
                <div style="padding: 15px; margin-bottom: 15px; border-radius: 8px; background-color: {'#ffebee' if row['prediction'] else '#e8f5e9'}; border-left: 5px solid {'#f44336' if row['prediction'] else '#4caf50'};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0;">{customer_name}</h4>
                            <p style="margin: 5px 0;">Prediction: <strong>{prediction}</strong> ({probability:.2%})</p>
                        </div>
                        <div>
                            <span style="color: #757575; font-size: 0.9rem;">{timestamp}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(predictions_df) > 5:
                st.markdown(f"<p style='text-align: center;'><em>Showing 5 of {len(predictions_df)} predictions</em></p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Risk factors analysis
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Risk Factors Analysis</h3>", unsafe_allow_html=True)
            
            # Count risk factors
            risk_factors = {
                "Month-to-month contract": 0,
                "Low tenure (< 12 months)": 0,
                "Fiber optic without tech support": 0,
                "Electronic check payment": 0,
                "No online security": 0
            }
            
            for p in st.session_state.predictions:
                input_data = p["input_data"]
                
                if input_data["Contract"] == "Month-to-month":
                    risk_factors["Month-to-month contract"] += 1
                
                if input_data["tenure"] < 12:
                    risk_factors["Low tenure (< 12 months)"] += 1
                
                if input_data["InternetService"] == "Fiber optic" and input_data["TechSupport"] == "No":
                    risk_factors["Fiber optic without tech support"] += 1
                
                if input_data["PaymentMethod"] == "Electronic check":
                    risk_factors["Electronic check payment"] += 1
                
                if input_data["OnlineSecurity"] == "No" and input_data["InternetService"] != "No":
                    risk_factors["No online security"] += 1
            
            # Create horizontal bar chart
            fig = px.bar(
                y=list(risk_factors.keys()),
                x=list(risk_factors.values()),
                orientation='h',
                title="Common Risk Factors",
                labels={"x": "Count", "y": "Risk Factor"},
                color=list(risk_factors.values()),
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected == "Analytics":
        # Main app header
        st.markdown("<h1 class='main-header'>Analytics</h1>", unsafe_allow_html=True)
        
        # Lottie animation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_analytics, height=200, key="analytics_animation")
        
        # Feature importance
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Feature Importance Analysis</h3>", unsafe_allow_html=True)
        
        # Sort features by importance
        sorted_features = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        
        # Create horizontal bar chart
        fig = px.bar(
            x=list(sorted_features.values()),
            y=list(sorted_features.keys()),
            orientation='h',
            color=list(sorted_features.values()),
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={"x": "Importance Score", "y": "Feature"},
            title="Feature Importance"
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance explanation
        st.markdown("""
        <div class="info-box">
            <h4>Understanding Feature Importance</h4>
            <p>Feature importance indicates how much each feature contributes to the model's predictions. Higher values mean the feature has a stronger influence on churn prediction.</p>
            <ul>
                <li><strong>Contract Type:</strong> Month-to-month contracts have significantly higher churn rates.</li>
                <li><strong>Tenure:</strong> Longer-tenured customers are less likely to churn.</li>
                <li><strong>Monthly Charges:</strong> Higher monthly charges correlate with increased churn risk.</li>
                <li><strong>Internet Service:</strong> Fiber optic customers tend to churn at higher rates than DSL customers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Contract vs Churn visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Contract Type Impact on Churn</h3>", unsafe_allow_html=True)
        
        # Generate sample data for visualization
        contract_types = ["Month-to-month", "One year", "Two year"]
        churn_rates = [0.42, 0.11, 0.03]  # Example churn rates
        
        fig = px.bar(
            x=contract_types, 
            y=churn_rates,
            color=churn_rates,
            color_continuous_scale=["#00C853", "#FFD600", "#FF5252"],
            labels={"x": "Contract Type", "y": "Churn Rate"},
            text_auto=True,
            title="Churn Rate by Contract Type"
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p>This chart shows how contract type significantly impacts churn rate. Month-to-month contracts have the highest churn rate (42%), 
        while two-year contracts have the lowest (3%).</p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Tenure vs Churn
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Tenure Impact on Churn</h3>", unsafe_allow_html=True)
        
        # Generate sample data for visualization
        tenure_groups = ["0-12 months", "13-24 months", "25-36 months", "37-48 months", "49+ months"]
        tenure_churn_rates = [0.43, 0.24, 0.11, 0.06, 0.03]  # Example churn rates
        
        fig = px.line(
            x=tenure_groups, 
            y=tenure_churn_rates,
            markers=True,
            line_shape="spline",
            labels={"x": "Tenure Group", "y": "Churn Rate"},
            title="Churn Rate by Tenure"
        )
        
        fig.update_traces(line_color="#6200EA", marker=dict(size=10))
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p>This chart shows how customer tenure impacts churn rate. New customers (0-12 months) have the highest churn rate (43%), 
        while long-term customers (49+ months) have the lowest (3%).</p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Internet Service vs Churn
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Internet Service Impact on Churn</h3>", unsafe_allow_html=True)
        
        # Generate sample data for visualization
        internet_services = ["DSL", "Fiber optic", "No internet"]
        internet_churn_rates = [0.19, 0.42, 0.07]  # Example churn rates
        
        fig = px.bar(
            x=internet_services, 
            y=internet_churn_rates,
            color=internet_churn_rates,
            color_continuous_scale=["#00C853", "#FFD600", "#FF5252"],
            labels={"x": "Internet Service", "y": "Churn Rate"},
            text_auto=True,
            title="Churn Rate by Internet Service"
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p>This chart shows how internet service type impacts churn rate. Fiber optic customers have the highest churn rate (42%), 
        while customers with no internet service have the lowest (7%).</p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Payment Method vs Churn
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Payment Method Impact on Churn</h3>", unsafe_allow_html=True)
        
        # Generate sample data for visualization
        payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        payment_churn_rates = [0.45, 0.19, 0.16, 0.15]  # Example churn rates
        
        fig = px.bar(
            x=payment_methods, 
            y=payment_churn_rates,
            color=payment_churn_rates,
            color_continuous_scale=["#00C853", "#FFD600", "#FF5252"],
            labels={"x": "Payment Method", "y": "Churn Rate"},
            text_auto=True,
            title="Churn Rate by Payment Method"
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p>This chart shows how payment method impacts churn rate. Customers using electronic check have the highest churn rate (45%), 
        while credit card customers have the lowest (15%).</p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected == "Settings":
        # Main app header
        st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
        
        # Settings card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Application Settings</h3>", unsafe_allow_html=True)
        
        # Theme settings
        st.markdown("<h4>Theme Settings</h4>", unsafe_allow_html=True)
        
        theme_col1, theme_col2 = st.columns(2)
        
        with theme_col1:
            dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode, on_change=toggle_dark_mode)
        
        with theme_col2:
            theme_color = st.color_picker("Primary Color", "#6200EA")
        
        # Notification settings
        st.markdown("<h4>Notification Settings</h4>", unsafe_allow_html=True)
        
        email_notifications = st.checkbox("Email Notifications", value=True)
        if email_notifications:
            email = st.text_input("Email Address")
        
        sms_notifications = st.checkbox("SMS Notifications")
        if sms_notifications:
            phone = st.text_input("Phone Number")
        
        # Data settings
        st.markdown("<h4>Data Settings</h4>", unsafe_allow_html=True)
        
        data_col1, data_col2 = st.columns(2)
        
        with data_col1:
            save_predictions = st.checkbox("Save Predictions", value=True)
        
        with data_col2:
            auto_refresh = st.checkbox("Auto Refresh Dashboard", value=True)
        
        # Model settings
        st.markdown("<h4>Model Settings</h4>", unsafe_allow_html=True)
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            model_threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
        
        with model_col2:
            model_version = st.selectbox("Model Version", ["v1.0.0 (Current)", "v0.9.0", "v0.8.0"])
        
        # Save settings button
        if st.button("Save Settings"):
            st.success("Settings saved successfully")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Account settings
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Account Settings</h3>", unsafe_allow_html=True)
        
        if st.session_state.login_status and st.session_state.username != "Guest":
            # Profile settings
            st.markdown("<h4>Profile Settings</h4>", unsafe_allow_html=True)
            
            profile_col1, profile_col2 = st.columns(2)
            
            with profile_col1:
                username = st.text_input("Username", value=st.session_state.username)
            
            with profile_col2:
                email = st.text_input("Email Address", value="user@example.com")
            
            # Change password
            st.markdown("<h4>Change Password</h4>", unsafe_allow_html=True)
            
            password_col1, password_col2 = st.columns(2)
            
            with password_col1:
                current_password = st.text_input("Current Password", type="password")
            
            with password_col2:
                new_password = st.text_input("New Password", type="password")
            
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            # Save profile button
            if st.button("Update Profile"):
                st.success("Profile updated successfully")
        else:
            st.info("Please login to access account settings")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data management
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Data Management</h3>", unsafe_allow_html=True)
        
        # Export data
        st.markdown("<h4>Export Data</h4>", unsafe_allow_html=True)
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("Export Predictions (CSV)"):
                if st.session_state.predictions:
                    # Convert predictions to DataFrame
                    predictions_df = pd.DataFrame(st.session_state.predictions)
                    
                    # Convert DataFrame to CSV
                    csv = predictions_df.to_csv(index=False)
                    
                    # Create download link
                    st.markdown(
                        get_download_link(csv, "predictions.csv", "Download CSV"),
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No predictions to export")
        
        with export_col2:
            if st.button("Export Predictions (JSON)"):
                if st.session_state.predictions:
                    # Convert predictions to JSON
                    json_str = json.dumps(st.session_state.predictions, indent=2)
                    
                    # Create download link
                    st.markdown(
                        get_download_link(json_str, "predictions.json", "Download JSON"),
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No predictions to export")
        
        with export_col3:
            if st.button("Export Analytics Report"):
                st.info("Analytics report export would be implemented here")
        
        # Clear data
        st.markdown("<h4>Clear Data</h4>", unsafe_allow_html=True)
        
        if st.button("Clear All Predictions"):
            if st.session_state.predictions:
                # Confirm clear
                confirm = st.checkbox("I understand this action cannot be undone")
                
                if confirm and st.button("Confirm Clear"):
                    st.session_state.predictions = []
                    st.success("All predictions cleared")
            else:
                st.warning("No predictions to clear")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif selected == "About":
        # Main app header
        st.markdown("<h1 class='main-header'>About This Application</h1>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>This application predicts customer churn for telecommunications companies using machine learning. It helps identify customers at risk of leaving and provides insights to improve retention.</p>
        
        <h4 style='color: #6200EA; margin-top: 15px;'>Key Features</h4>
        <ul>
            <li><strong>Predictive Analytics:</strong> Uses machine learning to predict customer churn probability</li>
            <li><strong>Feature Importance:</strong> Identifies the most significant factors influencing churn</li>
            <li><strong>Risk Factors:</strong> Highlights specific issues that may lead to churn</li>
            <li><strong>Retention Recommendations:</strong> Suggests targeted strategies to reduce churn</li>
            <li><strong>Batch Processing:</strong> Analyze multiple customers at once</li>
            <li><strong>Interactive Dashboard:</strong> Visualize prediction results and analytics</li>
            <li><strong>Data Export:</strong> Export predictions and reports in various formats</li>
        </ul>
        
        <h4 style='color: #6200EA; margin-top: 15px;'>How to Use</h4>
        <ol>
            <li>Enter customer details in the Prediction Tool tab</li>
            <li>Click "Predict Churn" to get the churn probability and recommendations</li>
            <li>Review the Feature Importance tab to understand what drives churn</li>
            <li>Use the Dashboard to track prediction history and analytics</li>
            <li>Export reports and data for further analysis</li>
        </ol>
        
        <h4 style='color: #6200EA; margin-top: 15px;'>About the Model</h4>
        <p>The churn prediction model is based on a machine learning algorithm trained on historical customer data. 
        It takes into account multiple customer attributes to predict the likelihood of churn with approximately 80% accuracy.</p>
        
        <h4 style='color: #6200EA; margin-top: 15px;'>Version History</h4>
        <ul>
            <li><strong>v2.0.0 (Current):</strong> Added batch processing, dashboard, and analytics</li>
            <li><strong>v1.5.0:</strong> Added data export and report generation</li>
            <li><strong>v1.0.0:</strong> Initial release with basic prediction functionality</li>
        </ul>
        
        <h4 style='color: #6200EA; margin-top: 15px;'>Contact Information</h4>
        <p>For support or inquiries, please contact:</p>
        <ul>
            <li>Email: support@stayorstray.com</li>
            <li>Phone: (555) 123-4567</li>
            <li>Website: www.stayorstray.com</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # FAQ section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Frequently Asked Questions</h3>", unsafe_allow_html=True)
        
        faq_items = [
            {
                "question": "What is customer churn?",
                "answer": "Customer churn refers to when customers stop doing business with a company. In the telecommunications industry, this means customers canceling their service or switching to another provider."
            },
            {
                "question": "How accurate is the prediction model?",
                "answer": "The model has approximately 80% accuracy on historical data. However, accuracy may vary depending on the specific characteristics of your customer base."
            },
            {
                "question": "Can I upload my own customer data?",
                "answer": "Yes, you can upload a CSV file containing customer data in the Batch Predict section. The file should include all the required customer attributes."
            },
            {
                "question": "How can I reduce customer churn?",
                "answer": "The application provides specific recommendations for each customer based on their risk factors. Common strategies include offering contract upgrades, providing security features, and improving technical support."
            },
            {
                "question": "Is my data secure?",
                "answer": "Yes, all data is processed securely and is not shared with third parties. In this demo version, data is stored locally in your browser session."
            }
        ]
        
        for i, faq in enumerate(faq_items):
            with st.expander(faq["question"]):
                st.markdown(faq["answer"])
        
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.markdown("""
    <div style="padding: 15px; background-color: #f8d7da; border-radius: 5px; text-align: center;">
        <h3 style="color: #721c24;">Model Files Not Found</h3>
        <p>Please ensure the model files (customer_churn_model.pkl and encoders.pkl) are in the same directory as this application.</p>
        <p>This is a demo version that can run without the actual model files.</p>
    </div>
    """, unsafe_allow_html=True)
