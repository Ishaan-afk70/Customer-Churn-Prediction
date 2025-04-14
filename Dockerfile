# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements if you have it, or write them directly here
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy the app code and model files into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit config to disable browser auto-launch
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Command to run the app
CMD ["streamlit", "run", "churn.py", "--server.port=8501", "--server.enableCORS=false"]
