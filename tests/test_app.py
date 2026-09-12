import pytest
import joblib
import os
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app


# setup test client and load model
client = TestClient(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')

# A sample customer data
VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

# 1. API test
def test_health_check():
    """Test if the health endpoint returns 200 and 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Test if the predict endpoint returns valid churn probabilities."""
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    data = response.json()
    assert 'churn_prediction' in data
    assert 'churn_probability' in data
    assert data['churn_prediction'] in ["Yes", "No"]
    assert 0.0 <= data['churn_probability'] <= 1.0

def test_predict_invalid_data():
    """Test if the API rejects bad data (e.g., wrong type for SeniorCitizen)."""
    bad_customer = VALID_CUSTOMER.copy()
    bad_customer["SeniorCitizen"] = "Not a number"  # Should be int

    response = client.post("/predict", json=bad_customer)
    # FastAPI should return a 422 Unprocessable Entity error
    assert response.status_code == 422


# 2. Model test
def test_model_accuracy():
    """
    Model CI/CD Test:
    Train the model to ensure it meets our minimum accuracy threshold.
    If a developer accidentally breaks the pipeline, this test will fail and stop deployment! 
    """

    from src.preprocess import load_data, clean_data, build_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = load_data()
    X, y = clean_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Assert the model is at least 75% accurate
    assert acc > 0.75, f"Model accuracy {acc:.4f} is below the acceptable threshold of 0.75!"

