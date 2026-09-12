from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 1. Load the trained pipeline
# We do this at startup so it only loads once, making the API super fast.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Run `python -m src.train` first.")

# 2. Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn probability using a trained ML pipeline.",
    version="1.0.0"
)

# 3. Define the Input Data Schema using Pydantic
# This automatically validates the data coming in and generates the Swagger UI!
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Predicts the probability of a customer churning.
    """
    # Convert the incoming Pydantic data into a Pandas DataFrame
    # (Our Scikit-Learn pipeline expects a DataFrame to apply column transformations)
    input_df = pd.DataFrame([customer.model_dump()])
    
    try:
        # Predict probability: model.predict_proba returns [prob_no_churn, prob_churn]
        probabilities = model.predict_proba(input_df)
        churn_probability = float(probabilities[0][1])
        
        # Determine class based on 0.5 threshold
        churn_prediction = "Yes" if churn_probability > 0.5 else "No"
        
        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(churn_probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 5. Health check endpoint (useful for CI/CD later)
@app.get("/health")
def health_check():
    return {"status": "healthy"}
