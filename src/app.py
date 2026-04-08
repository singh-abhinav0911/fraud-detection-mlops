from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI(title="Fraud Detection API")

# Lazy-loaded globals
_model = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        with open("model.pkl", "rb") as f:
            _model = pickle.load(f)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        with open("scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
    return _scaler

# Input schema
class Transaction(BaseModel):
    features: list[float]  # 30 features

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=422, detail="Expected exactly 30 features")
    
    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = get_scaler().transform(data)
    prediction = get_model().predict(data_scaled)[0]
    probability = get_model().predict_proba(data_scaled)[0][1]
    
    return {
        "fraud": bool(prediction),
        "probability": round(float(probability), 4),
        "risk": "HIGH" if probability > 0.5 else "LOW"
    }