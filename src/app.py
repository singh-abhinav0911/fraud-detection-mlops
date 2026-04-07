from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    features: list[float]  # 30 features

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    
    return {
        "fraud": bool(prediction),
        "probability": round(float(probability), 4),
        "risk": "HIGH" if probability > 0.5 else "LOW"
    }