from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.models.schemas import TransactionItem
from app.services.ml_engine import (
    predict_spending_forecast,
    detect_spending_anomalies,
    predict_transaction_category
)

app = FastAPI(
    title="Vaulto AI Microservice",
    description="Machine Learning predictions, anomaly detection, and transaction categorization",
    version="1.0.0"
)

# ---------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------
class ForecastRequest(BaseModel):
    user_id: str
    transactions: List[TransactionItem]

class AnomalyRequest(BaseModel):
    user_id: str
    contamination: float = 0.05
    transactions: List[TransactionItem]

class CategorizeRequest(BaseModel):
    description: str

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Vaulto AI Microservice is running ✅"}

@app.post("/api/v1/ml/forecast", tags=["Machine Learning"])
def get_spending_forecast(payload: ForecastRequest):
    result = predict_spending_forecast(payload.transactions)
    return {"user_id": payload.user_id, **result}

@app.post("/api/v1/ml/anomalies", tags=["Machine Learning"])
def get_spending_anomalies(payload: AnomalyRequest):
    result = detect_spending_anomalies(payload.transactions, payload.contamination)
    return {"user_id": payload.user_id, **result}

@app.post("/api/v1/ml/categorize", tags=["Machine Learning"])
def categorize_transaction(payload: CategorizeRequest):
    return predict_transaction_category(payload.description)
