from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    ForecastRequest,
    ForecastResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    CategorizeRequest,
    CategorizeResponse,
)
from app.services.ml_engine import (
    predict_spending_forecast,
    detect_spending_anomalies,
    predict_transaction_category,
)

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.post("/forecast", response_model=ForecastResponse)
async def get_spending_forecast(payload: ForecastRequest):
    """Predict month-end total expense based on historical transaction velocity."""
    try:
        result = predict_spending_forecast(payload.transactions)
        return ForecastResponse(user_id=payload.user_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Forecast Error: {str(e)}")


@router.post("/anomalies", response_model=AnomalyDetectionResponse)
async def get_spending_anomalies(payload: AnomalyDetectionRequest):
    """Detect unusual transaction amounts using an Isolation Forest model."""
    try:
        result = detect_spending_anomalies(payload.transactions, payload.contamination)
        return AnomalyDetectionResponse(user_id=payload.user_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Anomaly Error: {str(e)}")


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_transaction(payload: CategorizeRequest):
    """Predict the spending category for a free-text transaction description."""
    try:
        return predict_transaction_category(payload.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Categorization Error: {str(e)}")
