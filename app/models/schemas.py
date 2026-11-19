from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# ==========================================
# 1. Base Transaction Schemas
# ==========================================

class TransactionItem(BaseModel):
    id: Optional[str] = None
    amount: float = Field(..., description="Transaction amount in currency unit", json_schema_extra={"example": 450.0})
    category: str = Field(..., description="Expense or income category", json_schema_extra={"example": "Food"})
    date: str = Field(..., description="ISO formatted date string YYYY-MM-DD", json_schema_extra={"example": "2026-09-24"})
    description: Optional[str] = Field(None, description="Optional note or merchant name", json_schema_extra={"example": "Dinner at restaurant"})
    type: str = Field("expense", description="'expense' or 'income'", json_schema_extra={"example": "expense"})


# ==========================================
# 2. Machine Learning Schemas (Forecast & Anomalies)
# ==========================================

class ForecastRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", json_schema_extra={"example": "usr_12345"})
    transactions: List[TransactionItem] = Field(..., description="Historical transaction array")

class ForecastResponse(BaseModel):
    user_id: str
    projected_month_end_spend: float = Field(..., description="Projected total month-end expense")
    daily_avg_spend: float = Field(..., description="Average daily spending rate")
    model_r2: Optional[float] = Field(None, description="R^2 of the fitted trend; null when too little history exists")
    prediction_interval_90: List[float] = Field(..., description="90% interval from regression residuals (low, high)")
    insights: List[str] = Field(default=[], description="Generated narrative insights from the forecast")


class AnomalyDetectionRequest(BaseModel):
    user_id: str
    transactions: List[TransactionItem]
    contamination: Optional[float] = Field(0.05, ge=0.001, le=0.5, description="Expected proportion of anomalies (5% default)")

class AnomalyItem(BaseModel):
    transaction_id: Optional[str] = None
    amount: float
    category: str
    date: str
    anomaly_score: float = Field(..., description="Isolation Forest decision score")
    reason: str = Field(..., description="Explanation why transaction was flagged")

class AnomalyDetectionResponse(BaseModel):
    user_id: str
    anomalies_found: int
    anomalies: List[AnomalyItem]


class CategorizeRequest(BaseModel):
    description: str = Field(..., description="Transaction description to categorize", json_schema_extra={"example": "dinner at restaurant"})


class CategorizeResponse(BaseModel):
    description: str
    predicted_category: str
    confidence: float
    message: Optional[str] = None


# ==========================================
# 3. AI Chat & Tool Calling Schemas
# ==========================================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Message author role")
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(..., description="User's natural language input", json_schema_extra={"example": "I spent ₹500 on dinner today"})
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation messages")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational LLM response back to the user")
    tools_used: List[str] = Field(default=[], description="Names of tools executed for this reply")