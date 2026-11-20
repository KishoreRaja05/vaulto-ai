from pydantic import BaseModel, BeforeValidator, Field
from typing import Annotated, Any, Dict, List, Literal, Optional

ALLOWED_CATEGORIES = (
    "Food", "Transport", "Shopping", "Bills", "Entertainment",
    "Healthcare", "Education", "Rent", "Salary", "Freelance", "Other",
)


def normalize_category(value):
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    for allowed in ALLOWED_CATEGORIES:
        if value.strip().lower() == allowed.lower():
            return allowed
    return "Other"


Category = Annotated[
    Literal[
        "Food", "Transport", "Shopping", "Bills", "Entertainment",
        "Healthcare", "Education", "Rent", "Salary", "Freelance", "Other",
    ],
    BeforeValidator(normalize_category),
]

# ==========================================
# 1. Base Transaction Schemas
# ==========================================

class TransactionItem(BaseModel):
    id: Optional[str] = None
    amount: float = Field(..., description="Transaction amount in currency unit", json_schema_extra={"example": 450.0})
    category: Category = Field(..., description="Expense or income category", json_schema_extra={"example": "Food"})
    date: str = Field(..., description="ISO formatted date string YYYY-MM-DD", json_schema_extra={"example": "2026-09-24"})
    description: Optional[str] = Field(None, description="Optional note or merchant name", json_schema_extra={"example": "Dinner at restaurant"})
    type: str = Field("expense", description="'expense' or 'income'", json_schema_extra={"example": "expense"})


# ==========================================
# 2. Machine Learning Schemas (Forecast & Anomalies)
# ==========================================

class ForecastRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", json_schema_extra={"example": "usr_12345"})
    transactions: List[TransactionItem] = Field(..., description="Historical transaction array")

class ForecastContext(BaseModel):
    """Structured numbers the chat LLM is allowed to use for a forecast answer."""

    month_income: float
    month_expenses_so_far: float
    days_elapsed: int
    days_in_month: int
    days_remaining: int
    projected_month_end_expenses: float
    budget: Optional[float] = None
    top_category: Optional[str] = None
    top_category_amount: float = 0.0
    number_of_transactions: int
    months_of_history: int
    confidence: Literal["low", "medium", "high"]
    excluded_outliers: List[Dict[str, Any]] = Field(default=[])


class ForecastResponse(BaseModel):
    user_id: str
    projected_month_end_spend: float = Field(..., description="Projected total month-end expense")
    daily_avg_spend: float = Field(..., description="Average daily spending rate")
    model_r2: Optional[float] = Field(None, description="R^2 of the fitted trend; null when too little history exists")
    prediction_interval_90: Optional[List[float]] = Field(None, description="90% interval; omitted when confidence is low")
    confidence: Literal["low", "medium", "high"] = "low"
    forecast_method: str = "rule_based"
    context: ForecastContext
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