from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. Base Transaction Schemas
# ==========================================

class TransactionItem(BaseModel):
    id: Optional[str] = None
    amount: float = Field(..., description="Transaction amount in currency unit", example=450.0)
    category: str = Field(..., description="Expense or income category", example="Food & Dining")
    date: str = Field(..., description="ISO formatted date string YYYY-MM-DD", example="2026-09-24")
    description: Optional[str] = Field(None, description="Optional note or merchant name", example="Dinner at restaurant")
    type: str = Field("expense", description="'expense' or 'income'", example="expense")


# ==========================================
# 2. Machine Learning Schemas (Forecast & Anomalies)
# ==========================================

class ForecastRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", example="usr_12345")
    transactions: List[TransactionItem] = Field(..., description="Historical transaction array")

class ForecastResponse(BaseModel):
    user_id: str
    projected_month_end_spend: float = Field(..., description="Scikit-learn projected total month-end expense")
    daily_avg_spend: float = Field(..., description="Average daily spending rate")
    confidence_score: float = Field(..., description="Model confidence score (0.0 to 1.0)")
    insights: List[str] = Field(default=[], description="Generated narrative insights from the forecast")


class AnomalyDetectionRequest(BaseModel):
    user_id: str
    transactions: List[TransactionItem]
    contamination: Optional[float] = Field(0.05, description="Expected proportion of anomalies (5% default)")

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


# ==========================================
# 3. AI Chat & Tool Calling Schemas
# ==========================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(..., description="User's natural language input", example="I spent ₹500 on dinner today")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation messages")

class ToolCallPayload(BaseModel):
    tool_name: str = Field(..., description="Target action e.g., 'create_transaction', 'query_analytics'")
    parameters: Dict[str, Any] = Field(..., description="Extracted tool argument parameters")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational LLM response back to the user")
    tool_call: Optional[ToolCallPayload] = Field(None, description="Structured tool call payload to send back to Node backend")
    rag_context_used: bool = Field(False, description="Whether pgvector semantic search was triggered")