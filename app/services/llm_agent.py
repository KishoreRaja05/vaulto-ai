"""Groq/Llama tool-calling agent.

The LLM only produces structured tool calls. Every data operation goes through
the Node backend over HTTP with the caller's JWT; this service never writes to
the database directly.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse, TransactionItem
from app.services.ml_engine import detect_spending_anomalies, predict_spending_forecast

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are Vaulto's finance assistant. Use the provided tools to answer "
    "questions about the user's transactions and to record transactions they "
    "ask you to add. Never invent transaction data. When a tool returns an "
    "error, explain it plainly to the user. Be concise."
)


class BackendError(Exception):
    """Raised when the Node backend cannot serve a data request."""


# --- Untrusted tool arguments are validated before any execution -----------

class GetSpendingSummaryArgs(BaseModel):
    start_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class CreateTransactionArgs(BaseModel):
    amount: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    description: Optional[str] = None
    type: str = Field("expense", pattern=r"^(expense|income)$")


class PredictMonthlySpendingArgs(BaseModel):
    pass


class DetectAnomalyArgs(BaseModel):
    contamination: float = Field(0.05, ge=0.001, le=0.5)


TOOL_ARGS = {
    "get_spending_summary": GetSpendingSummaryArgs,
    "create_transaction": CreateTransactionArgs,
    "predict_monthly_spending": PredictMonthlySpendingArgs,
    "detect_anomaly": DetectAnomalyArgs,
}


def _tool_definition(name: str, description: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": TOOL_ARGS[name].model_json_schema(),
        },
    }


TOOL_DEFINITIONS = [
    _tool_definition("get_spending_summary", "Summarize the user's spending by category."),
    _tool_definition("create_transaction", "Record a new transaction for the user."),
    _tool_definition("predict_monthly_spending", "Forecast the user's month-end spending."),
    _tool_definition("detect_anomaly", "Find unusual transactions in the user's history."),
]


# --- Node backend access (the only path to data; no direct DB access) ------

def backend_request(
    method: str,
    path: str,
    *,
    jwt: Optional[str],
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Any:
    if not jwt:
        raise BackendError("Missing Authorization header; cannot access your account data.")
    url = settings.NODE_BACKEND_URL.rstrip("/") + path
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.request(
                method, url, headers={"Authorization": jwt}, params=params, json=body
            )
    except httpx.RequestError as exc:
        raise BackendError(f"Cannot reach the Vaulto backend ({exc.__class__.__name__}).") from exc

    if response.status_code == 401:
        raise BackendError("The Vaulto backend rejected your session (401). Please sign in again.")
    if response.status_code == 403:
        raise BackendError("The Vaulto backend denied access to this data (403).")
    if response.status_code >= 400:
        raise BackendError(f"The Vaulto backend returned status {response.status_code}.")
    return response.json() if response.content else {}


def fetch_transactions(user_id: str, jwt: Optional[str]) -> List[Dict[str, Any]]:
    data = backend_request("GET", "/transactions", jwt=jwt, params={"user_id": user_id})
    if isinstance(data, dict):
        data = data.get("transactions", data.get("data", []))
    return data if isinstance(data, list) else []


def to_transaction_items(rows: List[Dict[str, Any]]) -> List[TransactionItem]:
    items = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            items.append(TransactionItem(
                id=str(row.get("id")) if row.get("id") is not None else None,
                amount=float(row["amount"]),
                category=str(row.get("category") or "Other"),
                date=str(row["date"]),
                description=str(row.get("description") or ""),
                type=str(row.get("type") or "expense"),
            ))
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return items


# --- Tool implementations --------------------------------------------------

def get_spending_summary(args: GetSpendingSummaryArgs, user_id: str, jwt: Optional[str]) -> Dict[str, Any]:
    rows = fetch_transactions(user_id, jwt)
    items = [
        t for t in to_transaction_items(rows)
        if (not args.start_date or str(t.date)[:10] >= args.start_date)
        and (not args.end_date or str(t.date)[:10] <= args.end_date)
    ]
    expenses = [t for t in items if t.type == "expense"]
    by_category: Dict[str, float] = {}
    for item in expenses:
        by_category[item.category] = by_category.get(item.category, 0.0) + item.amount
    return {
        "transaction_count": len(items),
        "total_expense": round(sum(by_category.values()), 2),
        "by_category": {k: round(v, 2) for k, v in by_category.items()},
    }


def create_transaction(args: CreateTransactionArgs, user_id: str, jwt: Optional[str]) -> Dict[str, Any]:
    body = {"user_id": user_id, **args.model_dump(exclude_none=True)}
    result = backend_request("POST", "/transactions", jwt=jwt, body=body)
    return {"created": True, "transaction": result}


def predict_monthly_spending(args: PredictMonthlySpendingArgs, user_id: str, jwt: Optional[str]) -> Dict[str, Any]:
    return predict_spending_forecast(to_transaction_items(fetch_transactions(user_id, jwt)))


def detect_anomaly(args: DetectAnomalyArgs, user_id: str, jwt: Optional[str]) -> Dict[str, Any]:
    return detect_spending_anomalies(
        to_transaction_items(fetch_transactions(user_id, jwt)), args.contamination
    )


TOOL_HANDLERS = {
    "get_spending_summary": get_spending_summary,
    "create_transaction": create_transaction,
    "predict_monthly_spending": predict_monthly_spending,
    "detect_anomaly": detect_anomaly,
}


def execute_tool(name: str, raw_arguments: Any, user_id: str, jwt: Optional[str]) -> Dict[str, Any]:
    if name not in TOOL_HANDLERS:
        return {"error": f"Unknown tool '{name}'."}
    try:
        data = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
        args = TOOL_ARGS[name].model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}

    try:
        return TOOL_HANDLERS[name](args, user_id, jwt)
    except BackendError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"{name} failed: {exc.__class__.__name__}: {str(exc)[:200]}"}


# --- Chat orchestration ----------------------------------------------------

def run_chat(payload: ChatRequest, jwt: Optional[str]) -> ChatResponse:
    api_key = settings.GROQ_API_KEY
    if not api_key or api_key.lower().startswith("your"):
        return ChatResponse(
            reply="AI chat is not configured. Set GROQ_API_KEY to enable it.", tools_used=[]
        )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in payload.history or []:
        messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": payload.message})

    try:
        client = Groq(api_key=api_key)
    except Exception as exc:
        return ChatResponse(
            reply=f"AI service could not be initialized: {exc.__class__.__name__}.", tools_used=[]
        )

    tools_used: List[str] = []
    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except Exception as exc:
            detail = str(exc)[:200]
            return ChatResponse(
                reply=f"AI service error: {exc.__class__.__name__}: {detail}",
                tools_used=tools_used,
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return ChatResponse(reply=message.content or "I have no response to share.", tools_used=tools_used)

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        })
        for call in tool_calls:
            name = call.function.name
            tools_used.append(name)
            result = execute_tool(name, call.function.arguments, payload.user_id, jwt)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })

    return ChatResponse(
        reply=f"I stopped after {MAX_TOOL_ITERATIONS} tool calls. Please narrow your request.",
        tools_used=tools_used,
    )
