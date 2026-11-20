import calendar
from datetime import date
from types import SimpleNamespace

import pytest

import app.services.llm_agent as agent
from app.config import settings
from app.models.schemas import TransactionItem
from app.services.ml_engine import predict_spending_forecast

SAMPLE_TRANSACTIONS = [
    {"amount": 100, "category": "Food", "date": "2026-09-01", "type": "expense"},
    {"amount": 150, "category": "Food", "date": "2026-09-02", "type": "expense"},
    {"amount": 120, "category": "Transport", "date": "2026-09-03", "type": "expense"},
    {"amount": 5000, "category": "Rent", "date": "2026-09-04", "type": "expense"},
    {"amount": 130, "category": "Food", "date": "2026-09-05", "type": "expense"},
]


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(name, arguments, call_id="call_1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=function)


@pytest.fixture()
def fake_backend(monkeypatch):
    calls = []

    def request(method, path, *, jwt, params=None, body=None):
        calls.append({"method": method, "path": path, "jwt": jwt, "params": params, "body": body})
        if method == "GET":
            return [{"id": "t1", "amount": 100, "category": "Food",
                     "date": "2026-09-01", "type": "expense"}]
        return {"id": "t2", "status": "created"}

    monkeypatch.setattr(agent, "backend_request", request)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key-not-real")
    return calls


@pytest.fixture()
def fake_groq(monkeypatch):
    """Two-step Groq conversation: one tool call, then a final answer."""
    seen = []

    class FakeGroq:
        def __init__(self, api_key):
            self.n = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            seen.append(kwargs["messages"])
            self.n += 1
            if self.n == 1:
                return completion(tool_calls=[tool_call("get_spending_summary", "{}")])
            return completion(content="You spent ₹100 on food.")

    monkeypatch.setattr(agent, "Groq", FakeGroq)
    return seen


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_forecast(client):
    response = client.post("/api/v1/ml/forecast", json={
        "user_id": "u1", "transactions": SAMPLE_TRANSACTIONS,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "u1"
    assert data["projected_month_end_spend"] >= 5500
    # 5 transactions is far below the minimum-data threshold for the ML model.
    assert data["model_r2"] is None
    assert data["confidence"] == "low"
    assert data["prediction_interval_90"] is None
    assert data["context"]["number_of_transactions"] == 5
    assert data["context"]["confidence"] == "low"
    assert len(data["insights"]) > 0


def test_forecast_without_history(client):
    response = client.post("/api/v1/ml/forecast", json={
        "user_id": "u1", "transactions": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model_r2"] is None
    assert data["prediction_interval_90"] is None
    assert data["context"]["projected_month_end_expenses"] == 0.0
    assert data["context"]["confidence"] == "low"


def test_anomalies(client):
    response = client.post("/api/v1/ml/anomalies", json={
        "user_id": "u1", "transactions": SAMPLE_TRANSACTIONS, "contamination": 0.3,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["anomalies_found"] == len(data["anomalies"])
    assert data["anomalies_found"] > 0
    assert "unusually" in data["anomalies"][0]["reason"]

    invalid = client.post("/api/v1/ml/anomalies", json={
        "user_id": "u1", "transactions": SAMPLE_TRANSACTIONS, "contamination": 0.9,
    })
    assert invalid.status_code == 422


def test_categorize(client):
    for description, expected in [
        ("dinner at restaurant", "Food"),
        ("uber to office", "Transport"),
        ("monthly rent", "Rent"),
    ]:
        response = client.post("/api/v1/ml/categorize", json={"description": description})
        assert response.status_code == 200
        assert response.json()["predicted_category"] == expected


def test_chat(client, fake_backend, fake_groq):
    response = client.post(
        "/api/v1/chat",
        json={"user_id": "u1", "message": "How much did I spend?"},
        headers={"Authorization": "Bearer test-jwt"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "reply": "You spent ₹100 on food.",
        "tools_used": ["get_spending_summary"],
    }
    assert fake_backend[0]["jwt"] == "Bearer test-jwt"
    assert fake_backend[0]["params"] == {"user_id": "u1"}


def test_chat_missing_key(client, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "your_key_here")
    response = client.post("/api/v1/chat", json={"user_id": "u1", "message": "hi"})
    assert response.status_code == 200
    assert "not configured" in response.json()["reply"]


def test_chat_invalid_tool_args(client, monkeypatch, fake_backend):
    class BadArgsGroq:
        def __init__(self, api_key):
            self.n = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.n += 1
            if self.n == 1:
                args = '{"amount": -5, "category": "Food", "date": "not-a-date"}'
                return completion(tool_calls=[tool_call("create_transaction", args)])
            return completion(content="I could not record that.")

    monkeypatch.setattr(agent, "Groq", BadArgsGroq)
    response = client.post("/api/v1/chat", json={"user_id": "u1", "message": "add -5"})
    assert response.status_code == 200
    assert response.json()["tools_used"] == ["create_transaction"]
    assert fake_backend == []  # invalid arguments must never reach the backend


@pytest.mark.parametrize("category", agent.ALLOWED_CATEGORIES)
def test_create_transaction_args_accept_allowed_categories(category):
    args = agent.CreateTransactionArgs(amount=100, category=category, date="2026-09-24")
    assert args.category == category


def test_chat_normalizes_unknown_category_to_other(client, monkeypatch, fake_backend):
    class SpaceshipGroq:
        def __init__(self, api_key):
            self.n = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.n += 1
            if self.n == 1:
                args = '{"amount": 100, "category": "Spaceship", "date": "2026-09-24"}'
                return completion(tool_calls=[tool_call("create_transaction", args)])
            return completion(content="Recorded.")

    monkeypatch.setattr(agent, "Groq", SpaceshipGroq)
    response = client.post("/api/v1/chat", json={"user_id": "u1", "message": "add spaceship expense"})
    assert response.status_code == 200
    assert response.json()["tools_used"] == ["create_transaction"]
    assert len(fake_backend) == 1
    assert fake_backend[0]["method"] == "POST"
    assert fake_backend[0]["path"] == "/transactions"
    assert fake_backend[0]["body"]["category"] == "Other"
    assert fake_backend[0]["body"]["amount"] == 100
    assert fake_backend[0]["body"]["date"] == "2026-09-24"


def test_create_transaction_tool_description_lists_categories():
    tool = next(t for t in agent.TOOL_DEFINITIONS if t["function"]["name"] == "create_transaction")
    description = tool["function"]["description"]
    for category in agent.ALLOWED_CATEGORIES:
        assert category in description
    assert "Use \"Other\" when nothing fits" in description
    properties = tool["function"]["parameters"]["properties"]["category"]
    assert sorted(properties["enum"]) == sorted(agent.ALLOWED_CATEGORIES)
    assert "Use \"Other\" when nothing fits" in properties["description"]


def test_legacy_category_normalized_for_ml_endpoints(client):
    transactions = SAMPLE_TRANSACTIONS + [
        {"amount": 200, "category": "Groceries", "date": "2026-09-06", "type": "expense"},
    ]
    forecast = client.post("/api/v1/ml/forecast", json={"user_id": "u1", "transactions": transactions})
    assert forecast.status_code == 200

    anomaly = client.post("/api/v1/ml/anomalies", json={
        "user_id": "u1", "transactions": transactions, "contamination": 0.2,
    })
    assert anomaly.status_code == 200


def test_chat_backend_auth_error(client, monkeypatch, fake_groq):
    def denied(*args, **kwargs):
        raise agent.BackendError("The Vaulto backend rejected your session (401).")

    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(agent, "backend_request", denied)
    response = client.post(
        "/api/v1/chat", json={"user_id": "u1", "message": "summary"},
        headers={"Authorization": "Bearer expired"},
    )
    assert response.status_code == 200
    assert response.json()["tools_used"] == ["get_spending_summary"]
    tool_messages = [m for m in fake_groq[-1] if m["role"] == "tool"]
    assert "401" in tool_messages[0]["content"]


def test_chat_groq_failure(client, monkeypatch):
    class BrokenGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            raise RuntimeError("provider offline")

    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(agent, "Groq", BrokenGroq)
    response = client.post("/api/v1/chat", json={"user_id": "u1", "message": "hi"})
    assert response.status_code == 200
    assert "AI service error" in response.json()["reply"]


def test_chat_tool_iteration_cap(client, monkeypatch, fake_backend):
    class LoopingGroq:
        def __init__(self, api_key):
            self.n = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.n += 1
            return completion(tool_calls=[tool_call(
                "get_spending_summary", "{}", call_id=f"call_{self.n}"
            )])

    monkeypatch.setattr(agent, "Groq", LoopingGroq)
    response = client.post("/api/v1/chat", json={"user_id": "u1", "message": "loop"})
    assert response.status_code == 200
    assert len(response.json()["tools_used"]) == 5
    assert "5 tool calls" in response.json()["reply"]


def _month_day(months_ago, day):
    today = date.today()
    index = today.year * 12 + today.month - 1 - months_ago
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}-{day:02d}"


def test_sparse_forecast_excludes_one_off_and_has_no_interval():
    today = date.today()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    elapsed = min(today.day, days_in_month)
    transactions = [
        TransactionItem(amount=475, category="Food", date=_month_day(0, (i % elapsed) + 1), type="expense")
        for i in range(6)
    ]
    transactions += [
        TransactionItem(amount=10000, category="Education", date=_month_day(0, min(2, elapsed)), type="expense"),
        TransactionItem(amount=20000, category="Salary", date=_month_day(0, 1), type="income"),
    ]

    result = predict_spending_forecast(transactions)
    ctx = result["context"]
    expected = (2850 / elapsed) * days_in_month + 10000

    assert result["forecast_method"] == "rule_based"
    assert ctx["confidence"] == "low"
    assert result["prediction_interval_90"] is None
    assert result["model_r2"] is None
    assert ctx["month_income"] == 20000
    assert ctx["month_expenses_so_far"] == 12850
    assert ctx["projected_month_end_expenses"] == pytest.approx(expected, abs=1)
    assert ctx["top_category"] == "Education"
    assert ctx["top_category_amount"] == 10000
    assert len(ctx["excluded_outliers"]) == 1
    assert ctx["excluded_outliers"][0]["amount"] == 10000
    assert "rough" in " ".join(result["insights"])


def test_ml_model_not_used_without_two_full_months():
    transactions = [
        TransactionItem(amount=100, category="Food", date=_month_day(0, (i % 28) + 1), type="expense")
        for i in range(30)
    ]
    result = predict_spending_forecast(transactions)
    assert result["forecast_method"] == "rule_based"
    assert result["context"]["number_of_transactions"] == 30
    assert result["context"]["months_of_history"] == 0
    assert result["context"]["confidence"] == "low"
    assert result["prediction_interval_90"] is None


def test_ml_model_used_when_data_minimums_are_met():
    transactions = [
        TransactionItem(
            amount=100 + i,
            category="Food",
            date=_month_day(2 - (i // 10), (i % 10) + 1),
            type="expense",
        )
        for i in range(30)
    ]
    result = predict_spending_forecast(transactions)
    assert result["forecast_method"] == "ml"
    assert result["context"]["months_of_history"] == 2
    assert result["context"]["confidence"] in ("medium", "high")
    low, high = result["prediction_interval_90"]
    assert low <= high


def test_sanitize_reply_and_system_prompt():
    assert agent.sanitize_reply("**≈£20,300**") == "≈₹20,300"
    assert agent.sanitize_reply("`# Heading`\n__₹1,200__") == "Heading\n₹1,200"
    assert "Indian Rupees" in agent.SYSTEM_PROMPT
    assert "plain text only" in agent.SYSTEM_PROMPT
    assert "answer the user's question directly" in agent.SYSTEM_PROMPT


def test_chat_history_order_and_backward_compatibility(client, monkeypatch, fake_backend):
    seen = []

    class SimpleGroq:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            seen.append(kwargs["messages"])
            return completion(content="Here is the answer.")

    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(agent, "Groq", SimpleGroq)

    response = client.post("/api/v1/chat", json={
        "user_id": "u1",
        "message": "and what about transport?",
        "history": [
            {"role": "user", "content": "Will I overspend this month?"},
            {"role": "assistant", "content": "No, you are not on track."},
        ],
    })
    assert response.status_code == 200
    roles = [message["role"] for message in seen[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert seen[0][1]["content"] == "Will I overspend this month?"
    assert seen[0][-1]["content"] == "and what about transport?"

    seen.clear()
    legacy = client.post("/api/v1/chat", json={"user_id": "u1", "message": "hi"})
    assert legacy.status_code == 200
    assert [message["role"] for message in seen[0]] == ["system", "user"]
