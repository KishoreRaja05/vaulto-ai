from types import SimpleNamespace

import pytest

import app.services.llm_agent as agent
from app.config import settings

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
    assert data["model_r2"] is not None
    low, high = data["prediction_interval_90"]
    assert low <= high
    assert len(data["insights"]) > 0


def test_forecast_without_history(client):
    response = client.post("/api/v1/ml/forecast", json={
        "user_id": "u1", "transactions": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model_r2"] is None
    assert data["prediction_interval_90"] == [0.0, 0.0]


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
