# Vaulto AI — ML / AI microservice

`vaulto-ai` is the Python/FastAPI service for the Vaulto finance platform. It provides
spending forecasts, anomaly detection, transaction categorization, and a Groq/Llama
chat agent that calls tools through the Node backend. It does **not** own authentication
or write to the database directly.

## Current status

| Capability | Status | Notes |
|---|---|---|
| Spending forecast | Working | Linear trend on cumulative daily spend; reports R² and a 90% residual interval |
| Anomaly detection | Working | IsolationForest; caller-supplied `contamination` controls the score threshold |
| Transaction categorization | Working | TF-IDF (word + char) + LogisticRegression; ~0.88 accuracy on the current dataset |
| Chat / tool calling | Working | Groq tool calling with 4 tools; requires `GROQ_API_KEY` and a running Node backend |
| RAG / pgvector | **Planned** | Not implemented; `app/services/rag_engine.py` is empty |
| Auth, transaction CRUD | Owned by `vaulto-backend` | LLM tool calls are validated here, authorized and written by the Node service |
| Docker / compose | Working | Builds models during image build |
| Tests | Working | `python -m pytest -q` |

### Architecture

```text
                          +-------------------+
                          |   Vaulto client   |
                          +---------+---------+
                                    |
              +---------------------+---------------------+
              |                                           |
      +-------v-------+                           +-------v-------+
      | vaulto-backend| <------ HTTP + JWT -------|   vaulto-ai   |
      | Node/Express  |        /transactions      | Python/FastAPI|
      | auth + CRUD   |                           | ML + LLM agent|
      +-------+-------+                           +-------+-------+
              |                                           |
      +-------v-------+                           +-------v-------+
      | Supabase DB   |                           | saved_models/ |
      | source of truth                           | .joblib files |
      +---------------+                           +---------------+
```

The LLM is treated as an untrusted request generator: its tool arguments are validated
with Pydantic, the user ID and JWT come from the incoming request, and every data
operation goes through `vaulto-backend` over HTTP. No direct database writes occur here.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
python train_models.py            # required once: models are gitignored
uvicorn app.main:app --reload
```

Fill in `GROQ_API_KEY` in `.env` to enable chat. `NODE_BACKEND_URL` must point at the
Node service, which is expected to expose `GET /transactions` and `POST /transactions`
and accept the forwarded `Authorization` header.

### Docker

```bash
docker compose up --build
```

The image installs pinned dependencies, runs `train_models.py`, and serves uvicorn on
port 8000. `docker-compose.yml` passes `.env` via `env_file`.

### Tests

```bash
python -m pytest -q
```

Groq and the Node backend are mocked in tests. The ML tests use the artifacts produced
by `python train_models.py`, so run training first on a fresh checkout.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/v1/ml/forecast` | Project month-end spending |
| POST | `/api/v1/ml/anomalies` | Flag unusual transactions |
| POST | `/api/v1/ml/categorize` | Categorize a description |
| POST | `/api/v1/chat` | Conversational tool calling (`Authorization: Bearer <jwt>`) |
| GET | `/api/v1/chat/status` | Chat service status |

Chat tools: `get_spending_summary`, `create_transaction`, `predict_monthly_spending`,
`detect_anomaly`. Tool calling is capped at 5 iterations per request.

## Configuration

| Variable | Purpose |
|---|---|
| `PORT`, `ENVIRONMENT` | Service metadata |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq credentials and model name |
| `NODE_BACKEND_URL` | Base URL of the Node service used by all data tools |
| `CORS_ORIGINS` | Comma-separated browser origins allowed by CORS |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Reserved for planned RAG; unused today |

## Training notes

`train_models.py` is the only training entry point. It reads the two CSVs in `data/`
using paths relative to this repository, canonicalizes category labels with an alias
map plus fuzzy matching, and clips amounts per category to the 1st–99th percentile.

The supplied dataset is synthetic and assigns categories largely at random relative to
descriptions (for example, "restaurant dinner" appears under many labels). To bootstrap
a usable model, unambiguous descriptions are relabelled from keyword rules before
training; the reported accuracy therefore measures imitation of those rules, not
real-world labeling quality. Replace this with genuinely labeled data when available.

## Layout

```text
vaulto-ai/
├── app/
│   ├── main.py                 # FastAPI app, CORS, routers
│   ├── config.py               # Environment settings
│   ├── api/v1/                 # ml.py and chat.py routers
│   ├── models/schemas.py       # Pydantic request/response schemas
│   ├── services/
│   │   ├── ml_engine.py        # Forecast, anomaly, categorizer inference
│   │   ├── llm_agent.py        # Groq tool calling + backend HTTP
│   │   └── rag_engine.py       # Planned (empty)
│   └── saved_models/           # Generated by train_models.py, gitignored
├── data/                       # Training CSVs
├── tests/                      # API tests
├── train_models.py             # Reproducible trainer
├── Dockerfile, docker-compose.yml
├── requirements.txt            # Pinned dependencies
└── .env.example                # Placeholders only
```
