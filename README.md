# 🤖 Vaulto AI — Microservice Engine

**`vaulto-ai`** is the dedicated Python AI/ML microservice for the **Vaulto AI Finance Platform**. Built with **FastAPI**, **Scikit-learn**, and **Supabase `pgvector`**, it handles heavy numerical modeling, predictive financial forecasting, spending anomaly detection, Retrieval-Augmented Generation (RAG), and LLM intent orchestration [1, 2].

---

## 🏛️ Polyglot Architecture & Role

Vaulto uses a **polyglot microservices architecture** where each service is built with the language best suited for its system responsibilities [3]:

                            Vaulto AI Architecture
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        ▼                                                           ▼
vaulto-backend (Node.js)                                    vaulto-ai (Python) (Application Authority & Auth)                           (AI / ML Service Engine) │                                                           │ ▼                                                           ▼ Supabase PostgreSQL                                    LLM + ML + RAG (pgvector)

* **`vaulto-backend` (Node.js + Express):** Governs user authentication, transaction CRUD operations, authorization, and database security [4, 5].
* **`vaulto-ai` (Python + FastAPI):** Handles machine learning modeling, vector embeddings, semantic RAG search, and LLM tool calling orchestration [1, 4].
* **Security Gate Principle:** The LLM does **not** have direct write access to the production database [6, 7]. Natural language actions (e.g., *"Add ₹500 for dinner"*) are parsed into structured tool calls by `vaulto-ai` and routed back to `vaulto-backend` for JWT authentication and permission validation before writing to PostgreSQL [5, 7, 8].

---

## 🚀 Key Features

1. **📈 Spending Forecast (ML):** Trains regression models (Linear Regression, Random Forest) on historical user transactions to project month-end expense totals [9, 10].
2. **🚨 Anomaly Detection (ML):** Deploys an **Isolation Forest** model to detect unusual transaction amounts relative to category averages [11, 12].
3. **📚 Financial RAG Knowledge Base:** Performs semantic similarity searches over personal finance concepts using `pgvector` in Supabase [13, 14].
4. **💬 Conversational AI Tool Calling:** Uses LLMs (via Groq / Llama) to parse user intent and execute actions (*create*, *query*, *update*, *delete* transactions) via backend APIs [15, 16].

---

## 📂 Directory Layout

```text
vaulto-ai/
├── app/
│   ├── __init__.py           # Package marker
│   ├── main.py               # FastAPI entry point & CORS configuration
│   ├── config.py             # Environment variable management (Pydantic BaseSettings)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── chat.py       # Endpoint: /api/v1/chat (LLM orchestration & tools)
│   │       └── ml.py         # Endpoint: /api/v1/ml (Forecasts & Anomaly detection)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_engine.py      # Scikit-learn models (Regression, Isolation Forest)
│   │   ├── rag_engine.py     # Supabase pgvector search integration
│   │   └── llm_agent.py      # Groq / Llama tool caller & intent parser
│   └── models/
│       ├── __init__.py
│       └── schemas.py        # Pydantic request/response validation schemas
├── .env.example              # Environment variables template
├── .gitignore                # Excludes venv/, .env, __pycache__
├── Dockerfile                # Multi-stage Python 3.12 deployment image
└── requirements.txt          # Package dependencies