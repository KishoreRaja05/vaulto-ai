# Vaulto AI — ML / AI microservice

`vaulto-ai` is the Python/FastAPI service for the Vaulto finance platform. It provides
spending forecasts, anomaly detection, transaction categorization, and a Groq-powered
chat agent that calls tools through the Node backend. It does **not** own authentication
or write to the database directly.

## Current status

| Capability | Status | Notes |
|---|---|---|
| Spending forecast | Working | Linear trend on cumulative daily spend; reports R² and a 90% residual interval |
| Anomaly detection | Working | IsolationForest; caller-supplied `contamination` controls the score threshold |
| Transaction categorization | Working | TF-IDF (word + char) + LogisticRegression; ~0.88 accuracy on the current dataset |
| Chat / tool calling | Working | Groq tool calling with 5 tools; requires `GROQ_API_KEY` and a running Node backend for data tools |
| RAG | Working (knowledge base being written) | Local `fastembed` `BAAI/bge-small-en-v1.5` over `data/knowledge/*.md`; in-memory index, no pgvector yet |
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

### Knowledge base (RAG)

`app/services/rag_engine.py` embeds the markdown notes in `data/knowledge/` locally with
`fastembed` (`BAAI/bge-small-en-v1.5`). Blank files are skipped. Each file is split on
`#`/`##` headings into sections; the heading stays with its chunk, and sections longer
than roughly 200 words are split on blank lines only, so a chunk never starts inside a
sentence or list item or crosses a heading. Chunks under 15 words are dropped. Vectors
are L2-normalized and searched with dot product. Each chunk is embedded as
`title + chunk`, while the plain chunk alone is returned to the LLM. The model and index
load lazily on the first search. The index lives in memory only; pgvector is a possible
future step if the knowledge base grows or needs to be shared between workers. The
`search_financial_knowledge` chat tool is for general finance concepts and how Vaulto
features work, never for the user's own transactions.

Results must clear a score threshold, `0.65` by default and configurable with
`RAG_MIN_SCORE`. `python check_rag.py` prints the top three chunks for 21 queries and a
pass/fail summary. A threshold cannot fully separate off-topic questions that happen to
share words with finance topics, so the system prompt also tells the model to say when
it has no information rather than answer from general knowledge.

**Knowledge file format.** Every `data/knowledge/*.md` file must start with `# Title`
followed by one plain definition sentence written the way a user would ask. Use plain
paragraphs and short bullet lists only: no `## Metadata` blocks (Target Keywords, RAG
Summary or Vector Search Key), no `---` separators, no LaTeX, no tables, no numbering in
headings, and a blank line between every heading, list and paragraph. Facts and numbers
should stay intact; do not describe features that are not in the code.

### Knowledge base fallback

When a finance question is not covered by `data/knowledge/`, the assistant follows four
behaviours:

1. Covered question: answer only from the retrieved text, name the topic it came from and
   add no facts that are not in the text.
2. Uncovered finance question (for example "what is a mutual fund?"): begin with "I don't
   have this in my Vaulto knowledge base.", then give a short general explanation of at
   most four sentences labelled "General information:". No tax figures or slabs,
   strategies, products, fund recommendations, return estimates or buy/sell advice.
3. Advice or recommendation request (for example "should I buy gold?"): recommend
   nothing. Say Vaulto cannot give investment advice, explain the concept at most.
4. Off-topic question (cooking, sports, weather, tech support): reply "I can only help
   with personal finance and Vaulto." Do not answer it and do not add a disclaimer.

The `search_financial_knowledge` tool returns `{"found": false, "instruction": ...}` on a
miss. `run_chat` then guarantees this disclaimer on general-information answers, unless
the model already wrote "not financial advice":

> This is general information, not financial advice. For decisions about tax, investments or loans, consult a qualified professional.

No disclaimer is added for off-topic refusals or a plain knowledge-base answer that names
no sensitive topic. Coverage gaps are logged at INFO with the query only (never a JWT,
user ID or personal data); hits log the top score and source at DEBUG. The general
information is still model-generated and can be wrong, which is why it is capped,
labelled and followed by a disclaimer rather than presented as knowledge-base content.

For personal money situations ("I need 1 lakh urgently", "am I saving enough?"), the prompt
requires data tools first so the answer uses real numbers, then the knowledge tool for
concepts. Urgent needs must never be answered with a loan, credit line or card: use the
retrieved emergency-fund and spending guidance, and suggest a trusted person, bank or
professional if there are no savings. Only options in the retrieved text may be named;
anything else belongs under "General information:". If such a message ends without a
data-tool call, one INFO line is logged with no JWT, user ID or message content.

The disclaimer is also appended in code when the final answer mentions loan, credit,
borrow, mutual fund, invest, stock, FD, fixed deposit or tax (the editable
`DISCLAIMER_KEYWORDS`), even if the words came from the knowledge base. Prompt rules reduce
but cannot guarantee the model never mentions borrowing, hence the code-level disclaimer.

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

The image installs pinned dependencies, pre-downloads the `fastembed` model into
`FASTEMBED_CACHE`, copies `data/knowledge/`, runs `train_models.py`, and serves uvicorn
on port 8000. `docker-compose.yml` passes `.env` via `env_file`.

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
`detect_anomaly`, `search_financial_knowledge`. Tool calling is capped at 5 iterations
per request. To the hosted model, `get_spending_summary` sends transaction count, total
expense and per-category totals; `create_transaction` sends `created`, `category`,
`category_created` and the created transaction's amount, type, category, date and
description (no id, userId or createdAt); `predict_monthly_spending` sends the forecast
totals, scores, confidence, insights and context including outlier amount, category,
date and reason but no transaction IDs; `detect_anomaly` sends the count and each
anomaly's amount, category, date, score and reason but no transaction IDs;
`search_financial_knowledge` sends knowledge chunks (source, text, score) or a miss
instruction.

## Configuration

| Variable | Purpose |
|---|---|
| `PORT`, `ENVIRONMENT` | Service metadata |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq credentials and model name (default `openai/gpt-oss-120b`) |
| `NODE_BACKEND_URL` | Base URL of the Node service used by all data tools |
| `FASTEMBED_CACHE` | Local model cache dir (default `/app/models_cache`); pre-downloaded in the Docker image |
| `RAG_MIN_SCORE` | Minimum cosine score for `search_financial_knowledge` results (default `0.65`) |
| `CORS_ORIGINS` | Comma-separated browser origins allowed by CORS |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Reserved for future RAG persistence; unused today (in-memory index) |

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
│   │   ├── llm_agent.py        # Groq tool calling + backend HTTP + knowledge tool
│   │   └── rag_engine.py       # Local fastembed RAG (lazy, in-memory index)
│   └── saved_models/           # Generated by train_models.py, gitignored
├── data/
│   ├── *.csv                   # Training CSVs
│   └── knowledge/              # Empty markdown files; add your own content
├── tests/                      # API tests
├── train_models.py             # Reproducible trainer
├── Dockerfile, docker-compose.yml
├── requirements.txt            # Pinned dependencies
└── .env.example                # Placeholders only
```
