FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE=/app/models_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download the local embedding model into FASTEMBED_CACHE at build time.
RUN python -c "import os; from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=os.environ['FASTEMBED_CACHE']).query_embed('warmup'))"

COPY app ./app
# Includes data/knowledge/ (empty markdown files until you fill them in).
COPY data ./data
COPY train_models.py .
RUN python train_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
