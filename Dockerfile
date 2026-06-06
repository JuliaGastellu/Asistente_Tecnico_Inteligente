# ───── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/install/huggingface_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --prefix=/install -r /tmp/requirements-runtime.txt

# Pre-download HuggingFace embedding model so runtime has no network dependency
RUN PYTHONPATH=/install/lib/python3.10/site-packages \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ───── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/install/huggingface_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home --shell /bin/false appuser

# Copy installed packages and HuggingFace model cache from builder
COPY --from=builder /install /usr/local
COPY --from=builder /install/huggingface_cache /install/huggingface_cache

WORKDIR /app

# Copy application source and RAG corpus (layer-cache friendly: static first)
COPY --chown=appuser:appuser data/documents/ ./data/documents/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser setup_data.py ./

# Ensure runtime-writable directories exist with correct ownership
# NOTE: chroma_db_advanced matches consigna spec; actual config.py path is data/chroma_db
RUN mkdir -p /app/chroma_db_advanced /app/reports /app/data/chroma_db /app/logs && \
    chown -R appuser:appuser /app/chroma_db_advanced /app/reports /app/data /app/logs

USER appuser

EXPOSE 8000

# NOTE: consigna spec says "main:app"; adapted to "src.api.main:app" to match
# the actual project structure (main.py lives at src/api/main.py, WORKDIR=/app)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
