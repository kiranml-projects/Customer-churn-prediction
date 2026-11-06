FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# curl is only used by the HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first -> this layer is cached until requirements change
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serving/ ./serving/
COPY models/churn_model.pkl models/scaler.pkl ./models/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.model_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
