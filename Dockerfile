FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY training/ training/
COPY serving/ serving/
COPY shared/ shared/

# Training stage
FROM base AS training
CMD ["python", "-m", "training.train_dlrm", "--mode", "local"]

# Serving stage
FROM base AS serving
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
