# Serving image: FastAPI + onnxruntime + numpy (no PyTorch). Train first so artifacts/ exists.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 ARTIFACTS=/app/artifacts THREADS=2
RUN useradd --uid 1000 --create-home app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[api]"
COPY artifacts/ artifacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
