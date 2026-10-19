# Hugging Face Space (Docker SDK): onnxruntime + numpy + Gradio, no PyTorch — small image, fast build.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 THREADS=2
WORKDIR /home/user/app
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
