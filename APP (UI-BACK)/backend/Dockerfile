FROM python:3.11-slim


WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_CACHE_HOME=/app/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN python -m pip install --no-cache-dir torch==2.6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "paddlex[ocr]"

COPY pipeline.py .
COPY models ./models
COPY bilstm_for_tables ./bilstm_for_tables

EXPOSE 8000 9102

CMD ["uvicorn", "pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
