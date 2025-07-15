FROM python:3.12-slim

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
