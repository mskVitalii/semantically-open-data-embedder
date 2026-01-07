FROM python:3.12-slim

# Build argument for model name
ARG MODEL_NAME=BAAI/bge-m3

# Environment variables
ENV MODEL_NAME=${MODEL_NAME}
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Pre-download the model during build
RUN python -c "from sentence_transformers import SentenceTransformer; \
model_name = '${MODEL_NAME}'; \
trust_remote_code = 'jina' in model_name.lower(); \
SentenceTransformer(model_name, trust_remote_code=trust_remote_code)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
