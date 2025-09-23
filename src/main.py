import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from .model import Embedder

app = FastAPI()
embedder = Embedder()

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedWithIdsRequest(BaseModel):
    texts: list[dict[str, str]]



@app.post("/embed")
async def embed(request: EmbedRequest):
    vectors = await asyncio.to_thread(embedder.embed_batch, request.texts)
    return {"embeddings": vectors}


@app.post("/embed_with_ids")
async def embed_with_ids(request: EmbedWithIdsRequest):
    vectors = await asyncio.to_thread(embedder.embed_batch_with_ids, request.texts)
    return {"embeddings": vectors}


class EmbedRequest(BaseModel):
    text: str

@app.post("/embed_tokens_tsv")
async def embed_tokens_tsv(request: EmbedRequest):
    result = await asyncio.to_thread(embedder.embed_tokens, request.text)
    tokens = result["tokens"]
    embeddings = np.array(result["embeddings"])

    # формируем TSV в памяти
    vectors_tsv = "\n".join(["\t".join(f"{x:.6f}" for x in row) for row in embeddings])
    metadata_tsv = "\n".join(tokens)

    # возвращаем оба файла как plain text в JSON
    return {
        "vectors_tsv": vectors_tsv,
        "metadata_tsv": metadata_tsv
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
