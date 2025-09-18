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


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
