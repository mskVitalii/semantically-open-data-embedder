from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import asyncio

from .model import Embedder

app = FastAPI()
embedder = Embedder()

class EmbedRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    vectors = await asyncio.to_thread(embedder.embed_batch, request.texts)
    return {"embeddings": vectors}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
