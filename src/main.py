import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from .model import Embedder

app = FastAPI()
embedder = Embedder()

class EmbedRequest(BaseModel):
    texts: list[str]
    dimension: int = 1024

class EmbedWithIdsRequest(BaseModel):
    texts: list[dict[str, str]]
    dimension: int = 1024



@app.post("/embed")
async def embed(request: EmbedRequest):
    vectors = await asyncio.to_thread(embedder.embed_batch, request.texts, request.dimension)
    return {"embeddings": vectors, "dimension": request.dimension}


@app.post("/embed_with_ids")
async def embed_with_ids(request: EmbedWithIdsRequest):
    vectors = await asyncio.to_thread(embedder.embed_batch_with_ids, request.texts, request.dimension)
    return {"embeddings": vectors, "dimension": request.dimension}


class EmbedTokensRequest(BaseModel):
    text: str
    dimension: int = 1024

@app.post("/embed_tokens_tsv")
async def embed_tokens_tsv(request: EmbedTokensRequest):
    result = await asyncio.to_thread(embedder.embed_tokens, request.text, request.dimension)
    tokens = result["tokens"]
    embeddings = np.array(result["embeddings"])

    # build TSV in memory
    vectors_tsv = "\n".join(["\t".join(f"{x:.6f}" for x in row) for row in embeddings])
    metadata_tsv = "\n".join(tokens)

    # return both files as plain text in JSON
    return {
        "vectors_tsv": vectors_tsv,
        "metadata_tsv": metadata_tsv,
        "dimension": request.dimension
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
