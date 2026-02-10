import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import asyncio

from .model import Embedder
from .sparse import sparse_hash_vector_batch, DEFAULT_SPARSE_DIM

app = FastAPI()
embedder = Embedder()


class EmbedRequest(BaseModel):
    texts: list[str]
    dimension: int = 1024
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    sparse_dim: int = DEFAULT_SPARSE_DIM
    sparse_mode: Literal["binary", "tf"] = "tf"


class EmbedWithIdsRequest(BaseModel):
    texts: list[dict[str, str]]
    dimension: int = 1024
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    sparse_dim: int = DEFAULT_SPARSE_DIM
    sparse_mode: Literal["binary", "tf"] = "tf"


@app.post("/embed")
async def embed(request: EmbedRequest):
    if request.mode == "dense":
        vectors = await asyncio.to_thread(embedder.embed_batch, request.texts, request.dimension)
        return {"embeddings": vectors, "dimension": request.dimension}

    if request.mode == "sparse":
        sparse_vectors = sparse_hash_vector_batch(
            request.texts, dim=request.sparse_dim, mode=request.sparse_mode
        )
        return {"embeddings": sparse_vectors, "sparse_dim": request.sparse_dim}

    # hybrid
    dense_future = asyncio.to_thread(embedder.embed_batch, request.texts, request.dimension)
    sparse_vectors = sparse_hash_vector_batch(
        request.texts, dim=request.sparse_dim, mode=request.sparse_mode
    )
    dense_vectors = await dense_future

    embeddings = [
        {"dense": d, "sparse": s}
        for d, s in zip(dense_vectors, sparse_vectors)
    ]
    return {
        "embeddings": embeddings,
        "dimension": request.dimension,
        "sparse_dim": request.sparse_dim,
    }


@app.post("/embed_with_ids")
async def embed_with_ids(request: EmbedWithIdsRequest):
    if request.mode == "dense":
        vectors = await asyncio.to_thread(embedder.embed_batch_with_ids, request.texts, request.dimension)
        return {"embeddings": vectors, "dimension": request.dimension}

    texts = [item["text"] for item in request.texts]

    if request.mode == "sparse":
        sparse_vectors = sparse_hash_vector_batch(
            texts, dim=request.sparse_dim, mode=request.sparse_mode
        )
        embeddings = [
            {"id": item["id"], "sparse": sv}
            for item, sv in zip(request.texts, sparse_vectors)
        ]
        return {"embeddings": embeddings, "sparse_dim": request.sparse_dim}

    # hybrid
    dense_future = asyncio.to_thread(embedder.embed_batch_with_ids, request.texts, request.dimension)
    sparse_vectors = sparse_hash_vector_batch(
        texts, dim=request.sparse_dim, mode=request.sparse_mode
    )
    dense_results = await dense_future

    embeddings = [
        {"id": dr["id"], "dense": dr["embedding"], "sparse": sv}
        for dr, sv in zip(dense_results, sparse_vectors)
    ]
    return {
        "embeddings": embeddings,
        "dimension": request.dimension,
        "sparse_dim": request.sparse_dim,
    }


class EmbedTokensRequest(BaseModel):
    text: str
    dimension: int = 1024


def pca_3d(embeddings: np.ndarray) -> list[list[float]]:
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:3].T
    return projected.tolist()


@app.post("/embed_tokens")
async def embed_tokens(request: EmbedTokensRequest):
    result = await asyncio.to_thread(embedder.embed_tokens, request.text, request.dimension)
    embeddings = np.array(result["embeddings"])
    points_3d = pca_3d(embeddings)
    return {
        "tokens": result["tokens"],
        "points_3d": points_3d,
        "dimension": request.dimension,
    }


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
