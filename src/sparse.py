"""Sparse embedding via hashing trick (model-independent)."""

import re
from collections import Counter
from typing import Literal

import mmh3

_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[._/\-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*",
    re.UNICODE,
)

DEFAULT_SPARSE_DIM = 1 << 20  # 1_048_576


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def sparse_hash_vector(
    text: str,
    dim: int = DEFAULT_SPARSE_DIM,
    mode: Literal["binary", "tf"] = "tf",
) -> dict:
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    if mode == "binary":
        indices = sorted({mmh3.hash(tok, signed=False) % dim for tok in tokens})
        return {"indices": indices, "values": [1.0] * len(indices)}

    # tf mode
    counts: dict[int, float] = {}
    for tok in tokens:
        idx = mmh3.hash(tok, signed=False) % dim
        counts[idx] = counts.get(idx, 0.0) + 1.0
    indices = sorted(counts.keys())
    values = [counts[i] for i in indices]
    return {"indices": indices, "values": values}


def sparse_hash_vector_batch(
    texts: list[str],
    dim: int = DEFAULT_SPARSE_DIM,
    mode: Literal["binary", "tf"] = "tf",
) -> list[dict]:
    return [sparse_hash_vector(text, dim=dim, mode=mode) for text in texts]
