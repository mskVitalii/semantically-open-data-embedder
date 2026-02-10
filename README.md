# Open Data Embedder

Embedding service for Open Government Data datasets. Supports dense, sparse (hashing trick), and hybrid vectorization.

## Models

| Model | Dimension | Port | Docker image |
|-------|-----------|------|--------------|
| BAAI/bge-m3 | 1024 | 8080 | `mskkote/embedder-baai-bge-m3` |
| intfloat/multilingual-e5-base | 768 | 8081 | `mskkote/embedder-intfloat-multilingual-e5-base` |
| jinaai/jina-embeddings-v3 | 1024 | 8082 | `mskkote/embedder-jinaai-jina-embeddings-v3` |
| sentence-transformers/LaBSE | 768 | 8083 | `mskkote/embedder-sentence-transformers-labse` |

Each container serves one dense model. Sparse vectorization is model-independent (hashing trick) and available on every container.

## Quick Start

```bash
# Local dev (default model: BAAI/bge-m3)
make dev

# Local dev with specific model
make dev MODEL=intfloat/multilingual-e5-base

# Docker
make build MODEL=BAAI/bge-m3
make run
```

## API

### Modes

All embedding endpoints accept a `mode` parameter:

- **`dense`** (default) — dense embedding from the model
- **`sparse`** — sparse vector via hashing trick (no model call, sub-ms)
- **`hybrid`** — both dense + sparse in parallel

### `POST /embed`

Embed a list of texts.

**Request:**

```json
{
  "texts": ["Einwohner 2022 Dresden", "Population 2021 Leipzig"],
  "dimension": 1024,
  "mode": "hybrid",
  "sparse_dim": 1048576,
  "sparse_mode": "tf"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `texts` | `list[str]` | required | Texts to embed |
| `dimension` | `int` | `1024` | Dense vector dimension (truncation). Ignored when `mode="sparse"` |
| `mode` | `"dense" \| "sparse" \| "hybrid"` | `"dense"` | Vectorization mode |
| `sparse_dim` | `int` | `1048576` | Hash space size for sparse. Ignored when `mode="dense"` |
| `sparse_mode` | `"binary" \| "tf"` | `"tf"` | Sparse weights: `binary` = 1.0, `tf` = term frequency. Ignored when `mode="dense"` |

**Response — `mode="dense"`:**

```json
{
  "embeddings": [[0.023, -0.089, ...], [0.015, 0.072, ...]],
  "dimension": 1024
}
```

**Response — `mode="sparse"`:**

```json
{
  "embeddings": [
    {"indices": [4821, 129887, 483201], "values": [1.0, 1.0, 2.0]},
    {"indices": [55012, 129887], "values": [1.0, 1.0]}
  ],
  "sparse_dim": 1048576
}
```

**Response — `mode="hybrid"`:**

```json
{
  "embeddings": [
    {
      "dense": [0.023, -0.089, ...],
      "sparse": {"indices": [4821, 129887, 483201], "values": [1.0, 1.0, 2.0]}
    }
  ],
  "dimension": 1024,
  "sparse_dim": 1048576
}
```

### `POST /embed_with_ids`

Same as `/embed` but preserves document IDs.

**Request:**

```json
{
  "texts": [
    {"id": "dataset_42", "text": "Einwohner 2022 Dresden"},
    {"id": "dataset_99", "text": "Population 2021 Leipzig"}
  ],
  "mode": "hybrid"
}
```

**Response — `mode="hybrid"`:**

```json
{
  "embeddings": [
    {
      "id": "dataset_42",
      "dense": [0.023, -0.089, ...],
      "sparse": {"indices": [4821, 129887], "values": [1.0, 1.0]}
    }
  ],
  "dimension": 1024,
  "sparse_dim": 1048576
}
```

For `mode="dense"` response is identical to the current format (backward compatible).

### `POST /embed_tokens_tsv`

Tokenize text and embed each token. Returns TSV. No `mode` parameter — dense only.

```json
{
  "text": "some text to tokenize",
  "dimension": 1024
}
```

### `GET /healthz`

```json
{"status": "ok"}
```

## Sparse Vectorization

Sparse vectors are built via the **hashing trick** — a model-independent method that maps text tokens to a fixed-size sparse vector space using MurmurHash3.

**How it works:**

1. Tokenize text with a regex that preserves compound tokens (`MD_3.1e`, `479/520`, `2022`)
2. Hash each token: `index = mmh3.hash(token, unsigned) % sparse_dim`
3. Set value at that index: `1.0` (binary) or term frequency count (tf)
4. Store only non-zero entries as `{indices, values}`

The same `sparse_dim` and `sparse_mode` must be used at indexing and query time.

## Qdrant Integration

Sparse response maps directly to `qdrant_client.http.models.SparseVector`:

```python
from qdrant_client.http import models

sparse = embedder_response["sparse"]
qdrant_sparse = models.SparseVector(
    indices=sparse["indices"],
    values=sparse["values"],
)
```

Collection with named dense + sparse vectors:

```python
client.recreate_collection(
    collection_name="datasets_metadata",
    vectors_config={
        "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "lexical": models.SparseVectorParams(),
    },
)
```

Hybrid search with RRF fusion:

```python
client.query_points(
    collection_name=collection,
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=100),
        models.Prefetch(query=qdrant_sparse, using="lexical", limit=100),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=25,
)
```

## Development

```bash
make dev                # run locally with hot reload
make test               # test dense
make test-sparse        # test sparse
make test-hybrid        # test hybrid
make health             # health check
make build-all          # build Docker images for all 4 models
make push-all           # push all images
```
