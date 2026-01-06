from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.dimensions = 1024

    def embed_batch(self, texts: list[str], dimension: int = None) -> list[list[float]]:
        dim = dimension if dimension is not None else self.dimensions
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=4,
            show_progress_bar=False
        )
        return [v[:dim].tolist() for v in vectors]

    def embed_batch_with_ids(self, texts_with_ids: list[dict[str, str]], dimension: int = None) -> list[dict[str, object]]:
        texts = [item["text"] for item in texts_with_ids]
        vectors = self.embed_batch(texts, dimension)
        return [
            {"id": item["id"], "embedding": vector}
            for item, vector in zip(texts_with_ids, vectors)
        ]

    def embed_tokens(self, text: str, dimension: int = None) -> dict[str, list]:
        dim = dimension if dimension is not None else self.dimensions
        tokens = self.model.tokenizer.tokenize(text)
        embeddings = self.model.encode(tokens, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings[:, :dim]
        return {"tokens": tokens, "embeddings": embeddings.tolist()}
