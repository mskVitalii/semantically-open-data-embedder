from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.dimensions = 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=4,
            show_progress_bar=False
        )
        return [v[:self.dimensions].tolist() for v in vectors]

    def embed_batch_with_ids(self, texts_with_ids: list[dict[str, str]]) -> list[dict[str, object]]:
        texts = [item["text"] for item in texts_with_ids]
        vectors = self.embed_batch(texts)
        return [
            {"id": item["id"], "embedding": vector}
            for item, vector in zip(texts_with_ids, vectors)
        ]
