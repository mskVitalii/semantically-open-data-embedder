from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.dimensions = 1024

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=4,
            show_progress_bar=False
        )
        return [v[:self.dimensions].tolist() for v in vectors]
