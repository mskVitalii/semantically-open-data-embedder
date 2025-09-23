import numpy as np
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

    def embed_tokens(self, text: str) -> dict[str, list]:
        tokens = self.model.tokenizer.tokenize(text)
        token_ids = self.model.tokenizer.convert_tokens_to_ids(tokens)
        embeddings = self.model.encode(tokens, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings[:, :self.dimensions]
        return {"tokens": tokens, "embeddings": embeddings.tolist()}

    def export_tokens_tsv(self, text: str, vectors_file: str, metadata_file: str):
        result = self.embed_tokens(text)
        tokens = result["tokens"]
        embeddings = np.array(result["embeddings"])

        np.savetxt(vectors_file, embeddings, delimiter="\t", fmt="%.6f")
        with open(metadata_file, "w", encoding="utf-8") as f:
            for t in tokens:
                f.write(t + "\n")
        return vectors_file, metadata_file
