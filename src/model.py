import os
import logging

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = None):
        # Priority: parameter → env var → default
        self.model_name = model_name or os.getenv("MODEL_NAME", "BAAI/bge-m3")

        self._log_device_info()

        # Enable trust_remote_code for jina models (they use custom code)
        trust_remote_code = "jina" in self.model_name.lower()
        self.model = SentenceTransformer(self.model_name, trust_remote_code=trust_remote_code)
        self.dimensions = 1024

        logger.info(f"Model loaded on device: {self.model.device}")

    @staticmethod
    def _log_device_info():
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"VRAM: {vram:.1f} GB")
        else:
            logger.info("CUDA not available, using CPU")

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
