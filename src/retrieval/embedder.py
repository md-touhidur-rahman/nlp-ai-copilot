from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True  # good for cosine similarity


class Embedder:
    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()
        self.model = SentenceTransformer(self.config.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns embeddings as float32 numpy array: shape (n, dim)
        """
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) >= 32,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
