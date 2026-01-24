from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.retrieval.embedder import Embedder
from src.retrieval.index import FaissIndex


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source: str
    text: str


class RetrievalService:
    def __init__(self, index_dir: Path = Path("data/index")):
        self.index_dir = index_dir
        self.embedder = Embedder()
        self.index = FaissIndex.load(index_dir / "faiss.index", index_dir / "meta.json")

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        qvec: np.ndarray = self.embedder.embed_query(query)
        results: List[Tuple[float, dict]] = self.index.search(qvec, top_k=top_k)

        out: List[RetrievedChunk] = []
        for score, meta in results:
            out.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    score=score,
                    source=meta.get("source", ""),
                    text=meta.get("text", ""),
                )
            )
        return out
