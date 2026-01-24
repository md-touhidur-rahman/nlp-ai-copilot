from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
import faiss


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    source: str


def load_chunks_jsonl(chunks_path: Path) -> List[dict]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


class FaissIndex:
    """
    Simple cosine-similarity index using normalized vectors + inner product.
    If vectors are normalized, inner product == cosine similarity.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product
        self.meta: List[dict] = []

    def add(self, vectors: np.ndarray, meta: List[dict]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n, {self.dim}), got {vectors.shape}")
        self.index.add(vectors)
        self.meta.extend(meta)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec, top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((float(score), self.meta[int(idx)]))
        return out

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(index_path: Path, meta_path: Path) -> "FaissIndex":
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = index.d
        obj = FaissIndex(dim=dim)
        obj.index = index
        obj.meta = meta
        return obj
