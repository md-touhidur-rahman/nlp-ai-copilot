from pathlib import Path

from src.retrieval.embedder import Embedder
from src.retrieval.index import FaissIndex, load_chunks_jsonl


def main():
    chunks_path = Path("data/processed/sample.chunks.jsonl")  # start with sample
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}. Generate chunks first.")

    chunks = load_chunks_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]

    embedder = Embedder()
    vecs = embedder.embed_texts(texts)

    index = FaissIndex(dim=vecs.shape[1])
    index.add(vecs, meta=chunks)

    out_dir = Path("data/index")
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save(out_dir / "faiss.index", out_dir / "meta.json")

    print("Built index!")
    print("Chunks:", len(chunks))
    print("Dim:", vecs.shape[1])
    print("Saved:", out_dir / "faiss.index", "and", out_dir / "meta.json")


if __name__ == "__main__":
    main()
