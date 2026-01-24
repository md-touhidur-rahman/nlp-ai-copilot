from pathlib import Path

from src.retrieval.embedder import Embedder
from src.retrieval.index import FaissIndex


def main():
    index_dir = Path("data/index")
    index = FaissIndex.load(index_dir / "faiss.index", index_dir / "meta.json")
    embedder = Embedder()

    print("Type a question (or 'exit'):")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        qvec = embedder.embed_query(q)
        results = index.search(qvec, top_k=5)

        print("\nTop matches:\n")
        for score, meta in results:
            print(f"- {meta['chunk_id']}  score={score:.4f}  source={meta.get('source','')}")
            preview = meta["text"][:220].replace("\n", " ")
            print(f"  {preview}...")
        print()


if __name__ == "__main__":
    main()
