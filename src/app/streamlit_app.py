from src.retrieval.service import RetrievalService

import json
from pathlib import Path
import streamlit as st

from src.ingestion.pipeline import run_pipeline

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

st.set_page_config(page_title="NLP AI Copilot", layout="wide")
st.title("NLP AI Copilot — Ingestion Dashboard")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Sidebar: choose PDF
st.sidebar.header("Documents")
pdfs = sorted([p for p in RAW_DIR.glob("*.pdf")])

if not pdfs:
    st.warning("No PDFs found in data/raw. Add a PDF there first.")
    st.stop()

selected = st.sidebar.selectbox("Select a PDF", pdfs, format_func=lambda p: p.name)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Selected PDF")
    st.write(selected.name)
    st.write("Path:", str(selected))

    if st.button("Run pipeline (extract → clean → chunk)"):
        with st.spinner("Running pipeline..."):
            chunks_path = run_pipeline(selected)
        st.success(f"Done! Chunks saved to: {chunks_path}")

with colB:
    st.subheader("Processed artifacts")
    related = sorted(PROCESSED_DIR.glob(f"{selected.stem}*"))
    if related:
        for f in related:
            st.write("•", f.name)
    else:
        st.info("No processed files yet. Run the pipeline.")

st.divider()

# Show chunks if available
chunks_file = PROCESSED_DIR / f"{selected.stem}.chunks.jsonl"
if chunks_file.exists():
    st.subheader("Chunks Viewer")
    query = st.text_input("Search inside chunks (keyword search)", "")

    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if query.strip():
        q = query.lower()
        chunks = [c for c in chunks if q in c["text"].lower()]

    st.write(f"Showing {len(chunks)} chunk(s) from {chunks_file.name}")

    for c in chunks[:20]:
        with st.expander(f'{c["chunk_id"]}  ({c["start_char"]}-{c["end_char"]})'):
            st.write(c["text"][:2000])
else:
    st.info("No chunks found yet. Run the pipeline to generate chunks.")


st.divider()
st.subheader("Semantic Search (FAISS)")

# Load retrieval service once
@st.cache_resource
def get_retriever():
    return RetrievalService()

if (Path("data/index/faiss.index").exists()) and (Path("data/index/meta.json").exists()):
    retriever = get_retriever()

    q = st.text_input("Ask a question (semantic search)", "")
    top_k = st.slider("Top K", 1, 10, 5)

    if q.strip():
        with st.spinner("Searching..."):
            results = retriever.search(q, top_k=top_k)

        for r in results:
            with st.expander(f"{r.chunk_id}  score={r.score:.4f}  source={r.source}"):
                st.write(r.text)
else:
    st.info("No FAISS index found. Run: PYTHONPATH=. python src/retrieval/build_index.py")
