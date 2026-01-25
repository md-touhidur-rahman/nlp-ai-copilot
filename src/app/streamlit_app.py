from pathlib import Path
import json

import streamlit as st

from src.ingestion.pipeline import run_pipeline
from src.retrieval.service import RetrievalService
from src.agent.rag_answerer import RAGAnswerer

# Data modules (Phase 3)
from src.data.loader import load_table
from src.data.filtering import filter_sales
from src.data.aggregate import group_sum_by_date, group_sum, group_sum_by_month, group_sum_by_year
from src.data.plotting import prepare_line_chart


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DATASET_DEFAULT = Path("data/datasets/tabular/store_sales_train.csv")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="NLP AI Copilot", layout="wide")
st.title("NLP AI Copilot")

# -------------------------
# Helpers (cached resources)
# -------------------------
@st.cache_resource
def get_retriever():
    return RetrievalService()


@st.cache_resource
def get_rag():
    return RAGAnswerer()


# -------------------------
# Main Tabs
# -------------------------
tab_docs, tab_data = st.tabs(["📄 Docs + RAG", "📊 Data Explorer"])


# =========================================================
# TAB 1 — DOCS + INGESTION + CHUNKS VIEWER + SEARCH + RAG
# =========================================================
with tab_docs:
    st.header("Docs & RAG Dashboard")

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

    st.divider()
    st.subheader("RAG Answer (Local LLM – Ollama)")

    rag = get_rag()
    q = st.text_input("Ask a question (RAG)", key="rag_question")

    if q.strip():
        with st.spinner("Retrieving and generating answer..."):
            out = rag.answer(q)

        st.markdown("### Answer")
        st.write(out["answer"])

        # Shows if answer came from document or general mode
        st.caption(f"Mode: {out['mode']} | top_score={out['top_score']:.4f}")

        st.markdown("### Sources")
        for c in out["chunks"]:
            with st.expander(f"{c.chunk_id} (score={c.score:.4f})"):
                st.write(c.text)


# ======================================
# TAB 2 — DATA EXPLORER (Phase 3)
# ======================================
with tab_data:
    st.header("Data Explorer (Store Sales)")

    st.write(
        "This page is **pure computation** (filter → aggregate → plot). "
        "No LLM needed yet. Later we’ll add LLM → tool routing."
    )

    # Path input
    csv_path = st.text_input(
        "CSV path",
        value=str(DATASET_DEFAULT),
        help="Put your CSV in data/datasets/tabular/ and enter the relative path here."
    )

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.number_input("store_nbr", min_value=1, value=1, step=1)
    with col2:
        family = st.text_input("family", value="GROCERY I")
    with col3:
        metric = st.selectbox("metric", options=["sales"], index=0)

    col4, col5 = st.columns(2)
    with col4:
        start = st.text_input("start date (YYYY-MM-DD)", value="2013-01-01")
    with col5:
        end = st.text_input("end date (YYYY-MM-DD)", value="2013-02-01")

    run_btn = st.button("Run analysis")

    if run_btn:
        path_obj = Path(csv_path)
        if not path_obj.exists():
            st.error(f"File not found: {csv_path}")
            st.stop()

        with st.spinner("Loading CSV..."):
            df = load_table(csv_path)

        with st.spinner("Filtering..."):
            df2 = filter_sales(
                df,
                store_nbr=int(store),
                family=family,
                start=start,
                end=end,
                date_col="date",
            )

        if len(df2) == 0:
            st.warning("No rows after filtering. Try different store/family/date range.")
            st.stop()

        with st.spinner("Aggregating..."):
            daily = group_sum_by_date(df2, date_col="date", value_col=metric)

        chart_df = prepare_line_chart(daily, x="date", y=metric)

        st.success(f"Filtered rows: {len(df2)} | Days: {len(daily)}")

        st.subheader("Preview (daily totals)")
        st.dataframe(daily.head(30), use_container_width=True)

        st.subheader("Trend")
        st.line_chart(chart_df, x="date", y=metric)

        # Optional: basic stats
        st.subheader("Quick stats")
        st.write(
            {
                "min": float(daily[metric].min()),
                "max": float(daily[metric].max()),
                "mean": float(daily[metric].mean()),
                "sum": float(daily[metric].sum()),
            }
        )
