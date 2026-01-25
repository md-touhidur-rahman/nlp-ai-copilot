# src/app/pages/chat.py
import os
import pandas as pd
import streamlit as st
from typing import Any, Dict, Optional

from src.agent.router import Route
from src.agent.graph import run_agent
from src.agent import tools

st.set_page_config(page_title="Evidence Copilot", layout="wide")

DEFAULT_DATASET_PATH = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")


# --------------------------
# Helpers
# --------------------------
def safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    return str(obj)


def render_trace_table(trace_list: Any) -> None:
    if not trace_list:
        st.info("No trace available.")
        return
    rows = []
    for t in trace_list:
        if isinstance(t, dict):
            rows.append({"node": t.get("node", ""), "ms": t.get("ms", None)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Trace format not recognized.")


def plot_result(df_out: Any, dq: Optional[Dict[str, Any]] = None) -> None:
    if df_out is None or getattr(df_out, "empty", True):
        st.info("No rows to plot.")
        return

    cols = list(df_out.columns)
    if len(cols) < 2:
        st.info("Result has fewer than 2 columns — cannot plot.")
        return

    x_col, y_col = cols[0], cols[1]
    df_plot = df_out.copy()
    mode = (dq or {}).get("mode", "")

    try:
        if mode == "monthly":
            df_plot["_sort_key"] = pd.to_datetime(df_plot[x_col].astype(str) + "-01", errors="coerce")
            df_plot = df_plot.sort_values("_sort_key").drop(columns=["_sort_key"])
        elif mode == "yearly":
            df_plot["_sort_key"] = pd.to_datetime(df_plot[x_col].astype(str) + "-01-01", errors="coerce")
            df_plot = df_plot.sort_values("_sort_key").drop(columns=["_sort_key"])
        else:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="coerce")
            df_plot = df_plot.sort_values(x_col)
    except Exception:
        pass

    try:
        st.line_chart(df_plot.set_index(x_col)[y_col])
    except Exception:
        try:
            st.bar_chart(df_plot.set_index(x_col)[y_col])
        except Exception:
            st.info("Could not auto-plot this result.")


def render_data_evidence(data_out: Dict[str, Any]) -> None:
    dq = (data_out or {}).get("dq", {}) or {}
    df_res = (data_out or {}).get("df", None)

    tab_ans, tab_table, tab_plot, tab_parsed = st.tabs(["Answer", "Table", "Plot", "Parsed"])

    with tab_table:
        if df_res is not None:
            st.dataframe(df_res, use_container_width=True)
        else:
            st.info("No table returned.")

    with tab_plot:
        if df_res is not None:
            plot_result(df_res, dq=dq)
        else:
            st.info("No plot available.")

    with tab_parsed:
        st.json(safe_json(dq))


def render_rag_evidence(rag_out: Dict[str, Any]) -> None:
    chunks = (rag_out or {}).get("chunks", []) or []
    top_score = (rag_out or {}).get("top_score")

    if top_score is not None:
        st.caption(f"Top score: {top_score}")

    if not chunks:
        st.info("No chunks returned.")
        return

    for i, ch in enumerate(chunks, start=1):
        if isinstance(ch, dict):
            chunk_id = ch.get("chunk_id") or ch.get("id") or f"chunk_{i}"
            score = ch.get("score", None)
            text = ch.get("text") or ch.get("content") or ""
            hdr = f"{i}. {chunk_id}" + (f" (score={score})" if score is not None else "")
            st.markdown(f"**{hdr}**")
            st.markdown(text)
        else:
            st.markdown(str(ch))


def load_uploaded_csv(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


# --------------------------
# UI Header
# --------------------------
st.title("Evidence Copilot")
st.caption("Evidence-grounded answers for documents (RAG) and datasets (computed).")


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("Routing")
    st.caption("AUTO runs the agent. Override forces a single tool path.")

    override = st.toggle("Override routing", value=False, key="override_routing")
    forced_route = st.selectbox(
        "Force route",
        options=[Route.DATA.value, Route.RAG.value, Route.MIXED.value, Route.GENERAL.value],
        index=0,
        disabled=not override,
        key="forced_route_select",
    )

    st.divider()

    st.header("Default dataset")
    st.code(DEFAULT_DATASET_PATH)
    st.caption("Found CSV" if os.path.exists(DEFAULT_DATASET_PATH) else "CSV not found")

    st.divider()

    st.header("Upload dataset (CSV)")
    st.caption("Upload a CSV to use for DATA answers")
    upload = st.file_uploader("CSV upload", type=["csv"], key="csv_uploader")

    user_df = load_uploaded_csv(upload)
    if user_df is not None:
        st.success(f"Loaded CSV: {user_df.shape[0]} rows × {user_df.shape[1]} cols")
        with st.expander("Preview", expanded=False):
            st.dataframe(user_df.head(25), use_container_width=True)

    st.divider()

    st.header("Grounding policy")
    st.write("RAG: answers based on retrieved chunks (or refuses).")
    st.write("DATA: computed values + tables + plots (no guessing).")

    st.divider()
    if st.button("Clear chat", key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()


# --------------------------
# Session state
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta_safe}]


# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta:
            with st.expander("Details", expanded=False):
                st.json(safe_json(meta))


# Suggestions (chips)
SUGGESTIONS = [
    "sales in 2013",
    "sales in march to june in 2013",
    "monthly sales from 2014 to 2016",
    "sales in store 2 in march to june in 2013",
    "top 5 months by sales in 2013",
    "according to the document, summarize the main points",
]
st.caption("Suggestions:")
cols = st.columns(3)
for i, s in enumerate(SUGGESTIONS):
    with cols[i % 3]:
        if st.button(s, key=f"suggest_{i}"):
            st.session_state["draft_prompt"] = s
            st.rerun()


prompt = st.chat_input("Ask about your documents or your dataset...", key="chat_input")

if not prompt and st.session_state.get("draft_prompt"):
    prompt = st.session_state.pop("draft_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

    # Build dataset context for tools/agent
    dataset_ctx: Optional[Dict[str, Any]] = None
    if user_df is not None:
        dataset_ctx = {"df": user_df, "name": "uploaded.csv"}
    else:
        dataset_ctx = {"path": DEFAULT_DATASET_PATH, "name": "store_sales_train.csv"}

    with st.chat_message("assistant"):
        # --------------------------
        # OVERRIDE MODE: direct tool calls
        # --------------------------
        if override:
            chosen = forced_route
            st.markdown(f"**Mode:** `{chosen}`")
            st.caption(f"Forced route: {chosen}")

            if chosen == Route.DATA.value:
                out = tools.run_data(prompt, dataset=dataset_ctx)
                if out.get("error"):
                    answer = f"Data pipeline error: {out['error']}"
                    if out.get("hint"):
                        answer += f"\n\n{out.get('hint')}"
                    st.error(answer)
                else:
                    answer = out.get("summary") or "Computed result from dataset."
                    st.markdown(answer)

                # Evidence tabs (plot included)
                with st.expander("Evidence", expanded=True):
                    render_data_evidence(out)

                meta_safe = {"mode": "data", "route": "data"}
                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})

            elif chosen == Route.RAG.value:
                out = tools.run_rag(prompt)
                if out.get("error"):
                    answer = out["error"]
                    st.error(answer)
                else:
                    answer = out.get("answer", "")
                    st.markdown(answer)

                with st.expander("Evidence", expanded=True):
                    render_rag_evidence(out)

                meta_safe = {"mode": "rag", "route": "rag"}
                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})

            else:
                answer = "Override is set to GENERAL/MIXED. For override mode, choose DATA or RAG."
                st.info(answer)
                meta_safe = {"mode": "general", "route": chosen}
                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})

        # --------------------------
        # AUTO MODE: agent
        # --------------------------
        else:
            state = run_agent(prompt, dataset=dataset_ctx)

            mode = state.get("mode", "general")
            route_val = state.get("route")
            answer = state.get("answer", "")
            router_meta = state.get("router_meta", {}) or {}
            evidence = state.get("evidence", {}) or {}
            trace_list = state.get("trace", []) or []

            st.markdown(f"**Mode:** `{mode}`")
            if router_meta.get("router_explanation"):
                st.caption(router_meta["router_explanation"])
            if router_meta.get("trace_path"):
                st.caption(f"Trace saved: {router_meta['trace_path']}")

            st.markdown(answer)

            with st.expander("Evidence", expanded=True):
                if mode == "data":
                    data_out = evidence.get("data", {}) or {}
                    render_data_evidence(data_out)
                elif mode == "rag":
                    rag_out = evidence.get("rag", {}) or {}
                    render_rag_evidence(rag_out)
                else:
                    st.json(safe_json(evidence))

            with st.expander("Trace (node timings)", expanded=False):
                render_trace_table(trace_list)

            # ✅ Do not store DF in message history meta
            meta_safe = {"mode": mode, "route": route_val, "router_meta": router_meta}
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})
