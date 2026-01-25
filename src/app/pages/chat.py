# src/app/pages/chat.py
import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.agent.graph import run_agent
from src.agent.router import Route
from src.agent import tools

st.set_page_config(page_title="Unified Chat", layout="wide")

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


def _find_best_xy(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    # prefer date-like
    for c in cols:
        if isinstance(c, str) and c.lower() in ("date", "datetime", "timestamp", "time"):
            if numeric_cols:
                return {"x": c, "y": numeric_cols[0]}

    # common aggregated form: first label, second numeric
    if len(cols) >= 2 and pd.api.types.is_numeric_dtype(df[cols[1]]):
        return {"x": cols[0], "y": cols[1]}

    non_numeric = [c for c in cols if c not in numeric_cols]
    if non_numeric and numeric_cols:
        return {"x": non_numeric[0], "y": numeric_cols[0]}

    if numeric_cols:
        return {"x": "__index__", "y": numeric_cols[0]}

    return None


def plot_dataframe(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No rows to plot.")
        return

    pick = _find_best_xy(df)
    if not pick:
        st.info("No plottable numeric columns found.")
        return

    x_col, y_col = pick["x"], pick["y"]
    df_plot = df.copy()

    if x_col == "__index__":
        df_plot = df_plot.reset_index(drop=False).rename(columns={"index": "index"})
        x_col = "index"

    try:
        df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="ignore")
    except Exception:
        pass

    try:
        df_plot = df_plot.sort_values(x_col)
    except Exception:
        pass

    try:
        st.line_chart(df_plot.set_index(x_col)[y_col])
    except Exception:
        try:
            st.bar_chart(df_plot.set_index(x_col)[y_col])
        except Exception:
            st.info("Could not render plot for this output.")


def render_data_evidence(data_out: Dict[str, Any]) -> None:
    dq = data_out.get("dq", {}) or {}
    df_res = data_out.get("df")

    if df_res is not None and not isinstance(df_res, pd.DataFrame):
        try:
            if isinstance(df_res, list):
                df_res = pd.DataFrame(df_res)
            elif isinstance(df_res, dict):
                df_res = pd.DataFrame(df_res)
        except Exception:
            df_res = None

    with st.expander("Evidence (computed)", expanded=True):
        st.write("Parsed query (structured)")
        st.json(safe_json(dq))

        if isinstance(df_res, pd.DataFrame) and not df_res.empty:
            st.write("Result table")
            st.dataframe(df_res, use_container_width=True)

            st.write("Plot")
            plot_dataframe(df_res)
        else:
            if data_out.get("error"):
                st.error(data_out["error"])
                if data_out.get("hint"):
                    st.info(data_out["hint"])
            else:
                st.warning("No result table available to display/plot (empty or missing).")


def render_rag_evidence(rag_out: Dict[str, Any]) -> None:
    chunks = rag_out.get("chunks", []) or []
    top_score = rag_out.get("top_score", None)

    with st.expander("Evidence (retrieved chunks)", expanded=True):
        if top_score is not None:
            st.write(f"Top score: {top_score}")

        if not chunks:
            st.info("No chunks returned.")
            return

        for i, ch in enumerate(chunks, start=1):
            if isinstance(ch, dict):
                chunk_id = ch.get("chunk_id") or ch.get("id") or f"chunk_{i}"
                score = ch.get("score", None)
                text = ch.get("text") or ch.get("content") or ""
                header = f"{i}. {chunk_id}"
                if score is not None:
                    header += f" (score={score})"
                st.markdown(f"**{header}**")
                st.markdown(text)
            else:
                st.markdown(f"**{i}.**")
                st.markdown(str(ch))


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("Routing")
    st.caption("AUTO uses agent. Override forces a single tool path.")
    override = st.toggle("Override routing", value=False, key="chat_override_toggle")
    forced_route = st.selectbox(
        "Force route",
        options=[Route.DATA.value, Route.RAG.value, Route.MIXED.value, Route.GENERAL.value],
        index=0,
        disabled=not override,
        key="chat_forced_route_select",
    )

    st.divider()
    st.header("Default dataset")
    st.code(DEFAULT_DATASET_PATH)
    st.caption("Found CSV" if os.path.exists(DEFAULT_DATASET_PATH) else "CSV not found at this path")

    st.divider()
    st.header("Upload dataset (CSV)")
    uploaded = st.file_uploader("CSV upload", type=["csv"], key="chat_csv_uploader")

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            tools.set_active_df(df_up, name=getattr(uploaded, "name", "uploaded.csv"))
            st.success(f"Using uploaded dataset: {df_up.shape[0]} rows × {df_up.shape[1]} cols")
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(df_up.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            tools.clear_active_df()
    else:
        tools.clear_active_df()

    st.divider()
    st.header("Grounding policy")
    st.write("RAG: cites retrieved chunks or refuses.")
    st.write("DATA: computed values + table + plot (no guessing).")

    st.divider()
    if st.button("Clear chat history", key="chat_clear_history_btn"):
        st.session_state.messages = []
        st.rerun()


# --------------------------
# Main
# --------------------------
st.title("Unified Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Details", expanded=False):
                st.json(safe_json(msg["meta"]))

prompt = st.chat_input("Ask about your documents or your dataset...", key="chat_input")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

    with st.chat_message("assistant"):
        state = run_agent(prompt, override=override, forced_route=forced_route if override else None)

        mode = state.get("mode", "general")
        answer = state.get("answer", "")
        evidence = state.get("evidence", {}) or {}
        router_meta = state.get("router_meta", {}) or {}

        st.markdown(f"**Mode:** `{mode}`")
        if router_meta.get("router_explanation"):
            st.caption(router_meta["router_explanation"])
        if router_meta.get("trace_path"):
            st.caption(f"Trace saved: `{router_meta['trace_path']}`")

        st.markdown(answer)

        if mode == "data":
            data_out = evidence.get("data", {}) or {}
            render_data_evidence(data_out)

        elif mode == "rag":
            rag_out = evidence.get("rag", {}) or {}
            render_rag_evidence(rag_out)

        # ✅ DO NOT store df/evidence in history meta
        meta_safe = {"mode": mode, "route": state.get("route"), "router_meta": router_meta}

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})
