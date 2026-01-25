# src/app/pages/chat.py
import os
import pandas as pd
import streamlit as st
from typing import Any, Dict, Optional

from src.agent.router import Route
from src.agent.graph import run_agent
from src.agent import tools

st.set_page_config(page_title="Unified Chat", layout="wide")

DATASET_REL_PATH = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")


# --------------------------
# Helpers
# --------------------------
def safe_json(obj: Any) -> Any:
    """Make objects safe for st.json."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    return str(obj)


def render_trace(trace_list: Any) -> None:
    """Render node timings table from list[dict]."""
    if not trace_list:
        st.info("No trace available.")
        return

    rows = []
    for t in trace_list:
        if isinstance(t, dict):
            rows.append(
                {
                    "node": t.get("node", ""),
                    "ms": round(float(t.get("ms", 0.0)), 2) if t.get("ms") is not None else None,
                }
            )
        else:
            # fallback
            rows.append({"node": str(t), "ms": None})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def plot_result(df_out: Any, dq: Optional[Dict[str, Any]] = None) -> None:
    """Robust plotting for daily/monthly/yearly."""
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
            sort_key = pd.to_datetime(df_plot[x_col].astype(str) + "-01", errors="coerce")
            df_plot["_sort_key"] = sort_key
            df_plot = df_plot.sort_values("_sort_key").drop(columns=["_sort_key"])
        elif mode == "yearly":
            sort_key = pd.to_datetime(df_plot[x_col].astype(str) + "-01-01", errors="coerce")
            df_plot["_sort_key"] = sort_key
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


def format_peak_and_top5(dq: Dict[str, Any], df_res: Any) -> None:
    """For monthly/yearly results: show peak + top-5."""
    if df_res is None or getattr(df_res, "empty", True):
        return
    if not hasattr(df_res, "columns") or len(df_res.columns) < 2:
        return

    mode = (dq or {}).get("mode", "")
    metric = (dq or {}).get("metric", "value")

    if mode not in ("monthly", "yearly"):
        return

    label_col = df_res.columns[0]
    value_col = df_res.columns[1]
    df_sorted = df_res.sort_values(value_col, ascending=False).reset_index(drop=True)

    top = df_sorted.iloc[0]
    try:
        st.success(f"Peak {mode}: {top[label_col]} with {float(top[value_col]):,.0f} {metric}")
    except Exception:
        st.success(f"Peak {mode}: {top[label_col]}")

    st.write("Top 5 peaks")
    st.dataframe(df_sorted.head(5), use_container_width=True)


# --------------------------
# UI
# --------------------------
st.title("Unified Chat")

with st.sidebar:
    st.header("Routing")
    st.caption("AUTO uses LangGraph. Override forces a single tool path.")

    override = st.toggle("Override routing", value=False)
    forced_route = st.selectbox(
        "Force route",
        options=[Route.DATA.value, Route.RAG.value, Route.MIXED.value, Route.GENERAL.value],
        index=0,
        disabled=not override,
    )

    st.divider()
    st.header("Dataset")
    st.code(DATASET_REL_PATH)
    st.caption("Found CSV" if os.path.exists(DATASET_REL_PATH) else "CSV not found at this path")

    st.divider()
    st.header("Grounding policy")
    st.write("RAG: cites retrieved chunks or refuses.")
    st.write("DATA: computed values + table + plot (no guessing).")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, meta}]

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta:
            with st.expander("Details", expanded=False):
                st.json(safe_json(meta))

prompt = st.chat_input("Ask about your PDFs or your sales dataset...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

    assistant_text = ""
    assistant_meta: Dict[str, Any] = {}

    with st.chat_message("assistant"):
        # --------------------------
        # OVERRIDE MODE (direct tool execution)
        # --------------------------
        if override:
            chosen = forced_route
            st.markdown(f"**Mode:** `{chosen}`")
            st.caption(f"Forced route: {chosen}")

            router_meta = tools.route_question(prompt)

            if chosen == Route.DATA.value:
                out = tools.run_data(prompt)
                assistant_meta = {"route": chosen, "router_meta": router_meta}

                if "error" in out:
                    assistant_text = f"Data pipeline error: {out['error']}"
                    st.error(assistant_text)
                    assistant_meta["error"] = out["error"]
                else:
                    dq = out.get("dq", {}) or {}
                    df_res = out.get("df", None)

                    assistant_text = "Computed result from dataset."
                    st.markdown(assistant_text)

                    format_peak_and_top5(dq, df_res)

                    with st.expander("Evidence (computed)", expanded=True):
                        st.write("Parsed DataQuery")
                        st.json(safe_json(dq))

                        if df_res is not None:
                            st.write("Result table")
                            st.dataframe(df_res, use_container_width=True)

                            st.write("Plot")
                            plot_result(df_res, dq=dq)

                    assistant_meta["data"] = {
                        "dq": dq,
                        "result_shape": getattr(df_res, "shape", None),
                        "columns": list(getattr(df_res, "columns", [])) if df_res is not None else [],
                    }

            elif chosen == Route.RAG.value:
                out = tools.run_rag(prompt)
                assistant_meta = {"route": chosen, "router_meta": router_meta}

                if "error" in out:
                    assistant_text = out["error"]
                    st.error(assistant_text)
                    assistant_meta["error"] = out["error"]
                else:
                    assistant_text = out.get("answer", "")
                    st.markdown(assistant_text)

                    with st.expander("Evidence (retrieved chunks)", expanded=True):
                        top_score = out.get("top_score")
                        if top_score is not None:
                            st.write(f"Top score: {top_score}")

                        chunks = out.get("chunks", []) or []
                        if not chunks:
                            st.info("No chunks returned.")
                        else:
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

                    assistant_meta["rag"] = {
                        "top_score": out.get("top_score"),
                        "chunks_count": len(out.get("chunks", []) or []),
                    }

            else:
                assistant_text = (
                    "Override is set to GENERAL/MIXED. For Phase 5 override, choose DATA or RAG."
                )
                st.info(assistant_text)
                assistant_meta = {"route": chosen, "router_meta": router_meta}

        # --------------------------
        # AUTO MODE (LangGraph) — expects AgentState
        # --------------------------
        else:
            state = run_agent(prompt)  # MUST be AgentState

            chosen = getattr(state, "mode", "general")
            assistant_text = getattr(state, "answer", "")
            route_val = getattr(state, "route", None)
            router_meta = getattr(state, "router_meta", {}) or {}
            evidence = getattr(state, "evidence", {}) or {}
            trace_objs = getattr(state, "trace", []) or []

            # trace as list[dict]
            trace = []
            for t in trace_objs:
                if isinstance(t, dict):
                    trace.append(t)
                else:
                    trace.append(getattr(t, "__dict__", {"node": str(t), "ms": None}))

            assistant_meta = {
                "route": route_val,
                "mode": chosen,
                "router_meta": router_meta,
                "trace": trace,
            }

            st.markdown(f"**Mode:** `{chosen}`")
            st.caption(router_meta.get("router_explanation", ""))

            # trace path (Phase 6)
            trace_path = router_meta.get("trace_path")
            if trace_path:
                st.caption(f"Trace saved: {trace_path}")

            st.markdown(assistant_text)

            if chosen == "data":
                ev = evidence.get("data", {}) or {}
                dq = ev.get("dq", {}) or {}
                df_res = ev.get("df", None)

                format_peak_and_top5(dq, df_res)

                with st.expander("Evidence (computed)", expanded=True):
                    st.write("Parsed DataQuery")
                    st.json(safe_json(dq))

                    if df_res is not None:
                        st.write("Result table")
                        st.dataframe(df_res, use_container_width=True)

                        st.write("Plot")
                        plot_result(df_res, dq=dq)

            elif chosen == "rag":
                ev = evidence.get("rag", {}) or {}
                chunks = ev.get("chunks", []) or []
                top_score = ev.get("top_score", None)

                with st.expander("Evidence (retrieved chunks)", expanded=True):
                    if top_score is not None:
                        st.write(f"Top score: {top_score}")

                    if not chunks:
                        st.info("No chunks returned.")
                    else:
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

            with st.expander("Trace (node timings)", expanded=False):
                render_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "meta": assistant_meta}
    )
