# src/app/pages/chat.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.agent.graph import run_agent
from src.agent.router import Route
from src.agent import tools
from src.app.workspace import ws, add_dataset

APP_NAME = "Evidence Copilot"
DEFAULT_DATASET_REL_PATH = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")


# --------------------------
# Safe JSON (never store DF)
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


def evidence_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Compact evidence summary safe for JSON (no large payloads)."""
    if not isinstance(evidence, dict):
        return {"type": str(type(evidence))}
    out = {"keys": list(evidence.keys())}
    for k, v in evidence.items():
        if isinstance(v, dict):
            block = {"keys": list(v.keys())}
            df = v.get("df")
            if isinstance(df, pd.DataFrame):
                block["df_shape"] = list(df.shape)
                block["df_columns"] = list(df.columns)[:25]
            out[k] = block
        else:
            out[k] = {"type": str(type(v))}
    return out


def df_from_any(x: Any) -> Optional[pd.DataFrame]:
    return x if isinstance(x, pd.DataFrame) else None


def plot_df(df_out: pd.DataFrame, mode: str = "") -> None:
    if df_out is None or df_out.empty:
        st.info("No rows to plot.")
        return
    if len(df_out.columns) < 2:
        st.info("Need at least 2 columns to plot.")
        return

    x_col, y_col = df_out.columns[0], df_out.columns[1]
    df_plot = df_out.copy()

    try:
        if mode == "monthly":
            key = pd.to_datetime(df_plot[x_col].astype(str) + "-01", errors="coerce")
            df_plot["_k"] = key
            df_plot = df_plot.sort_values("_k").drop(columns=["_k"])
        elif mode == "yearly":
            key = pd.to_datetime(df_plot[x_col].astype(str) + "-01-01", errors="coerce")
            df_plot["_k"] = key
            df_plot = df_plot.sort_values("_k").drop(columns=["_k"])
        else:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="coerce")
            df_plot = df_plot.sort_values(x_col)
    except Exception:
        pass

    try:
        st.line_chart(df_plot.set_index(x_col)[y_col])
    except Exception:
        st.bar_chart(df_plot.set_index(x_col)[y_col])


def render_peak_top5(dq: Dict[str, Any], df_res: pd.DataFrame) -> None:
    if df_res is None or df_res.empty or len(df_res.columns) < 2:
        return
    mode = (dq or {}).get("mode", "")
    metric = (dq or {}).get("metric", "value")
    if mode not in ("monthly", "yearly"):
        return

    label_col, value_col = df_res.columns[0], df_res.columns[1]
    df_sorted = df_res.sort_values(value_col, ascending=False).reset_index(drop=True)
    top = df_sorted.iloc[0]

    try:
        st.success(f"Peak {mode}: {top[label_col]} with {float(top[value_col]):,.0f} {metric}")
    except Exception:
        st.success(f"Peak {mode}: {top[label_col]}")

    st.write("Top 5")
    st.dataframe(df_sorted.head(5), use_container_width=True)


# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Evidence-grounded answers over documents (RAG) and datasets (computed).")

w = ws()  # init workspace


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("Controls")

    override = st.toggle("Override routing", value=False, key="chat_override_routing")
    forced_route = st.selectbox(
        "Force route",
        options=[Route.DATA.value, Route.RAG.value, Route.MIXED.value, Route.GENERAL.value],
        index=0,
        disabled=not override,
        key="chat_forced_route",
    )

    st.divider()
    st.subheader("Active dataset")
    active_name = w.get("active_dataset")
    if active_name:
        st.success(f"Active: {active_name}")
    else:
        st.caption("Active: default dataset")

    st.caption("Default dataset path")
    st.code(DEFAULT_DATASET_REL_PATH)
    st.caption("Found CSV" if os.path.exists(DEFAULT_DATASET_REL_PATH) else "CSV not found")

    st.divider()
    st.subheader("Upload dataset (CSV)")
    st.caption("Upload a CSV. It becomes available in Data Explorer and for DATA answers.")
    up = st.file_uploader("CSV upload", type=["csv"], key="chat_csv_upload")

    if up is not None:
        try:
            df = pd.read_csv(up)
            name = getattr(up, "name", "uploaded.csv")

            # Put into workspace
            add_dataset(name, df)

            # Also tell DATA tool to use it immediately
            tools.set_active_df(df, name=name)

            st.success(f"Loaded: {name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.button("Use default dataset", key="chat_use_default"):
        # Clear active DF so tools fall back to default.csv
        tools.clear_active_df()
        w["active_dataset"] = None
        st.rerun()

    st.divider()
    st.subheader("Grounding policy")
    st.write("RAG: answers based on retrieved chunks (or refuses).")
    st.write("DATA: computed values + tables + plots (no guessing).")

    st.divider()
    if st.button("Clear chat", key="chat_clear_btn"):
        st.session_state.messages = []
        st.rerun()


# --------------------------
# Session state: chat history
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta:
            with st.expander("Details", expanded=False):
                st.json(safe_json(meta))


# --------------------------
# Chat input
# --------------------------
prompt = st.chat_input("Ask a question about your documents or your dataset...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

    with st.chat_message("assistant"):
        tab_answer, tab_evidence, tab_trace = st.tabs(["Answer", "Evidence", "Trace"])

        # Make sure tools knows the active dataset (if workspace has one)
        active_name = w.get("active_dataset")
        if active_name:
            ds = w.get("datasets", {}).get(active_name)
            if ds and isinstance(ds.get("df"), pd.DataFrame):
                tools.set_active_df(ds["df"], name=active_name)

        # --------------------------
        # OVERRIDE (direct tool execution)
        # --------------------------
        if override:
            chosen = forced_route
            router_meta = tools.route_question(prompt)

            with tab_answer:
                st.markdown(f"**Mode:** `{chosen}`")
                st.caption(f"Forced route. Router saw: {router_meta.get('router_explanation', '')}")

            assistant_text = ""
            evidence: Dict[str, Any] = {}

            if chosen == Route.DATA.value:
                out = tools.run_data(prompt)  # uses active DF if set
                evidence = {"data": out}

                if out.get("error"):
                    assistant_text = f"Data pipeline error: {out.get('error')}"
                    if out.get("hint"):
                        assistant_text += f"\n\n{out['hint']}"
                else:
                    assistant_text = out.get("summary", "Computed result from dataset.")

                with tab_answer:
                    st.markdown(assistant_text)

                with tab_evidence:
                    data_out = evidence.get("data", {}) or {}
                    dq = data_out.get("dq", {}) or {}
                    df_res = df_from_any(data_out.get("df"))

                    st.subheader("Parsed query (structured)")
                    st.json(safe_json(dq))

                    if df_res is not None:
                        render_peak_top5(dq, df_res)
                        st.subheader("Result table")
                        st.dataframe(df_res, use_container_width=True)
                        st.subheader("Plot")
                        plot_df(df_res, mode=str(dq.get("mode", "")))
                    else:
                        if data_out.get("error"):
                            st.error(data_out["error"])
                        if data_out.get("hint"):
                            st.info(data_out["hint"])
                        if not data_out.get("error") and not data_out.get("hint"):
                            st.info("No result table produced.")

            elif chosen == Route.RAG.value:
                out = tools.run_rag(prompt)
                evidence = {"rag": out}

                if out.get("error"):
                    assistant_text = out["error"]
                else:
                    assistant_text = out.get("answer", "")

                with tab_answer:
                    st.markdown(assistant_text)

                with tab_evidence:
                    rag_out = evidence.get("rag", {}) or {}
                    st.write("Top score:", rag_out.get("top_score"))
                    chunks = rag_out.get("chunks", []) or []
                    if not chunks:
                        st.info("No chunks returned.")
                    for i, ch in enumerate(chunks, 1):
                        if isinstance(ch, dict):
                            cid = ch.get("chunk_id") or f"chunk_{i}"
                            score = ch.get("score")
                            text = ch.get("text") or ""
                            header = f"{i}. {cid}" + (f" (score={score})" if score is not None else "")
                            st.markdown(f"**{header}**")
                            st.markdown(text)
                        else:
                            st.markdown(str(ch))

            else:
                assistant_text = "For override mode, choose DATA or RAG."
                with tab_answer:
                    st.info(assistant_text)

            with tab_trace:
                st.json(
                    safe_json(
                        {
                            "override": True,
                            "forced_route": chosen,
                            "router_meta": router_meta,
                            "evidence_summary": evidence_summary(evidence),
                        }
                    )
                )

            # 🔥 Important: no DF inside chat history meta
            meta_safe = {
                "mode": chosen,
                "router_meta": router_meta,
                "evidence_summary": evidence_summary(evidence),
            }
            st.session_state.messages.append({"role": "assistant", "content": assistant_text, "meta": meta_safe})

        # --------------------------
        # AUTO (agent)
        # --------------------------
        else:
            state = run_agent(prompt)  # returns dict
            mode = state.get("mode", "general")
            answer = state.get("answer", "")
            router_meta = state.get("router_meta", {}) or {}
            evidence = state.get("evidence", {}) or {}
            trace_list = state.get("trace", []) or []

            with tab_answer:
                st.markdown(f"**Mode:** `{mode}`")
                if router_meta.get("router_explanation"):
                    st.caption(router_meta["router_explanation"])
                if router_meta.get("trace_path"):
                    st.caption(f"Trace saved: {router_meta['trace_path']}")
                st.markdown(answer)

            with tab_evidence:
                if mode == "data":
                    data_out = evidence.get("data", {}) or {}
                    dq = data_out.get("dq", {}) or {}
                    df_res = df_from_any(data_out.get("df"))

                    st.subheader("Parsed query (structured)")
                    st.json(safe_json(dq))

                    if df_res is not None:
                        render_peak_top5(dq, df_res)
                        st.subheader("Result table")
                        st.dataframe(df_res, use_container_width=True)
                        st.subheader("Plot")
                        plot_df(df_res, mode=str(dq.get("mode", "")))
                    else:
                        if data_out.get("error"):
                            st.error(data_out["error"])
                        if data_out.get("hint"):
                            st.info(data_out["hint"])
                        if not data_out.get("error") and not data_out.get("hint"):
                            st.info("No result table produced.")

                elif mode == "rag":
                    rag_out = evidence.get("rag", {}) or {}
                    st.write("Top score:", rag_out.get("top_score"))
                    chunks = rag_out.get("chunks", []) or []
                    if not chunks:
                        st.info("No chunks returned.")
                    for i, ch in enumerate(chunks, 1):
                        if isinstance(ch, dict):
                            cid = ch.get("chunk_id") or f"chunk_{i}"
                            score = ch.get("score")
                            text = ch.get("text") or ""
                            header = f"{i}. {cid}" + (f" (score={score})" if score is not None else "")
                            st.markdown(f"**{header}**")
                            st.markdown(text)
                        else:
                            st.markdown(str(ch))
                else:
                    st.info("No evidence to show for this mode.")

            with tab_trace:
                if trace_list:
                    rows = []
                    for t in trace_list:
                        if isinstance(t, dict):
                            rows.append({"node": t.get("node", ""), "ms": t.get("ms", None), "route": t.get("route", None)})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No trace available.")
                st.divider()
                st.subheader("Raw (safe)")
                st.json(
                    safe_json(
                        {
                            "mode": mode,
                            "route": state.get("route"),
                            "router_meta": router_meta,
                            "evidence_summary": evidence_summary(evidence),
                        }
                    )
                )

            meta_safe = {
                "mode": mode,
                "route": state.get("route"),
                "router_meta": router_meta,
                "evidence_summary": evidence_summary(evidence),
            }
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_safe})
