# src/app/pages/traces.py
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

TRACE_DIR = os.path.join("data", "traces")

st.set_page_config(page_title="Traces", layout="wide")
st.title("Execution Traces")


# --------------------------
# Helpers
# --------------------------
def list_trace_files() -> List[str]:
    if not os.path.exists(TRACE_DIR):
        return []
    files = [f for f in os.listdir(TRACE_DIR) if f.endswith(".json")]
    files.sort(reverse=True)  # newest first
    return files


def load_trace(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load trace JSON: {e}")
        return None


def safe_preview(text: Any, max_chars: int = 800) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " ..."


def summarize_evidence(evidence: Any) -> Dict[str, Any]:
    """
    Avoid dumping huge payloads; show compact info.
    """
    if not isinstance(evidence, dict):
        return {"type": str(type(evidence)), "value": safe_preview(evidence, 300)}

    out: Dict[str, Any] = {"keys": list(evidence.keys())}
    for k, v in evidence.items():
        if isinstance(v, dict):
            out[k] = {"keys": list(v.keys())}
            # common DF holder in your tools is 'df'
            df = v.get("df")
            if isinstance(df, pd.DataFrame):
                out[k]["df_shape"] = list(df.shape)
                out[k]["df_columns"] = list(df.columns)[:25]
            else:
                # if df serialized
                if df is not None and not isinstance(df, (str, int, float, bool, list, dict)):
                    out[k]["df_type"] = str(type(df))
        else:
            out[k] = {"type": str(type(v)), "preview": safe_preview(v, 200)}
    return out


# --------------------------
# UI
# --------------------------
if st.button("Refresh traces", key="traces_refresh_btn"):
    st.rerun()

files = list_trace_files()

if not files:
    st.info("No traces found yet. Ask a question in the Chat page to generate trace files.")
    st.stop()

query = st.text_input("Filter by filename or question text (optional)", value="", key="trace_filter")

# Filter list (by filename only — question filter requires loading files, expensive)
filtered_files = [f for f in files if query.lower() in f.lower()] if query else files

selected = st.selectbox("Select a trace file", filtered_files, index=0, key="trace_select")
path = os.path.join(TRACE_DIR, selected)

trace = load_trace(path)
if trace is None:
    st.stop()

# Basic fields
question = trace.get("question", "")
route = trace.get("route", "")
mode = trace.get("mode", "")
answer = trace.get("answer", "")  # ✅ correct key used by graph.py
router_meta = trace.get("router_meta", {}) or {}
trace_rows = trace.get("trace", []) or []
evidence = trace.get("evidence", {}) or {}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Summary")
    st.write("**Question:**", question)
    st.write("**Route:**", route)
    st.write("**Mode:**", mode)

    st.write("**Answer preview:**")
    st.code(safe_preview(answer, 1200), language="text")

with col2:
    st.subheader("File")
    st.code(path)

    st.write("**Trace timestamp (from filename):**")
    st.caption(selected.replace("trace-", "").replace(".json", ""))

st.divider()

# Node timings
st.subheader("Node timings")
rows = []
for t in trace_rows:
    if isinstance(t, dict):
        rows.append(
            {
                "node": t.get("node", ""),
                "ms": t.get("ms", None),
                "route": t.get("route", None),
            }
        )

if rows:
    df = pd.DataFrame(rows)
    # keep only non-empty cols
    df = df[[c for c in df.columns if df[c].notna().any()]]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No node timing rows found.")

st.divider()

with st.expander("Router metadata", expanded=False):
    st.json(router_meta)

with st.expander("Evidence summary", expanded=False):
    st.json(summarize_evidence(evidence))

with st.expander("Raw trace JSON", expanded=False):
    st.json(trace)
