# src/app/pages/traces.py
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

TRACE_DIR = os.path.join("data", "traces")

st.set_page_config(page_title="Traces", layout="wide")
st.title("Execution Traces")


def list_trace_files() -> List[str]:
    if not os.path.exists(TRACE_DIR):
        return []
    files = [f for f in os.listdir(TRACE_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files


def load_trace(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load trace JSON: {e}")
        return None


def safe_preview(text: Any, max_chars: int = 1200) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    return s if len(s) <= max_chars else s[:max_chars] + " ..."


files = list_trace_files()
if not files:
    st.info("No traces found yet. Ask a question in the Chat page to generate traces.")
    st.stop()

selected = st.selectbox("Select a trace file", files, index=0, key="traces_select")
path = os.path.join(TRACE_DIR, selected)

trace = load_trace(path)
if trace is None:
    st.stop()

question = trace.get("question", "")
route = trace.get("route", "")
mode = trace.get("mode", "")
answer = trace.get("answer", "")
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
    st.code(safe_preview(answer), language="text")

with col2:
    st.subheader("File")
    st.code(path)

st.divider()
st.subheader("Node timings")

rows = []
for t in trace_rows:
    if isinstance(t, dict):
        rows.append({"node": t.get("node", ""), "ms": t.get("ms", None), "route": t.get("route", None)})

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No node timing rows found.")

with st.expander("Router metadata", expanded=False):
    st.json(router_meta)

with st.expander("Evidence (raw)", expanded=False):
    st.json(evidence)

with st.expander("Raw trace JSON", expanded=False):
    st.json(trace)
