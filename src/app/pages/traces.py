# src/app/pages/traces.py
import json
import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

TRACE_DIR = os.path.join("data", "traces")

st.set_page_config(page_title="Traces", layout="wide")
st.title("Execution Traces")


def list_trace_files() -> List[str]:
    if not os.path.exists(TRACE_DIR):
        return []
    files = [f for f in os.listdir(TRACE_DIR) if f.endswith(".json")]
    files.sort(reverse=True)  # newest first
    return files


def load_trace(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


files = list_trace_files()

if not files:
    st.info("No traces found yet. Run a question in the chat page to generate traces.")
    st.stop()

selected = st.selectbox("Select a trace file", files, index=0)
path = os.path.join(TRACE_DIR, selected)

trace = load_trace(path)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Summary")
    st.write("**Question:**", trace.get("question", ""))
    st.write("**Route:**", trace.get("route", ""))
    st.write("**Mode:**", trace.get("mode", ""))
    st.write("**Answer preview:**")
    st.code(trace.get("answer_preview", ""), language="text")

with col2:
    st.subheader("File")
    st.code(path)

st.divider()

st.subheader("Node timings")
rows = []
for t in trace.get("trace", []) or []:
    if isinstance(t, dict):
        rows.append({"node": t.get("node", ""), "ms": t.get("ms", None)})

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No node timing rows found.")

st.divider()

with st.expander("Router metadata", expanded=False):
    st.json(trace.get("router_meta", {}))

with st.expander("Evidence summary", expanded=False):
    st.json(trace.get("evidence", {}))

with st.expander("Raw trace JSON", expanded=False):
    st.json(trace)
