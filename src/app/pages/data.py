# src/app/pages/data.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.agent import tools
from src.app.workspace import ws
from src.data.loader import load_table
from src.data.schema import detect_schema

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")
st.caption("Explore default datasets and uploaded CSVs (shared from Chat).")

DEFAULT_DIR = Path("data/datasets/tabular")


def _load_default_csv(path: Path) -> pd.DataFrame:
    df = load_table(str(path))
    if df is None:
        raise ValueError(f"Failed to load: {path}")
    return df


def _guess_date_col(df: pd.DataFrame) -> Optional[str]:
    # already datetime?
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try parseable
    for c in df.columns:
        s = df[c].dropna().astype(str).head(50)
        if s.empty:
            continue
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return c
    return None


def _guess_numeric_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _guess_group_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            n = df[c].nunique(dropna=True)
            if 2 <= n <= 50:
                return c
    return None


w = ws()

# Build dataset menu: defaults + uploads
default_csvs = sorted(DEFAULT_DIR.glob("*.csv"))
uploaded = w.get("datasets", {}) or {}

options: list[Tuple[str, str]] = []  # (kind, name)
for p in default_csvs:
    options.append(("default", p.name))
for name in uploaded.keys():
    options.append(("uploaded", name))

if not options:
    st.warning("No datasets found. Upload a CSV from the Chat page.")
    st.stop()

# Select active dataset with a stable default
active_name = w.get("active_dataset")
default_index = 0
if active_name:
    for i, (_, nm) in enumerate(options):
        if nm == active_name:
            default_index = i
            break

kind, name = st.selectbox(
    "Select a dataset",
    options=options,
    index=default_index,
    format_func=lambda x: ("Uploaded: " if x[0] == "uploaded" else "Default: ") + x[1],
    key="data_dataset_select",
)

# Load df
df: pd.DataFrame
dataset_label: str

if kind == "uploaded":
    rec = uploaded.get(name)
    if not rec or not isinstance(rec.get("df"), pd.DataFrame):
        st.error("Uploaded dataset not found in workspace. Re-upload it from Chat.")
        st.stop()
    df = rec["df"].copy()
    dataset_label = f"Uploaded: {name}"
else:
    df = _load_default_csv(DEFAULT_DIR / name)
    dataset_label = f"Default: {name}"

# Set as active for agent/tools
w["active_dataset"] = name
tools.set_active_df(df, name=name)

st.success(f"Using: {dataset_label} ({df.shape[0]} rows, {df.shape[1]} cols)")

# Preview
with st.expander("Preview", expanded=True):
    st.dataframe(df.head(200), use_container_width=True)

# Schema
schema = detect_schema(df)
with st.expander("Detected schema", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Numeric**")
        st.write(schema.get("numeric", []))
    with c2:
        st.markdown("**Categorical**")
        st.write(schema.get("categorical", []))
    with c3:
        st.markdown("**Datetime**")
        st.write(schema.get("datetime", []))

st.divider()
st.subheader("Quick visualization")

date_guess = _guess_date_col(df)
metric_guess = _guess_numeric_col(df)
group_guess = _guess_group_col(df)

cols = list(df.columns)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    date_col = st.selectbox("Date column (optional)", options=["(none)"] + cols, index=(cols.index(date_guess) + 1) if date_guess in cols else 0, key="viz_date_col")
with colB:
    metric_col = st.selectbox("Metric (numeric)", options=cols, index=cols.index(metric_guess) if metric_guess in cols else 0, key="viz_metric_col")
with colC:
    group_col = st.selectbox("Group (optional)", options=["(none)"] + cols, index=(cols.index(group_guess) + 1) if group_guess in cols else 0, key="viz_group_col")

# Save mapping for generic DATA tool choices
tools.set_active_mapping(
    date_col=None if date_col == "(none)" else date_col,
    metric_col=metric_col,
    group_col=None if group_col == "(none)" else group_col,
)

agg = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], index=0, key="viz_agg")
chart = st.selectbox("Chart", ["line", "bar"], index=0, key="viz_chart")

work = df.copy()

# Parse/clean date column if selected
if date_col != "(none)":
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

# If group selected
if group_col != "(none)":
    if agg == "sum":
        out = work.groupby(group_col, dropna=False)[metric_col].sum().reset_index()
    elif agg == "mean":
        out = work.groupby(group_col, dropna=False)[metric_col].mean().reset_index()
    elif agg == "median":
        out = work.groupby(group_col, dropna=False)[metric_col].median().reset_index()
    elif agg == "max":
        out = work.groupby(group_col, dropna=False)[metric_col].max().reset_index()
    else:
        out = work.groupby(group_col, dropna=False)[metric_col].min().reset_index()

    out = out.sort_values(metric_col, ascending=False)

    st.write("Result table")
    st.dataframe(out.head(200), use_container_width=True)

    if chart == "bar":
        st.bar_chart(out.set_index(group_col)[metric_col].head(50))
    else:
        st.line_chart(out.set_index(group_col)[metric_col].head(50))

else:
    # If date selected -> time series
    if date_col != "(none)":
        # daily-ish aggregation by date
        if agg == "sum":
            out = work.groupby(date_col)[metric_col].sum().reset_index()
        elif agg == "mean":
            out = work.groupby(date_col)[metric_col].mean().reset_index()
        elif agg == "median":
            out = work.groupby(date_col)[metric_col].median().reset_index()
        elif agg == "max":
            out = work.groupby(date_col)[metric_col].max().reset_index()
        else:
            out = work.groupby(date_col)[metric_col].min().reset_index()

        out = out.sort_values(date_col)
        st.write("Result table")
        st.dataframe(out.head(400), use_container_width=True)

        if chart == "bar":
            st.bar_chart(out.set_index(date_col)[metric_col])
        else:
            st.line_chart(out.set_index(date_col)[metric_col])
    else:
        # overall scalar
        series = work[metric_col]
        val = getattr(series, agg)() if hasattr(series, agg) else series.sum()
        st.metric(f"{agg}({metric_col})", f"{float(val):,.4f}")
