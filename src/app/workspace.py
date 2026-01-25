# src/app/workspace.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

UPLOAD_DATASETS_DIR = os.path.join("data", "uploads", "datasets")
UPLOAD_DOCS_DIR = os.path.join("data", "uploads", "docs")


def init_workspace() -> None:
    """One shared state object for uploads + active selections."""
    os.makedirs(UPLOAD_DATASETS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DOCS_DIR, exist_ok=True)

    if "workspace" not in st.session_state:
        st.session_state["workspace"] = {
            # name -> {"type": "csv", "df": DataFrame, "path": str}
            "datasets": {},
            "active_dataset": None,
            # future: docs
            "docs": {},
            "active_docs_scope": "uploads",
        }


def ws() -> Dict[str, Any]:
    init_workspace()
    return st.session_state["workspace"]


def add_dataset(name: str, df: pd.DataFrame, path: Optional[str] = None) -> None:
    w = ws()
    w["datasets"][name] = {"type": "csv", "df": df, "path": path or ""}
    w["active_dataset"] = name


def get_dataset(name: str) -> Optional[Dict[str, Any]]:
    w = ws()
    return w.get("datasets", {}).get(name)


def list_datasets() -> Dict[str, Dict[str, Any]]:
    w = ws()
    return w.get("datasets", {}) or {}
