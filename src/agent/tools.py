# src/agent/tools.py
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.agent.router import RuleBasedRouter, Route

# --- RAG ---
try:
    from src.agent.rag_answerer import RAGAnswerer
except Exception:
    RAGAnswerer = None

# --- existing store-sales data pipeline ---
from src.data.loader import load_table
from src.data.schema import detect_schema
from src.data.filtering import filter_sales
from src.data.aggregate import (
    group_sum_by_date,
    group_sum_by_month,
    group_sum_by_year,
    group_sum,
)
from src.data.nl_parser import parse_data_query

_DEFAULT_DATASET = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")
_ROUTER = RuleBasedRouter()

# Optional global active df (legacy support)
_ACTIVE_DF: Optional[pd.DataFrame] = None
_ACTIVE_NAME: str = "default.csv"
_ACTIVE_DATE_COL: Optional[str] = None
_ACTIVE_METRIC_COL: Optional[str] = None
_ACTIVE_GROUP_COL: Optional[str] = None


# --------------------------
# Normalization (tests + Streamlit stability)
# --------------------------
def normalize_nulls(obj: Any) -> Any:
    """
    Test/Streamlit stable JSON normalization.

    - Converts string "NULL"/"null"/"None"/"" -> None
    - Recursively normalizes dicts/lists
    """
    if obj is None:
        return None

    # Convert common null strings to real None
    if isinstance(obj, str):
        s = obj.strip()
        if s == "":
            return None
        if s.lower() in ("null", "none", "nan"):
            return None
        if s.upper() == "NULL":
            return None
        return obj

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): normalize_nulls(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [normalize_nulls(x) for x in obj]

    if hasattr(obj, "__dict__"):
        return normalize_nulls(obj.__dict__)

    return str(obj)


# --------------------------
# Upload support (optional legacy)
# --------------------------
def set_active_df(df: pd.DataFrame, name: str = "uploaded.csv") -> None:
    global _ACTIVE_DF, _ACTIVE_NAME
    _ACTIVE_DF = df
    _ACTIVE_NAME = name


def clear_active_df() -> None:
    global _ACTIVE_DF, _ACTIVE_NAME
    _ACTIVE_DF = None
    _ACTIVE_NAME = "default.csv"


def set_active_mapping(
    date_col: Optional[str],
    metric_col: Optional[str],
    group_col: Optional[str],
) -> None:
    global _ACTIVE_DATE_COL, _ACTIVE_METRIC_COL, _ACTIVE_GROUP_COL
    _ACTIVE_DATE_COL = date_col
    _ACTIVE_METRIC_COL = metric_col
    _ACTIVE_GROUP_COL = group_col


# --------------------------
# Router meta (used by graph/chat)
# --------------------------
def route_question(question: str) -> Dict[str, Any]:
    rr = _ROUTER.route(question)
    return {
        "route": rr.route.value,
        "router_explanation": _ROUTER.explain_route(rr),
        "router_scores": {k.value: v for k, v in rr.scores.items()},
        "router_matched": {
            Route.DATA.value: rr.matched.get(Route.DATA, []),
            Route.RAG.value: rr.matched.get(Route.RAG, []),
        },
    }


# --------------------------
# RAG
# --------------------------
_RAG_INSTANCE: Any = None


def _get_rag() -> Any:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is not None:
        return _RAG_INSTANCE
    if RAGAnswerer is None:
        return None
    _RAG_INSTANCE = RAGAnswerer()
    return _RAG_INSTANCE


def _normalize_rag_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        answer = out.get("answer") or out.get("response") or out.get("text") or ""
        chunks = out.get("chunks") or out.get("evidence") or out.get("contexts") or []
        norm_chunks = []
        for i, ch in enumerate(chunks, 1):
            if isinstance(ch, dict):
                norm_chunks.append(
                    {
                        "chunk_id": ch.get("chunk_id") or ch.get("id") or f"chunk_{i}",
                        "score": ch.get("score"),
                        "text": ch.get("text") or ch.get("content") or ch.get("chunk") or "",
                    }
                )
            else:
                norm_chunks.append({"chunk_id": f"chunk_{i}", "score": None, "text": str(ch)})
        return {
            "answer": answer,
            "mode": out.get("mode", "rag"),
            "top_score": out.get("top_score"),
            "chunks": norm_chunks,
        }

    return {"answer": str(out), "mode": "rag", "top_score": None, "chunks": []}


def run_rag(question: str) -> Dict[str, Any]:
    rag = _get_rag()
    if rag is None:
        return {"error": "RAGAnswerer not available (import failed)."}
    try:
        out = rag.answer(question)
        return _normalize_rag_output(out)
    except Exception as e:
        return {"error": str(e)}


# --------------------------
# DATA helpers
# --------------------------
_YEAR_RANGE_PATTERNS = [
    re.compile(r"\bfrom\s+(?P<y1>\d{4})\s+to\s+(?P<y2>\d{4})\b", re.IGNORECASE),
    re.compile(r"\bbetween\s+(?P<y1>\d{4})\s+and\s+(?P<y2>\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(?P<y1>\d{4})\s*-\s*(?P<y2>\d{4})\b", re.IGNORECASE),
]

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_year_range(q: str) -> Optional[Tuple[str, str]]:
    """
    Public helper used by tests.
    Returns ('YYYY-01-01', 'YYYY-12-31') for patterns:
      - from 2014 to 2016
      - between 2014 and 2016
      - 2014-2016
    """
    for pat in _YEAR_RANGE_PATTERNS:
        m = pat.search(q)
        if m:
            y1 = int(m.group("y1"))
            y2 = int(m.group("y2"))
            if y2 < y1:
                y1, y2 = y2, y1
            return (f"{y1}-01-01", f"{y2}-12-31")
    return None


def _extract_month_range_with_year(q: str) -> Optional[Tuple[str, str]]:
    """
    Examples:
      - "january to march in 2013"
      - "2013 jan to mar"
    Returns inclusive YYYY-MM-DD start/end.
    """
    s = q.lower()
    m_year = re.search(r"\b(20\d{2})\b", s)
    if not m_year:
        return None
    year = int(m_year.group(1))

    # Find all month mentions by position, keep the first two
    months_found = []
    for token, num in _MONTHS.items():
        m = re.search(rf"\b{re.escape(token)}\b", s)
        if m:
            months_found.append((m.start(), num))
    months_found.sort(key=lambda x: x[0])
    months = [m[1] for m in months_found]

    if len(months) < 2:
        return None

    m1, m2 = months[0], months[1]
    if m2 < m1:
        m1, m2 = m2, m1

    start = f"{year}-{m1:02d}-01"
    end_dt = pd.Timestamp(year=year, month=m2, day=1) + pd.offsets.MonthEnd(0)
    end = end_dt.strftime("%Y-%m-%d")
    return start, end


def extract_month_range(q: str) -> Optional[Tuple[str, str]]:
    """
    Public helper used by tests.
    Requires an explicit year in the question.
    """
    return _extract_month_range_with_year(q)


def months_present_without_year(q: str, dq: Optional[Any] = None) -> bool:
    """
    Public helper used by tests.

    True if:
      - question mentions >= 2 months, AND
      - no explicit year in the question, AND
      - dq.start/dq.end are not already set (if dq provided)
    """
    s = (q or "").lower()
    has_year = re.search(r"\b(20\d{2})\b", s) is not None

    # count month mentions
    month_hits = 0
    for token in _MONTHS.keys():
        if re.search(rf"\b{re.escape(token)}\b", s):
            month_hits += 1
        if month_hits >= 2:
            break

    if month_hits < 2:
        return False
    if has_year:
        return False

    # If dq is provided and already has a parsed range, do NOT flag as ambiguous
    if dq is not None:
        start = getattr(dq, "start", None)
        end = getattr(dq, "end", None)
        if start is not None or end is not None:
            return False

    return True


def _enforce_inclusive_date_range(
    df: pd.DataFrame, date_col: str, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if start:
        d = d[d[date_col] >= pd.to_datetime(start)]
    if end:
        d = d[d[date_col] <= pd.to_datetime(end)]
    return d


def _looks_like_store_sales(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return {"date", "sales"}.issubset(cols)


def _load_dataset(dataset: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, str]:
    """
    dataset can be:
      - None -> default CSV
      - {"df": DataFrame, "name": "..."}
      - {"path": "...", "name": "..."}
    Also supports legacy global _ACTIVE_DF if set.
    """
    if isinstance(dataset, dict):
        if isinstance(dataset.get("df"), pd.DataFrame):
            name = str(dataset.get("name") or "uploaded.csv")
            return dataset["df"].copy(), name
        if dataset.get("path"):
            path = str(dataset["path"])
            df = load_table(path)
            if df is None:
                raise ValueError(f"Dataset failed to load: {path}")
            name = str(dataset.get("name") or os.path.basename(path))
            return df, name

    if _ACTIVE_DF is not None:
        return _ACTIVE_DF.copy(), _ACTIVE_NAME

    df = load_table(_DEFAULT_DATASET)
    if df is None:
        raise ValueError(f"Default dataset failed to load: {_DEFAULT_DATASET}")
    return df, os.path.basename(_DEFAULT_DATASET)


# --------------------------
# Generic fallback for unknown CSVs
# --------------------------
def _generic_choose_metric(df: pd.DataFrame) -> Optional[str]:
    if _ACTIVE_METRIC_COL and _ACTIVE_METRIC_COL in df.columns:
        return _ACTIVE_METRIC_COL
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _generic_choose_date(df: pd.DataFrame) -> Optional[str]:
    if _ACTIVE_DATE_COL and _ACTIVE_DATE_COL in df.columns:
        return _ACTIVE_DATE_COL

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    for c in df.columns:
        s = df[c].dropna().astype(str).head(50)
        if s.empty:
            continue
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return c

    return None


def _generic_choose_group(df: pd.DataFrame) -> Optional[str]:
    if _ACTIVE_GROUP_COL and _ACTIVE_GROUP_COL in df.columns:
        return _ACTIVE_GROUP_COL
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 50:
                return c
    return None


def _generic_op_from_question(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["average", "avg", "mean"]):
        return "mean"
    if "median" in q:
        return "median"
    if any(k in q for k in ["max", "maximum", "highest", "peak"]):
        return "max"
    if any(k in q for k in ["min", "minimum", "lowest"]):
        return "min"
    return "sum"


def _generic_aggregate(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    metric = _generic_choose_metric(df)
    if metric is None:
        return {
            "error": "No numeric columns found in this dataset.",
            "hint": "Upload a dataset with numeric columns (or map a numeric metric column) and ask again.",
            "dq": {"mode": "generic"},
            "df": None,
            "summary": "No numeric metric available.",
        }

    date_col = _generic_choose_date(df)
    group_col = _generic_choose_group(df)
    op = _generic_op_from_question(question)

    work = df.copy()
    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    if group_col:
        gb = work.groupby(group_col, dropna=False)[metric]
        series = getattr(gb, op)() if hasattr(gb, op) else gb.sum()
        out_df = series.reset_index().rename(columns={metric: op})
        return {
            "dq": {
                "mode": "generic_group",
                "metric": metric,
                "date_col": date_col,
                "group_col": group_col,
                "agg": op,
            },
            "df": out_df,
            "summary": f"Computed {op} of {metric} grouped by {group_col}.",
        }

    s = work[metric]
    val = getattr(s, op)() if hasattr(s, op) else s.sum()
    out_df = pd.DataFrame({"metric": [metric], op: [val]})
    return {
        "dq": {"mode": "generic", "metric": metric, "date_col": date_col, "group_col": None, "agg": op},
        "df": out_df,
        "summary": f"Computed {op} of {metric}.",
    }


# --------------------------
# Public DATA tool
# --------------------------
def run_data(question: str, dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q_low = str(question or "").strip().lower()

    try:
        df, name = _load_dataset(dataset)
    except Exception as e:
        return {"error": str(e), "dq": {}, "df": None, "summary": "Dataset load failed.", "dataset": None}

    # ---- Store sales pipeline ----
    if _looks_like_store_sales(df):
        try:
            dq = parse_data_query(question)

            # If user mentions months but did NOT include a year and parser couldn't infer dates -> force year
            has_month = any(re.search(rf"\b{re.escape(m)}\b", q_low) for m in _MONTHS.keys())
            has_year = re.search(r"\b(20\d{2})\b", q_low) is not None
            dq_start = getattr(dq, "start", None)
            dq_end = getattr(dq, "end", None)

            if has_month and (not has_year) and (dq_start is None) and (dq_end is None):
                return {
                    "error": "Month range detected but no year",
                    "hint": "Please include a year, e.g. 'sales in store 2 from january to march in 2013'.",
                    "dq": normalize_nulls(asdict(dq)),
                    "df": None,
                    "summary": "Missing year for month range.",
                    "dataset": name,
                }

            # Month-range override
            mr = extract_month_range(question)
            if mr is not None:
                dq.start, dq.end = mr[0], mr[1]

            # Year-range override
            yr = extract_year_range(question)
            if yr is not None:
                dq.start, dq.end = yr[0], yr[1]

            schema = detect_schema(df)

            df_filt = filter_sales(
                df,
                store_nbr=getattr(dq, "store_nbr", None),
                family=getattr(dq, "family", None),
                start=getattr(dq, "start", None),
                end=getattr(dq, "end", None),
            )

            if df_filt is None:
                return {
                    "error": "filter_sales() returned None",
                    "dq": normalize_nulls(asdict(dq)),
                    "df": None,
                    "summary": "Filter failed.",
                    "dataset": name,
                }

            df_filt = _enforce_inclusive_date_range(df_filt, "date", dq.start, dq.end)

            metric = getattr(dq, "metric", "sales")
            if metric not in df_filt.columns:
                return {
                    "error": f"Metric '{metric}' not found. Available: {list(df_filt.columns)}",
                    "hint": "Try 'sales' or 'onpromotion' for the store sales dataset.",
                    "dq": normalize_nulls(asdict(dq)),
                    "df": None,
                    "summary": "Metric missing.",
                    "dataset": name,
                }

            mode = getattr(dq, "mode", "daily")

            if mode == "daily":
                df_out = group_sum_by_date(df_filt, date_col="date", value_col=metric)
            elif mode == "monthly":
                df_out = group_sum_by_month(df_filt, date_col="date", value_col=metric)
            elif mode == "yearly":
                df_out = group_sum_by_year(df_filt, date_col="date", value_col=metric)
            elif mode == "family":
                df_out = group_sum(df_filt, group_col="family", value_col=metric)
            else:
                df_out = group_sum_by_date(df_filt, date_col="date", value_col=metric)

            # Clean monthly/yearly labels
            if df_out is not None and not df_out.empty and mode in ("monthly", "yearly"):
                time_col = df_out.columns[0]
                df_out = df_out.copy()
                if mode == "monthly":
                    df_out[time_col] = (
                        pd.to_datetime(df_out[time_col], errors="coerce")
                        .dt.to_period("M")
                        .astype(str)
                    )
                else:
                    df_out[time_col] = (
                        pd.to_datetime(df_out[time_col], errors="coerce")
                        .dt.year
                        .astype(str)
                    )

            return {
                "dq": normalize_nulls(asdict(dq)),
                "schema": schema,
                "df": df_out,
                "summary": "Computed result from dataset.",
                "dataset": name,
            }

        except Exception as e:
            return {
                "error": str(e),
                "dq": normalize_nulls({}),
                "df": None,
                "summary": "Data pipeline error.",
                "dataset": name,
            }

    # ---- Generic fallback (unknown CSV schema) ----
    try:
        out = _generic_aggregate(df, question)
        out["schema"] = detect_schema(df) if hasattr(df, "columns") else {}
        out["dataset"] = name
        out["dq"] = normalize_nulls(out.get("dq", {}))
        return out
    except Exception as e:
        return {
            "error": str(e),
            "dq": normalize_nulls({}),
            "df": None,
            "summary": "Generic data pipeline error.",
            "dataset": name,
        }


# --------------------------
# Trace-safe evidence summarizer (graph.py uses this)
# --------------------------
def summarize_evidence_for_trace(evidence: Any) -> Dict[str, Any]:
    """
    Prevent huge trace JSON and non-serializable objects (DataFrames).
    """
    if not isinstance(evidence, dict):
        return {"type": str(type(evidence)), "preview": str(evidence)[:500]}

    out: Dict[str, Any] = {"keys": list(evidence.keys())}
    for k, v in evidence.items():
        if isinstance(v, dict):
            block: Dict[str, Any] = {"keys": list(v.keys())}
            df = v.get("df")
            if isinstance(df, pd.DataFrame):
                block["df_shape"] = list(df.shape)
                block["df_columns"] = list(df.columns)[:50]
            out[k] = block
        else:
            out[k] = {"type": str(type(v)), "preview": str(v)[:500]}
    return out
