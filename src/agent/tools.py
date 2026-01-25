# src/agent/tools.py
import os
import re
import pandas as pd
from typing import Any, Dict, Optional, Tuple

from src.agent.router import RuleBasedRouter, Route

# --------------------------
# RAG
# --------------------------
try:
    from src.agent.rag_answerer import RAGAnswerer
except Exception:
    RAGAnswerer = None

# --------------------------
# DATA
# --------------------------
from src.data.nl_parser import parse_data_query
from src.data.loader import load_table
from src.data.schema import detect_schema
from src.data.filtering import filter_sales
from src.data.aggregate import (
    group_sum_by_date,
    group_sum_by_month,
    group_sum_by_year,
    group_sum,
)

DATASET_REL_PATH = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")
DATE_COL = "date"

# --------------------------
# Singletons (fast + simple)
# --------------------------
_ROUTER = RuleBasedRouter()
_RAG_ANSWERER = None
_DF_CACHE = None


def get_router() -> RuleBasedRouter:
    return _ROUTER


def get_rag_answerer() -> Any:
    global _RAG_ANSWERER
    if _RAG_ANSWERER is not None:
        return _RAG_ANSWERER
    if RAGAnswerer is None:
        return None
    _RAG_ANSWERER = RAGAnswerer()
    return _RAG_ANSWERER


def load_sales_df() -> Any:
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE
    _DF_CACHE = load_table(DATASET_REL_PATH)
    return _DF_CACHE


# --------------------------
# Deterministic date overrides
# --------------------------
_YEAR_RANGE_PATTERNS = [
    re.compile(r"\bfrom\s+(?P<y1>\d{4})\s+to\s+(?P<y2>\d{4})\b", re.IGNORECASE),
    re.compile(r"\bbetween\s+(?P<y1>\d{4})\s+and\s+(?P<y2>\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(?P<y1>\d{4})\s*-\s*(?P<y2>\d{4})\b", re.IGNORECASE),
]

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_TOKEN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Most specific first
_MONTH_RANGE_PATTERNS = [
    # "2013 may to october"
    re.compile(
        rf"\b(?P<y>\d{{4}})\b\s+(?P<m1>{_MONTH_TOKEN})\b\s*"
        rf"(?:to|\-|through|thru|until)\s*(?P<m2>{_MONTH_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "between 2013 may and october"
    re.compile(
        rf"\bbetween\s+(?P<y>\d{{4}})\s+(?P<m1>{_MONTH_TOKEN})\b\s+and\s+(?P<m2>{_MONTH_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "may to october 2013" / "may to october in 2013"
    re.compile(
        rf"\b(?P<m1>{_MONTH_TOKEN})\b\s*(?:to|\-|through|thru|until)\s*(?P<m2>{_MONTH_TOKEN})\b.*?\b(?P<y>\d{{4}})\b",
        re.IGNORECASE,
    ),
    # "between may and october 2013"
    re.compile(
        rf"\bbetween\s+(?P<m1>{_MONTH_TOKEN})\b\s+and\s+(?P<m2>{_MONTH_TOKEN})\b.*?\b(?P<y>\d{{4}})\b",
        re.IGNORECASE,
    ),
]


def extract_year_range(q: str) -> Optional[Tuple[str, str]]:
    for pat in _YEAR_RANGE_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        y1 = int(m.group("y1"))
        y2 = int(m.group("y2"))
        if y2 < y1:
            y1, y2 = y2, y1
        return (f"{y1}-01-01", f"{y2}-12-31")
    return None


def _month_num(m: str) -> int:
    return _MONTHS[m.strip().lower()]


def _last_day_of_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def extract_month_range(q: str) -> Optional[Tuple[str, str]]:
    """
    Supports:
    - '2013 may to october'
    - 'between 2013 may and october'
    - 'may to october 2013' / 'may to october in 2013'
    - 'between may and october 2013'
    Returns ('YYYY-MM-01', 'YYYY-MM-lastday')
    """
    for pat in _MONTH_RANGE_PATTERNS:
        m = pat.search(q)
        if not m:
            continue

        y = int(m.group("y"))
        m1 = _month_num(m.group("m1"))
        m2 = _month_num(m.group("m2"))

        if m2 < m1:
            m1, m2 = m2, m1

        start = f"{y}-{m1:02d}-01"
        end_day = _last_day_of_month(y, m2)
        end = f"{y}-{m2:02d}-{end_day:02d}"
        return (start, end)

    return None


def enforce_inclusive_date_range(df: Any, start: Optional[str], end: Optional[str]) -> Any:
    if df is None or DATE_COL not in df.columns:
        return df

    d = df.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")

    if start:
        d = d[d[DATE_COL] >= pd.to_datetime(start)]
    if end:
        d = d[d[DATE_COL] <= pd.to_datetime(end)]

    return d


def normalize_nulls(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str) and v.strip().upper() == "NULL":
            out[k] = None
        else:
            out[k] = v
    return out


def months_present_without_year(question: str, dq: Any) -> bool:
    q = question.lower()
    has_month = any(m in q for m in _MONTHS.keys())
    has_year = re.search(r"\b(19|20)\d{2}\b", question) is not None

    start = getattr(dq, "start", None)
    end = getattr(dq, "end", None)

    start_missing = (start is None) or (isinstance(start, str) and start.strip().upper() == "NULL")
    end_missing = (end is None) or (isinstance(end, str) and end.strip().upper() == "NULL")

    return has_month and (not has_year) and start_missing and end_missing


# --------------------------
# Tools
# --------------------------
def route_question(question: str) -> Dict[str, Any]:
    rr = get_router().route(question)
    return {
        "route": rr.route.value,
        "router_explanation": get_router().explain_route(rr),
        "router_scores": {k.value: v for k, v in rr.scores.items()},
        "router_matched": {
            Route.DATA.value: rr.matched.get(Route.DATA, []),
            Route.RAG.value: rr.matched.get(Route.RAG, []),
        },
    }


def run_rag(question: str) -> Dict[str, Any]:
    ans = get_rag_answerer()
    if ans is None:
        return {"error": "RAGAnswerer not available."}

    out = ans.answer(question)

    if isinstance(out, dict):
        return {
            "answer": out.get("answer", ""),
            "mode": out.get("mode", "rag"),
            "top_score": out.get("top_score", None),
            "chunks": out.get("chunks", []),
        }

    return {"answer": str(out), "mode": "rag", "top_score": None, "chunks": []}


def run_data(question: str) -> Dict[str, Any]:
    dq = parse_data_query(question)

    # Deterministic overrides
    mr = extract_month_range(question)
    if mr is not None:
        dq.start, dq.end = mr[0], mr[1]
    else:
        yr = extract_year_range(question)
        if yr is not None:
            dq.start, dq.end = yr[0], yr[1]

    # If months are present but no year, do not guess
    if months_present_without_year(question, dq):
        return {
            "error": "Month range detected but no year provided. Please include a year (e.g., 'store 2 sales Jan to Mar 2013')."
        }

    df = load_sales_df()
    if df is None:
        return {"error": f"Dataset load failed. Expected: {DATASET_REL_PATH}"}

    schema = detect_schema(df)

    df_filt = filter_sales(
        df,
        store_nbr=getattr(dq, "store_nbr", None),
        family=getattr(dq, "family", None),
        start=getattr(dq, "start", None),
        end=getattr(dq, "end", None),
    )
    if df_filt is None:
        return {"error": "filter_sales() returned None."}

    df_filt = enforce_inclusive_date_range(df_filt, getattr(dq, "start", None), getattr(dq, "end", None))

    metric = getattr(dq, "metric", "sales")
    if metric not in df_filt.columns:
        return {"error": f"Metric '{metric}' not found. Available: {list(df_filt.columns)}"}

    mode = getattr(dq, "mode", "daily")

    if mode == "daily":
        df_out = group_sum_by_date(df_filt, date_col=DATE_COL, value_col=metric)
    elif mode == "monthly":
        df_out = group_sum_by_month(df_filt, date_col=DATE_COL, value_col=metric)
    elif mode == "yearly":
        df_out = group_sum_by_year(df_filt, date_col=DATE_COL, value_col=metric)
    elif mode == "family":
        df_out = group_sum(df_filt, group_col="family", value_col=metric)
    else:
        df_out = group_sum_by_date(df_filt, date_col=DATE_COL, value_col=metric)

    # UX label cleanup (safe)
    if df_out is not None and not df_out.empty:
        time_col = df_out.columns[0]
        if mode == "monthly":
            df_out = df_out.copy()
            df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce").dt.to_period("M").astype(str)
        elif mode == "yearly":
            df_out = df_out.copy()
            df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce").dt.year.astype(str)

    dq_dict = dq.__dict__.copy() if hasattr(dq, "__dict__") else {"raw": str(dq)}
    dq_dict = normalize_nulls(dq_dict)

    return {"dq": dq_dict, "schema": schema, "df": df_out}
