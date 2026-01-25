# src/data/nl_parser.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataQuery:
    # Core filter fields (specific to your store_sales dataset)
    store_nbr: Optional[int] = None
    family: Optional[str] = None
    start: Optional[str] = None  # YYYY-MM-DD
    end: Optional[str] = None    # YYYY-MM-DD

    # What to compute
    metric: str = "sales"
    agg: str = "sum"
    group_by: str = "date"  # daily aggregation by date

    # ✅ NEW: what output the UI should produce
    mode: str = "daily"  # daily | monthly | yearly | family

    # raw user query (for debugging)
    raw: str = ""


_FAMILY_PATTERNS = [
    r"family\s*[:=]\s*['\"]?([A-Za-z0-9 _\-&/]+)['\"]?",
    r"family\s+is\s+['\"]?([A-Za-z0-9 _\-&/]+)['\"]?",
    r"\bfamily\s+([A-Za-z0-9 _\-&/]+?)(?=\s+(?:in|from|to|between|and|start|end|on|during)\b|$)",
]

_STORE_PATTERNS = [
    r"store[_\s]?nbr\s*[:=]\s*(\d+)",
    r"\bstore\s+(\d+)\b",
]

_DATE_RANGE_PATTERNS = [
    r"start\s*[:=]\s*(\d{4}-\d{2}-\d{2}).*end\s*[:=]\s*(\d{4}-\d{2}-\d{2})",
    r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
    r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})",
]

_MONTHS = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

_MONTH_YEAR_RE = re.compile(r"\b(?:in\s+)?([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_first(patterns: list[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _extract_store(text: str) -> Optional[int]:
    for pat in _STORE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_family(text: str) -> Optional[str]:
    fam = _extract_first(_FAMILY_PATTERNS, text)
    if fam:
        return fam.strip().upper()

    m = re.search(r"['\"]([A-Za-z0-9 _\-&/]+)['\"]", text)
    if m:
        return m.group(1).strip().upper()

    return None


def _extract_date_range(text: str) -> tuple[Optional[str], Optional[str]]:
    for pat in _DATE_RANGE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1), m.group(2)

    m = _MONTH_YEAR_RE.search(text)
    if m:
        mon_raw = m.group(1).lower()
        year = m.group(2)
        mon = _MONTHS.get(mon_raw)
        if mon:
            start = f"{year}-{mon}-01"
            mon_i = int(mon)
            if mon_i == 12:
                end = f"{int(year)+1}-01-01"
            else:
                end = f"{year}-{mon_i+1:02d}-01"
            return start, end

    return None, None


def parse_data_query(user_text: str) -> DataQuery:
    text = user_text.strip()
    q_lower = text.lower()
    dq = DataQuery(raw=text)

    dq.store_nbr = _extract_store(text)
    dq.family = _extract_family(text)

    start, end = _extract_date_range(text)
    dq.start, dq.end = start, end

    # Infer year-only range if start/end missing
    year_match = _YEAR_RE.search(q_lower)
    if year_match and not dq.start and not dq.end:
        year = year_match.group()
        dq.start = f"{year}-01-01"
        dq.end = f"{year}-12-31"

    # agg
    if re.search(r"\bmean\b|\bavg\b|\baverage\b", q_lower):
        dq.agg = "mean"
    else:
        dq.agg = "sum"

    # metric
    if re.search(r"\bonpromotion\b|\bpromo\b", q_lower):
        dq.metric = "onpromotion"
    else:
        dq.metric = "sales"

    # group_by stays daily; UI will decide aggregation
    dq.group_by = "date"

        # --- MODE DETECTION (must be before return) ---
    wants_monthly = any(k in q_lower for k in ["month", "months", "monthly"])
    wants_yearly = any(k in q_lower for k in ["year", "years", "yearly", "annual"])
    wants_family = ("family" in q_lower or "families" in q_lower) and any(
        k in q_lower for k in ["top", "compare", "highest", "maximum", "max"]
    )

    if wants_monthly:
        dq.mode = "monthly"
    elif wants_yearly:
        dq.mode = "yearly"
    elif wants_family:
        dq.mode = "family"
    else:
        dq.mode = "daily"

    return dq
