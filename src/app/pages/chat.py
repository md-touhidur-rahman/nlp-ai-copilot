# src/app/pages/chat.py
import os
import re
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from src.agent.router import RuleBasedRouter, Route

# --------------------------
# RAG (your repo layout)
# --------------------------
try:
    from src.agent.rag_answerer import RAGAnswerer
except Exception:
    RAGAnswerer = None  # show in UI error

# --------------------------
# DATA (your repo layout)
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

st.set_page_config(page_title="Unified Chat", layout="wide")

DATASET_REL_PATH = os.path.join("data", "datasets", "tabular", "store_sales_train.csv")
DATE_COL = "date"


# --------------------------
# Cached resources
# --------------------------
@st.cache_resource
def get_router() -> RuleBasedRouter:
    return RuleBasedRouter()


@st.cache_resource
def get_rag_answerer() -> Any:
    if RAGAnswerer is None:
        return None
    return RAGAnswerer()


@st.cache_data(show_spinner=False)
def load_sales_df() -> Any:
    return load_table(DATASET_REL_PATH)


# --------------------------
# Helpers: robust date range extraction + enforcement
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

_MONTH_RANGE_PATTERNS = [
    # "january to april in 2013" / "jan-apr 2013" / "march through june 2015"
    re.compile(
        r"\b(?P<m1>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        r"\s*(?:to|\-|through|thru|until)\s*"
        r"\b(?P<m2>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        r".*?\b(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
    # "from january to april 2013"
    re.compile(
        r"\bfrom\s+"
        r"(?P<m1>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        r"\s+to\s+"
        r"(?P<m2>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        r"\s+(?P<y>\d{4})\b",
        re.IGNORECASE,
    ),
]


def extract_year_range_from_question(q: str) -> Optional[Tuple[str, str]]:
    """
    'from 2014 to 2016' -> ('2014-01-01','2016-12-31')
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


def _month_num(m: str) -> int:
    return _MONTHS[m.strip().lower()]


def _last_day_of_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def extract_month_range_from_question(q: str) -> Optional[Tuple[str, str]]:
    """
    'sales in january to april in 2013' -> ('2013-01-01','2013-04-30')
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
        end = f"{y}-{m2:02d}-{_last_day_of_month(y, m2):02d}"
        return (start, end)
    return None


def enforce_inclusive_date_range(df: Any, start: Optional[str], end: Optional[str], date_col: str = DATE_COL) -> Any:
    """
    Inclusive date filtering (>= start and <= end).
    """
    if df is None or date_col not in df.columns:
        return df

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    if start:
        d = d[d[date_col] >= pd.to_datetime(start)]
    if end:
        d = d[d[date_col] <= pd.to_datetime(end)]

    return d


# --------------------------
# Helpers: RAG normalization
# --------------------------
def normalize_rag_output(out: Any) -> Dict[str, Any]:
    if isinstance(out, str):
        return {"answer": out, "mode": "rag", "top_score": None, "chunks": []}

    if not isinstance(out, dict):
        return {"answer": str(out), "mode": "rag", "top_score": None, "chunks": []}

    answer = out.get("answer") or out.get("response") or out.get("text") or ""
    mode = out.get("mode") or "rag"
    top_score = out.get("top_score")

    chunks = out.get("chunks") or out.get("evidence") or out.get("contexts") or []
    norm_chunks: List[Dict[str, Any]] = []

    for i, ch in enumerate(chunks, start=1):
        if isinstance(ch, str):
            norm_chunks.append({"chunk_id": f"chunk_{i}", "score": None, "text": ch})
        elif isinstance(ch, dict):
            norm_chunks.append(
                {
                    "chunk_id": ch.get("chunk_id") or ch.get("id") or f"chunk_{i}",
                    "score": ch.get("score"),
                    "text": ch.get("text") or ch.get("chunk") or ch.get("content") or "",
                }
            )
        else:
            norm_chunks.append({"chunk_id": f"chunk_{i}", "score": None, "text": str(ch)})

    return {"answer": answer, "mode": mode, "top_score": top_score, "chunks": norm_chunks}


def run_rag(question: str) -> Dict[str, Any]:
    ans = get_rag_answerer()
    if ans is None:
        return {"error": "RAGAnswerer not available. Check src/agent/rag_answerer.py import."}
    return normalize_rag_output(ans.answer(question))


# --------------------------
# Helpers: DATA execution (fixed date range + value_col)
# --------------------------
def run_data_query(question: str) -> Dict[str, Any]:
    dq = parse_data_query(question)

    # ✅ Deterministic overrides at chat layer (month-range first, then year-range)
    mr = extract_month_range_from_question(question)
    if mr is not None:
        dq.start, dq.end = mr[0], mr[1]
    else:
        yr = extract_year_range_from_question(question)
        if yr is not None:
            dq.start, dq.end = yr[0], yr[1]

    df = load_sales_df()
    if df is None:
        raise ValueError(
            "Dataset load failed: load_table() returned None. "
            f"Expected CSV at: {DATASET_REL_PATH}"
        )

    schema = detect_schema(df)

    df_filt = filter_sales(
        df,
        store_nbr=dq.store_nbr,
        family=dq.family,
        start=dq.start,
        end=dq.end,
    )
    if df_filt is None:
        raise ValueError("filter_sales() returned None (expected a DataFrame).")

    # ✅ guarantee inclusive range
    df_filt = enforce_inclusive_date_range(df_filt, dq.start, dq.end, date_col=DATE_COL)

    if DATE_COL not in df_filt.columns:
        raise ValueError(f"Missing date column '{DATE_COL}'. Found: {list(df_filt.columns)}")

    value_col = dq.metric
    if value_col not in df_filt.columns:
        raise ValueError(f"Metric '{value_col}' not found. Available: {list(df_filt.columns)}")

    # Aggregate based on dq.mode
    if dq.mode == "daily":
        df_out = group_sum_by_date(df_filt, date_col=DATE_COL, value_col=value_col)
        summary = f"Computed **daily {value_col} (sum)** for the selected filters."
    elif dq.mode == "monthly":
        df_out = group_sum_by_month(df_filt, date_col=DATE_COL, value_col=value_col)
        summary = f"Computed **monthly {value_col} (sum)** for the selected filters."
    elif dq.mode == "yearly":
        df_out = group_sum_by_year(df_filt, date_col=DATE_COL, value_col=value_col)
        summary = f"Computed **yearly {value_col} (sum)** for the selected filters."
    elif dq.mode == "family":
        df_out = group_sum(df_filt, group_col="family", value_col=value_col)
        summary = f"Computed **{value_col} (sum) by family** for the selected filters."
    else:
        df_out = group_sum_by_date(df_filt, date_col=DATE_COL, value_col=value_col)
        summary = f"Computed **daily {value_col} (sum)** (fallback)."

    # Clean time labels for UX (monthly/yearly)
    if df_out is not None and not df_out.empty:
        time_col = df_out.columns[0]
        df_out = df_out.copy()

        if dq.mode == "monthly":
            dt = pd.to_datetime(df_out[time_col], errors="coerce")
            df_out[time_col] = dt.dt.to_period("M").astype(str)

        elif dq.mode == "yearly":
            dt = pd.to_datetime(df_out[time_col], errors="coerce")
            df_out[time_col] = dt.dt.year.astype(str)

    # Stable chronological sorting (robust even after string labels)
    if df_out is not None and len(df_out.columns) >= 2:
        x_col = df_out.columns[0]
        try:
            df_out = df_out.copy()
            if dq.mode == "monthly":
                sort_key = pd.to_datetime(df_out[x_col] + "-01", errors="coerce")
                df_out["_sort_key"] = sort_key
                df_out = df_out.sort_values("_sort_key").drop(columns=["_sort_key"])
            elif dq.mode == "yearly":
                sort_key = pd.to_datetime(df_out[x_col] + "-01-01", errors="coerce")
                df_out["_sort_key"] = sort_key
                df_out = df_out.sort_values("_sort_key").drop(columns=["_sort_key"])
            else:
                df_out[x_col] = pd.to_datetime(df_out[x_col], errors="coerce")
                df_out = df_out.sort_values(x_col)
        except Exception:
            pass

    return {"dq": dq, "schema": schema, "df_result": df_out, "summary": summary}


# --------------------------
# UI helpers: plot + peaks
# --------------------------
def plot_result(df_out: Any, mode: str) -> None:
    cols = list(df_out.columns)
    if len(cols) < 2:
        st.info("Result has fewer than 2 columns — cannot plot.")
        return

    x_col, y_col = cols[0], cols[1]
    df_plot = df_out.copy()

    # Convert x to a real datetime index for charting if possible
    try:
        if mode == "monthly":
            df_plot["_x"] = pd.to_datetime(df_plot[x_col] + "-01", errors="coerce")
            df_plot = df_plot.sort_values("_x").set_index("_x")
            st.line_chart(df_plot[y_col])
            return
        if mode == "yearly":
            df_plot["_x"] = pd.to_datetime(df_plot[x_col] + "-01-01", errors="coerce")
            df_plot = df_plot.sort_values("_x").set_index("_x")
            st.line_chart(df_plot[y_col])
            return

        df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="coerce")
        df_plot = df_plot.sort_values(x_col).set_index(x_col)
        st.line_chart(df_plot[y_col])
    except Exception:
        st.bar_chart(df_out.set_index(x_col)[y_col])


def render_peak_and_top5(dq: Any, df_res: Any) -> None:
    if dq.mode not in ("monthly", "yearly"):
        return
    if df_res is None or getattr(df_res, "empty", True):
        return
    if len(df_res.columns) < 2:
        return

    label_col = df_res.columns[0]
    value_col = df_res.columns[1]
    df_sorted = df_res.sort_values(value_col, ascending=False).reset_index(drop=True)

    top = df_sorted.iloc[0]
    st.success(f"📌 Peak {dq.mode}: **{top[label_col]}** with **{top[value_col]:,.0f} {dq.metric}**")

    st.write("**Top 5 peaks:**")
    st.dataframe(df_sorted.head(5), use_container_width=True)


# --------------------------
# UI
# --------------------------
st.title("Unified Router Chat (Phase 4)")

router = get_router()

with st.sidebar:
    st.header("Routing (v1)")
    st.caption("Rule-based routing. Override for debugging/demo.")

    override = st.toggle("Override router", value=False)
    forced_route = st.selectbox(
        "Force route",
        options=[Route.DATA.value, Route.RAG.value, Route.MIXED.value, Route.GENERAL.value],
        index=0,
        disabled=not override,
    )

    st.divider()
    st.header("Dataset")
    st.code(DATASET_REL_PATH)
    st.caption("✅ Found CSV" if os.path.exists(DATASET_REL_PATH) else "❌ CSV not found at this path")

    st.divider()
    st.header("Grounding policy")
    st.write("- **RAG**: cites chunks or refuses (threshold).")
    st.write("- **DATA**: computed values + table + plot; no guessing.")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if meta:
            with st.expander("Details", expanded=False):
                st.json(meta)


prompt = st.chat_input("Ask about your PDFs or your sales dataset...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})

    rr = router.route(prompt)
    chosen = rr.route.value
    expl = router.explain_route(rr)
    if override:
        chosen = forced_route

    assistant_meta: Dict[str, Any] = {
        "route": chosen,
        "router_explanation": expl,
        "router_scores": {k.value: v for k, v in rr.scores.items()},
        "router_matched": {
            Route.DATA.value: rr.matched.get(Route.DATA, []),
            Route.RAG.value: rr.matched.get(Route.RAG, []),
        },
    }

    assistant_text = ""

    with st.chat_message("assistant"):
        st.markdown(f"**Mode:** `{chosen}`")
        st.caption(expl if not override else f"Forced route: {chosen}")

        if chosen == Route.DATA.value:
            try:
                out = run_data_query(prompt)
                dq = out["dq"]
                df_res = out["df_result"]

                assistant_text = out["summary"]
                st.markdown(assistant_text)

                render_peak_and_top5(dq, df_res)

                with st.expander("Evidence (computed)", expanded=True):
                    st.write("**Parsed DataQuery:**")
                    dq_dict = dq.__dict__.copy() if hasattr(dq, "__dict__") else {"raw": str(dq)}
                    # Make it JSON-friendly and consistent
                    for k, v in list(dq_dict.items()):
                      if v is None:
                         dq_dict[k] = None
                    st.json(dq_dict)

                    st.write("**Result table:**")
                    st.dataframe(df_res, use_container_width=True)

                    st.write("**Plot:**")
                    plot_result(df_res, dq.mode)

                assistant_meta["data"] = {
                    "dq": dq.__dict__ if hasattr(dq, "__dict__") else str(dq),
                    "result_shape": getattr(df_res, "shape", None),
                    "columns": list(getattr(df_res, "columns", [])),
                }

            except Exception as e:
                assistant_text = f"Data pipeline error: {e}"
                st.error(assistant_text)
                assistant_meta["error"] = str(e)

        elif chosen == Route.RAG.value:
            out = run_rag(prompt)
            if "error" in out:
                assistant_text = out["error"]
                st.error(assistant_text)
                assistant_meta["error"] = assistant_text
            else:
                assistant_text = out.get("answer", "")
                st.markdown(assistant_text)

                with st.expander("Evidence (retrieved chunks)", expanded=True):
                    st.write(f"**RAG mode:** `{out.get('mode', 'rag')}`")
                    if out.get("top_score") is not None:
                        st.write(f"**Top score:** {out['top_score']}")

                    chunks = out.get("chunks", [])
                    if not chunks:
                        st.info("No chunks returned.")
                    else:
                        for i, ch in enumerate(chunks, start=1):
                            chunk_id = ch.get("chunk_id", f"chunk_{i}")
                            score = ch.get("score", None)
                            text = ch.get("text", "")

                            header = f"**{i}. {chunk_id}**"
                            if score is not None:
                                header += f" (score={score})"
                            st.markdown(header)
                            st.markdown(text)

                assistant_meta["rag"] = {
                    "mode": out.get("mode"),
                    "top_score": out.get("top_score"),
                    "chunks_count": len(out.get("chunks", [])),
                }

        elif chosen == Route.MIXED.value:
            assistant_text = (
                "MIXED routing detected, but Phase 4 supports **single-pipeline execution only**.\n\n"
                "Use the sidebar to force **DATA** or **RAG** for now."
            )
            st.warning(assistant_text)

        else:
            assistant_text = (
                "I can’t confidently route this to **DATA** or **RAG**.\n\n"
                "Try:\n"
                "- Include **sales/store/family/date** for DATA\n"
                "- Include **pdf/document/cite** for RAG\n"
                "Or force a route from the sidebar."
            )
            st.info(assistant_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_text, "meta": assistant_meta}
    )
