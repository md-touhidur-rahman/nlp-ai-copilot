# src/data/generic_ops.py
from typing import Any, Dict, Optional
import pandas as pd


def describe_dataset(df: Any) -> Dict[str, Any]:
    if df is None:
        return {"error": "df is None"}
    try:
        return {
            "shape": getattr(df, "shape", None),
            "columns": list(getattr(df, "columns", [])),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing_count": {c: int(df[c].isna().sum()) for c in df.columns},
        }
    except Exception as e:
        return {"error": str(e)}


def aggregate_metric(df: Any, metric_col: str, agg: str = "sum") -> Any:
    """
    Returns a 1-row dataframe: [metric, value]
    """
    if df is None:
        raise ValueError("df is None")
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not in columns")

    s = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Metric '{metric_col}' has no numeric values after coercion.")

    agg = (agg or "sum").lower()

    if agg in ("sum", "total"):
        val = float(s.sum())
    elif agg in ("mean", "avg", "average"):
        val = float(s.mean())
    elif agg in ("max", "maximum", "highest", "peak"):
        val = float(s.max())
    elif agg in ("min", "minimum", "lowest"):
        val = float(s.min())
    elif agg in ("median",):
        val = float(s.median())
    elif agg in ("count",):
        val = float(s.count())
    else:
        # default safe
        val = float(s.sum())

    return pd.DataFrame({"metric": [metric_col], "value": [val]})


def topk_by_group(df: Any, group_col: str, metric_col: str, k: int = 10, agg: str = "sum") -> Any:
    """
    Returns top-k groups by aggregated metric (sum/mean/etc.)
    """
    if df is None:
        raise ValueError("df is None")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not in columns")
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not in columns")

    tmp = df[[group_col, metric_col]].copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[metric_col])

    agg = (agg or "sum").lower()
    if agg in ("mean", "avg", "average"):
        out = tmp.groupby(group_col, as_index=False)[metric_col].mean()
    elif agg in ("max", "maximum", "highest", "peak"):
        out = tmp.groupby(group_col, as_index=False)[metric_col].max()
    elif agg in ("min", "minimum", "lowest"):
        out = tmp.groupby(group_col, as_index=False)[metric_col].min()
    elif agg in ("median",):
        out = tmp.groupby(group_col, as_index=False)[metric_col].median()
    elif agg in ("count",):
        out = tmp.groupby(group_col, as_index=False)[metric_col].count()
    else:
        out = tmp.groupby(group_col, as_index=False)[metric_col].sum()

    out = out.sort_values(metric_col, ascending=False).head(int(k))
    out = out.reset_index(drop=True)
    return out


def histogram(df: Any, metric_col: str, bins: int = 20) -> Any:
    """
    Returns a dataframe with columns: bin_left, bin_right, count
    """
    if df is None:
        raise ValueError("df is None")
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not in columns")

    s = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Metric '{metric_col}' has no numeric values after coercion.")

    counts, edges = pd.cut(s, bins=bins, retbins=True, include_lowest=True)
    out = counts.value_counts().sort_index()
    # Build readable bins
    rows = []
    for interval, cnt in out.items():
        rows.append({"bin": str(interval), "count": int(cnt)})
    return pd.DataFrame(rows)


def trend_over_time(df: Any, date_col: str, metric_col: str, freq: str = "D", agg: str = "sum") -> Any:
    """
    Trend by time. freq: 'D' (daily), 'M' (monthly), 'Y' (yearly)
    Returns df with columns: time, value
    """
    if df is None:
        raise ValueError("df is None")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not in columns")
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not in columns")

    tmp = df[[date_col, metric_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, metric_col])

    if tmp.empty:
        raise ValueError("No valid rows after parsing date/metric columns.")

    tmp = tmp.set_index(date_col).sort_index()

    agg = (agg or "sum").lower()
    if agg in ("mean", "avg", "average"):
        s = tmp[metric_col].resample(freq).mean()
    elif agg in ("max", "maximum", "highest", "peak"):
        s = tmp[metric_col].resample(freq).max()
    elif agg in ("min", "minimum", "lowest"):
        s = tmp[metric_col].resample(freq).min()
    elif agg in ("median",):
        s = tmp[metric_col].resample(freq).median()
    elif agg in ("count",):
        s = tmp[metric_col].resample(freq).count()
    else:
        s = tmp[metric_col].resample(freq).sum()

    out = s.reset_index()
    out.columns = ["time", "value"]
    return out
