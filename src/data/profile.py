# src/data/profile.py
from typing import Any, Dict, List
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
)


def profile_df(df: Any, sample_rows: int = 200) -> Dict[str, Any]:
    """
    Profile any dataframe in a deterministic, lightweight way.
    - numeric columns
    - datetime candidates (native datetime OR parseable sample)
    - categorical columns (fallback)
    - missingness
    """
    if df is None:
        return {"ok": False, "error": "df is None"}

    if not hasattr(df, "columns"):
        return {"ok": False, "error": "df has no columns attribute"}

    n_rows = getattr(df, "shape", (0, 0))[0]
    n_cols = getattr(df, "shape", (0, 0))[1]

    numeric_cols: List[str] = []
    datetime_cols: List[str] = []
    categorical_cols: List[str] = []
    missingness: Dict[str, float] = {}

    # sample for parsing checks (fast)
    samp = df.head(sample_rows).copy()

    for col in df.columns:
        s = df[col]
        # missingness
        try:
            miss = float(s.isna().mean())
        except Exception:
            miss = 0.0
        missingness[str(col)] = round(miss, 4)

        # numeric
        if is_numeric_dtype(s):
            numeric_cols.append(str(col))
            continue

        # datetime native
        if is_datetime64_any_dtype(s):
            datetime_cols.append(str(col))
            continue

        # try parse sample as datetime (only if it looks like strings/objects)
        try:
            parsed = pd.to_datetime(samp[col], errors="coerce", infer_datetime_format=True)
            # accept if enough values parse
            if parsed.notna().mean() >= 0.8 and parsed.notna().sum() >= 5:
                datetime_cols.append(str(col))
                continue
        except Exception:
            pass

        # else categorical/other
        categorical_cols.append(str(col))

    return {
        "ok": True,
        "shape": (n_rows, n_cols),
        "columns": [str(c) for c in df.columns],
        "numeric_cols": numeric_cols,
        "datetime_cols": datetime_cols,
        "categorical_cols": categorical_cols,
        "missingness": missingness,
    }
