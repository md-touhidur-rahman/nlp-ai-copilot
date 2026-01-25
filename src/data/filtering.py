import pandas as pd
from typing import Optional

def filter_sales(
    df: pd.DataFrame,
    store_nbr: Optional[int] = None,
    family: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    out = df

    if store_nbr is not None and "store_nbr" in out.columns:
        out = out[out["store_nbr"] == store_nbr]

    if family is not None and "family" in out.columns:
        out = out[out["family"] == family]

    if start is not None:
        out = out[out[date_col] >= start]

    if end is not None:
        out = out[out[date_col] <= end]

    return out
