import pandas as pd

def prepare_line_chart(df: pd.DataFrame, x: str = "date", y: str = "sales") -> pd.DataFrame:
    """
    Streamlit likes a DataFrame where x is a column and y is numeric.
    """
    out = df.copy()
    out[x] = pd.to_datetime(out[x])
    out = out.sort_values(x)
    return out[[x, y]]
