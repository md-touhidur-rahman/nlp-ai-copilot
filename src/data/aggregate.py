import pandas as pd

def group_sum_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "sales",
) -> pd.DataFrame:
    out = (
        df.groupby(date_col, as_index=False)[value_col]
        .sum()
        .sort_values(date_col)
    )
    return out


def group_sum(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, as_index=False)[value_col]
          .sum()
          .sort_values(value_col, ascending=False)
    )
    return out

def group_sum_by_month(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

    tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)  # "2016-12"
    out = (
        tmp.groupby("month", as_index=False)[value_col]
           .sum()
           .sort_values(value_col, ascending=False)
    )
    return out

def group_sum_by_year(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

    tmp["year"] = tmp[date_col].dt.year.astype(int)
    out = (
        tmp.groupby("year", as_index=False)[value_col]
           .sum()
           .sort_values(value_col, ascending=False)
    )
    return out