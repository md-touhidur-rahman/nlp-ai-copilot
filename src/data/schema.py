import pandas as pd


def detect_schema(df: pd.DataFrame) -> dict:
    schema = {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "missing": df.isna().sum().to_dict(),
    }

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            schema["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema["datetime"].append(col)
        elif pd.api.types.is_object_dtype(dtype):
            # try datetime parse
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                df[col] = parsed
                schema["datetime"].append(col)
            except Exception:
                schema["categorical"].append(col)

    return schema
