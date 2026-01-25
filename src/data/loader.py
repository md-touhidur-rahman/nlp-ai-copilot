import pandas as pd
from pathlib import Path


def load_table(path: str) -> pd.DataFrame:
    """
    Load CSV or Parquet into a pandas DataFrame.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
