# tests/test_null_normalization.py
from src.agent.tools import normalize_nulls


def test_normalize_nulls_string_to_none():
    d = {"store_nbr": "NULL", "family": "NULL", "metric": "sales"}
    out = normalize_nulls(d)
    assert out["store_nbr"] is None
    assert out["family"] is None
    assert out["metric"] == "sales"
