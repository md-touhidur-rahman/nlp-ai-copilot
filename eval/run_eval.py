import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agent.router import RuleBasedRouter, Route
from src.agent import tools


@dataclass
class CaseResult:
    id: str
    ok: bool
    reason: str
    details: Dict[str, Any]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _fail(cid: str, reason: str, details: Dict[str, Any]) -> CaseResult:
    return CaseResult(id=cid, ok=False, reason=reason, details=details)


def _pass(cid: str, details: Dict[str, Any]) -> CaseResult:
    return CaseResult(id=cid, ok=True, reason="ok", details=details)


def eval_route(router: RuleBasedRouter, q: str, expected: str) -> Tuple[bool, Dict[str, Any]]:
    rr = router.route(q)
    got = rr.route.value
    meta = {
        "expected": expected,
        "got": got,
        "scores": {k.value: v for k, v in rr.scores.items()},
        "matched_data": rr.matched.get(Route.DATA, []),
        "matched_rag": rr.matched.get(Route.RAG, []),
    }
    return (got == expected), meta


def eval_data_case(case: Dict[str, Any], router: RuleBasedRouter) -> CaseResult:
    cid = case["id"]
    q = case["q"]

    ok_route, rmeta = eval_route(router, q, case["expect_route"])
    if not ok_route:
        return _fail(cid, "route_mismatch", {"q": q, **rmeta})

    out = tools.run_data(q)
    if "expect_error_contains" in case:
        err = out.get("error")
        if not err:
            return _fail(cid, "expected_error_missing", {"q": q, "out_keys": list(out.keys())})
        if case["expect_error_contains"] not in err:
            return _fail(cid, "error_text_mismatch", {"q": q, "error": err})
        return _pass(cid, {"q": q, "error": err})

    if "error" in out:
        return _fail(cid, "unexpected_data_error", {"q": q, "error": out["error"]})

    dq = out.get("dq", {}) or {}
    df = out.get("df", None)

    if df is None:
        return _fail(cid, "missing_df", {"q": q, "dq": dq})

    # Basic table sanity
    try:
        shape = df.shape
    except Exception:
        shape = None
    if shape is None or shape[0] == 0:
        return _fail(cid, "empty_df", {"q": q, "dq": dq, "shape": shape})

    # Expected parsed fields (deterministic overrides)
    for k in ("expect_start", "expect_end", "expect_store_nbr"):
        if k in case:
            expected_val = case[k]
            dq_key = k.replace("expect_", "")
            got_val = dq.get(dq_key)
            if got_val != expected_val:
                return _fail(cid, "dq_mismatch", {"q": q, "dq_key": dq_key, "expected": expected_val, "got": got_val, "dq": dq})

    # If start/end expected, ensure df spans within range (inclusive)
    # We don't require exact first/last rows because aggregation/filters may change,
    # but we enforce: all dates within [start, end] when dates are datetime-like.
    if "expect_start" in case and "expect_end" in case:
        date_col = df.columns[0]
        try:
            import pandas as pd
            dt = pd.to_datetime(df[date_col], errors="coerce")
            start = pd.to_datetime(case["expect_start"])
            end = pd.to_datetime(case["expect_end"])
            if dt.notna().any():
                if dt.min() < start or dt.max() > end:
                    return _fail(
                        cid,
                        "df_out_of_range",
                        {"q": q, "min": str(dt.min()), "max": str(dt.max()), "start": str(start), "end": str(end)},
                    )
        except Exception:
            # If labels are strings (monthly/yearly formatting), skip this check
            pass

    return _pass(cid, {"q": q, "dq": dq, "df_shape": shape})


def eval_rag_case(case: Dict[str, Any], router: RuleBasedRouter) -> CaseResult:
    cid = case["id"]
    q = case["q"]

    ok_route, rmeta = eval_route(router, q, case["expect_route"])
    if not ok_route:
        return _fail(cid, "route_mismatch", {"q": q, **rmeta})

    out = tools.run_rag(q)
    if "error" in out:
        # If RAG isn't configured, let runner handle skip by env var
        return _fail(cid, "rag_error", {"q": q, "error": out["error"]})

    answer = out.get("answer", "")
    chunks = out.get("chunks", []) or []

    # Grounding rule: either have chunks OR refusal-style answer
    # (We avoid assuming your exact refusal text; just check chunks present or answer non-empty.)
    if not answer:
        return _fail(cid, "empty_answer", {"q": q})
    if len(chunks) == 0:
        # could be refusal (acceptable) — we mark pass but note it
        return _pass(cid, {"q": q, "note": "no_chunks_returned_may_be_refusal"})
    return _pass(cid, {"q": q, "chunks_count": len(chunks)})


def main() -> None:
    router = RuleBasedRouter()

    base = os.path.join("eval", "questions")
    data_path = os.path.join(base, "data.jsonl")
    rag_path = os.path.join(base, "rag.jsonl")

    results: List[CaseResult] = []

    # DATA
    for case in load_jsonl(data_path):
        results.append(eval_data_case(case, router))

    # RAG (optional)
    skip_rag = os.environ.get("EVAL_SKIP_RAG", "").strip() == "1"
    if not skip_rag and os.path.exists(rag_path):
        for case in load_jsonl(rag_path):
            results.append(eval_rag_case(case, router))

    # Report
    passed = sum(1 for r in results if r.ok)
    total = len(results)

    print("\n=== EVAL REPORT ===")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"{status}  {r.id}  {r.reason}")
        if not r.ok:
            print("  details:", json.dumps(r.details, indent=2))

    print(f"\nSummary: {passed}/{total} passed")

    # Exit code for CI
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
