# src/agent/graph.py
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

from src.agent.router import RuleBasedRouter, Route

# IMPORTANT:
# graph.py may import tools.py
# tools.py must NOT import graph.py (prevents circular import)
from src.agent import tools

TRACES_DIR = os.path.join("data", "traces")


@dataclass
class AgentState:
    question: str
    mode: str = "general"  # "data" | "rag" | "mixed" | "general"
    route: Optional[str] = None
    answer: str = ""
    evidence: Dict[str, Any] = None
    router_meta: Dict[str, Any] = None
    trace: List[Dict[str, Any]] = None


_ROUTER = RuleBasedRouter()


def _ensure_traces_dir() -> None:
    os.makedirs(TRACES_DIR, exist_ok=True)


def _now_trace_path() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    ms = int(time.time() * 1000)
    return os.path.join(TRACES_DIR, f"trace-{ts}-{ms}.json")


def _save_trace(payload: Dict[str, Any]) -> str:
    _ensure_traces_dir()
    path = _now_trace_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def _trace_add(trace: List[Dict[str, Any]], node: str, t0: float, extra: Optional[Dict[str, Any]] = None) -> None:
    ms = (time.perf_counter() - t0) * 1000.0
    row = {"node": node, "ms": round(ms, 2)}
    if extra:
        row.update(extra)
    trace.append(row)


def _route(question: str) -> Dict[str, Any]:
    rr = _ROUTER.route(question)
    return {
        "route": rr.route.value,
        "router_explanation": _ROUTER.explain_route(rr),
        "router_scores": {k.value: v for k, v in rr.scores.items()},
        "router_matched": {
            Route.DATA.value: rr.matched.get(Route.DATA, []),
            Route.RAG.value: rr.matched.get(Route.RAG, []),
        },
    }


def run_agent(
    question: Union[str, Any],
    override: bool = False,
    forced_route: Optional[str] = None,
    dataset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Final agent entrypoint (Streamlit-safe):
    - question must be text (we coerce safely)
    - dataset optionally provides a user-uploaded df/schema context for DATA mode
    - returns a JSON-friendly dict (not a dataclass object)
    """
    q = str(question or "").strip()

    state = AgentState(
        question=q,
        evidence={},
        router_meta={},
        trace=[],
    )
    trace: List[Dict[str, Any]] = state.trace

    # -------------------------
    # Route
    # -------------------------
    t0 = time.perf_counter()
    router_meta = _route(q)
    route = router_meta["route"]

    if override and forced_route:
        route = forced_route
        router_meta["router_explanation"] = f"Forced route: {forced_route}"

    state.router_meta = router_meta
    state.route = route
    _trace_add(trace, "route", t0, {"route": route})

    # -------------------------
    # Execute
    # -------------------------
    if route == Route.DATA.value:
        t1 = time.perf_counter()
        out = tools.run_data(q, dataset=dataset)  # dataset-aware
        _trace_add(trace, "data_tool", t1)

        state.mode = "data"
        if isinstance(out, dict) and out.get("error"):
            state.answer = f"Data pipeline error: {out.get('error')}"
            if out.get("hint"):
                state.answer += f"\n\n{out.get('hint')}"
        else:
            state.answer = (out or {}).get("summary") or "Computed result from dataset."

        state.evidence = {"data": out}

    elif route == Route.RAG.value:
        t1 = time.perf_counter()
        out = tools.run_rag(q)
        _trace_add(trace, "rag_tool", t1)

        state.mode = "rag"
        if isinstance(out, dict) and out.get("error"):
            state.answer = out["error"]
        else:
            state.answer = (out or {}).get("answer", "") or ""

        state.evidence = {"rag": out}

    elif route == Route.MIXED.value:
        state.mode = "mixed"
        state.answer = (
            "MIXED routing detected. This version executes a single tool path only.\n\n"
            "Use Override routing to force DATA or RAG."
        )
        state.evidence = {"mixed": {"note": "Not implemented in this version."}}

    else:
        state.mode = "general"
        state.answer = (
            "I can’t confidently route this to DATA or RAG.\n\n"
            "Try adding:\n"
            "- DATA terms: sales, store, date range, average, plot\n"
            "- RAG terms: pdf, document, cite, according to\n"
            "Or use Override routing."
        )
        state.evidence = {}

    # -------------------------
    # Persist trace (include a small answer preview)
    # -------------------------
    trace_payload = {
        "question": state.question,
        "mode": state.mode,
        "route": state.route,
        "answer": state.answer,
        "answer_preview": (state.answer[:800] + " ...") if len(state.answer) > 800 else state.answer,
        "router_meta": state.router_meta,
        "evidence": tools.summarize_evidence_for_trace(state.evidence),
        "trace": state.trace,
    }
    trace_path = _save_trace(trace_payload)
    state.router_meta["trace_path"] = trace_path

    return asdict(state)
