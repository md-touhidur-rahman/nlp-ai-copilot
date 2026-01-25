# src/agent/graph.py
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.agent.router import RuleBasedRouter, Route
from src.agent import tools

TRACES_DIR = os.path.join("data", "traces")
_ROUTER = RuleBasedRouter()


@dataclass
class AgentState:
    question: str
    mode: str = "general"
    route: Optional[str] = None
    answer: str = ""
    evidence: Dict[str, Any] = None
    router_meta: Dict[str, Any] = None
    trace: List[Dict[str, Any]] = None


def _ensure_traces_dir() -> None:
    os.makedirs(TRACES_DIR, exist_ok=True)


def _trace_path() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    ms = int(time.time() * 1000)
    return os.path.join(TRACES_DIR, f"trace-{ts}-{ms}.json")


def _save_trace(payload: Dict[str, Any]) -> str:
    _ensure_traces_dir()
    path = _trace_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


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


def run_agent(question: str, override: bool = False, forced_route: Optional[str] = None) -> Dict[str, Any]:
    q = question.strip() if isinstance(question, str) else ""

    state = AgentState(question=q, evidence={}, router_meta={}, trace=[])

    router_meta = _route(q)
    route = router_meta["route"]

    if override and forced_route:
        route = forced_route
        router_meta["router_explanation"] = f"Forced route: {forced_route}"

    state.route = route
    state.router_meta = router_meta
    state.trace.append({"node": "route", "route": route})

    if route == Route.DATA.value:
        out = tools.run_data(q)
        state.mode = "data"
        if out.get("error"):
            state.answer = f"Data pipeline error: {out['error']}"
            if out.get("hint"):
                state.answer += f"\n\n{out['hint']}"
        else:
            state.answer = out.get("summary", "Computed result from dataset.")
        state.evidence = {"data": out}

    elif route == Route.RAG.value:
        out = tools.run_rag(q)
        state.mode = "rag"
        state.answer = out.get("answer", out.get("error", ""))
        state.evidence = {"rag": out}

    elif route == Route.MIXED.value:
        state.mode = "mixed"
        state.answer = (
            "MIXED routing detected.\n\n"
            "This version executes one tool path at a time.\n"
            "Use Override routing to force DATA or RAG."
        )
        state.evidence = {"mixed": {"note": "Not implemented"}}

    else:
        state.mode = "general"
        state.answer = (
            "I can’t confidently route this to DATA or RAG.\n\n"
            "Add DATA terms (sales, plot, average) or RAG terms (pdf, cite) or use Override."
        )
        state.evidence = {}

    payload = asdict(state)
    trace_path = _save_trace(payload)
    payload["router_meta"]["trace_path"] = trace_path
    return payload
