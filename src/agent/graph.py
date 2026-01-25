# src/agent/graph.py
import time
from typing import Any, Dict

from src.agent.state import AgentState, TraceEvent
from src.agent.router import Route
from src.agent import tools
from src.agent.tracing import save_trace


try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    StateGraph = None
    END = None


def _add_trace(state: AgentState, node: str, t0: float, inp: Dict[str, Any], out: Dict[str, Any]) -> None:
    ms = (time.perf_counter() - t0) * 1000.0
    state.trace.append(
        TraceEvent(
            node=node,
            ms=ms,
            input_summary=inp,
            output_summary=out,
        )
    )


def node_router(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    meta = tools.route_question(state.question)
    state.route = meta["route"]
    state.router_meta = meta
    _add_trace(
        state,
        "router",
        t0,
        {"question": state.question},
        {"route": state.route, "scores": meta.get("router_scores", {})},
    )
    return state


def node_data(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    out = tools.run_data(state.question)
    state.data = out
    _add_trace(
        state,
        "data_tool",
        t0,
        {"question": state.question},
        {"ok": "error" not in out, "keys": list(out.keys())},
    )
    return state


def node_rag(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    out = tools.run_rag(state.question)
    state.rag = out
    _add_trace(
        state,
        "rag_tool",
        t0,
        {"question": state.question},
        {"ok": "error" not in out, "chunks": len(out.get("chunks", []))},
    )
    return state


def node_verify(state: AgentState) -> AgentState:
    t0 = time.perf_counter()

    # deterministic verification rules (no LLM)
    if state.route == Route.DATA.value:
        if "error" in state.data:
            state.mode = "data"
            state.answer = f"Data pipeline error: {state.data['error']}"
            state.evidence = {"data": {"error": state.data["error"]}}
        else:
            df = state.data.get("df")
            if df is None or getattr(df, "empty", True):
                state.mode = "data"
                state.answer = "No rows matched your filters for that time range."
                state.evidence = {"data": {"dq": state.data.get("dq", {}), "empty": True}}
    elif state.route == Route.RAG.value:
        if "error" in state.rag:
            state.mode = "rag"
            state.answer = state.rag["error"]
            state.evidence = {"rag": {"error": state.rag["error"]}}
        else:
            chunks = state.rag.get("chunks", [])
            if not chunks:
                state.mode = "rag"
                state.answer = "I can’t answer from the documents because no relevant evidence was retrieved."
                state.evidence = {"rag": {"chunks": []}}
    else:
        # for GENERAL/MIXED we don’t do more in Phase 5 minimal
        pass

    _add_trace(state, "verify", t0, {"route": state.route}, {"mode": state.mode, "has_answer": bool(state.answer)})
    return state


def node_finalize(state: AgentState) -> AgentState:
    t0 = time.perf_counter()

    # If verify already set an answer, keep it.
    if state.answer:
        _add_trace(state, "finalize", t0, {"route": state.route}, {"mode": state.mode})
        return state

    if state.route == Route.DATA.value:
        state.mode = "data"
        state.answer = "Computed result from dataset."
        state.evidence = {"data": {"dq": state.data.get("dq", {}), "df": state.data.get("df")}}
    elif state.route == Route.RAG.value:
        state.mode = "rag"
        state.answer = state.rag.get("answer", "")
        state.evidence = {"rag": {"top_score": state.rag.get("top_score"), "chunks": state.rag.get("chunks", [])}}
    elif state.route == Route.MIXED.value:
        state.mode = "mixed"
        state.answer = "MIXED routing detected. Phase 5 minimal graph does not execute mixed tools yet."
        state.evidence = {"router": state.router_meta}
    else:
        state.mode = "general"
        state.answer = "I can’t confidently route this to DATA or RAG. Use the override to force a mode."
        state.evidence = {"router": state.router_meta}

    _add_trace(state, "finalize", t0, {"route": state.route}, {"mode": state.mode})
    return state


def build_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed. Run: pip install langgraph")

    g = StateGraph(AgentState)

    g.add_node("router", node_router)
    g.add_node("data_tool", node_data)
    g.add_node("rag_tool", node_rag)
    g.add_node("verify", node_verify)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("router")

    # route -> tool
    def _route_selector(state: AgentState) -> str:
        if state.route == Route.DATA.value:
            return "data_tool"
        if state.route == Route.RAG.value:
            return "rag_tool"
        # mixed/general go straight to verify/finalize
        return "verify"

    g.add_conditional_edges("router", _route_selector, {
        "data_tool": "data_tool",
        "rag_tool": "rag_tool",
        "verify": "verify",
    })

    g.add_edge("data_tool", "verify")
    g.add_edge("rag_tool", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# cached compiled graph in-process
_GRAPH = None


def run_agent(question: str) -> AgentState:
    """
    Phase 5/6 contract:
    - Always returns AgentState (dataclass)
    - Works even if LangGraph invoke() returns dict
    - Saves a lightweight trace JSON to data/traces/
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    # Start with a dataclass state
    init_state = AgentState(question=question)

    # LangGraph may return dict OR AgentState depending on compilation/state handling
    raw_out = _GRAPH.invoke(init_state)

    # ---- Normalize output to AgentState ----
    if isinstance(raw_out, dict):
        out_state = AgentState(question=raw_out.get("question", question))
        # copy known fields safely
        out_state.route = raw_out.get("route")
        out_state.mode = raw_out.get("mode", "general")
        out_state.answer = raw_out.get("answer", "")
        out_state.router_meta = raw_out.get("router_meta", {}) or {}
        out_state.evidence = raw_out.get("evidence", {}) or {}
        out_state.trace = raw_out.get("trace", []) or []
    else:
        out_state = raw_out  # already AgentState

    # ---- Save trace (Phase 6) ----
    payload = {
        "question": out_state.question,
        "route": out_state.route,
        "mode": out_state.mode,
        "answer_preview": (out_state.answer or "")[:300],
        "router_meta": out_state.router_meta,
        "trace": [
            t if isinstance(t, dict) else getattr(t, "__dict__", {"node": str(t), "ms": None})
            for t in (out_state.trace or [])
        ],
    }

    # Evidence summary only (no big tables)
    try:
        if out_state.mode == "data":
            dq = (out_state.evidence or {}).get("data", {}).get("dq", {})
            df = (out_state.evidence or {}).get("data", {}).get("df", None)
            payload["evidence"] = {
                "data": {
                    "dq": dq,
                    "df_shape": getattr(df, "shape", None),
                    "df_columns": list(getattr(df, "columns", [])) if df is not None else [],
                }
            }
        elif out_state.mode == "rag":
            chunks = (out_state.evidence or {}).get("rag", {}).get("chunks", []) or []
            payload["evidence"] = {"rag": {"chunks_count": len(chunks)}}
    except Exception:
        pass

    from src.agent.tracing import save_trace
    trace_path = save_trace(payload)

    # attach trace path for UI
    if out_state.router_meta is None:
        out_state.router_meta = {}
    out_state.router_meta["trace_path"] = trace_path

    return out_state
