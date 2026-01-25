# src/agent/state.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    node: str
    ms: float
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    question: str

    # routing
    route: Optional[str] = None
    router_meta: Dict[str, Any] = field(default_factory=dict)

    # tool outputs
    rag: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    # final response
    answer: str = ""
    mode: str = "general"   # "data" | "rag" | "general" | "mixed"
    evidence: Dict[str, Any] = field(default_factory=dict)

    # tracing
    trace: List[TraceEvent] = field(default_factory=list)
