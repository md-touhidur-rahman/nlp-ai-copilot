# src/agent/router.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import re


class Route(str, Enum):
    RAG = "rag"
    DATA = "data"
    MIXED = "mixed"
    GENERAL = "general"


@dataclass
class RouteResult:
    route: Route
    scores: Dict[Route, int]
    matched: Dict[Route, List[str]]


class RuleBasedRouter:
    """
    Simple rule-based router (Phase 4 v1.1).
    - DATA if data keywords / intent detected
    - RAG if doc keywords / intent detected
    - MIXED if both
    - fallback: if question looks "data-shaped", route to DATA instead of GENERAL
    """

    def __init__(self) -> None:
        # Keep these lists small + interpretable (interview-ready).
        self.data_keywords = [
            "sales", "onpromotion", "store", "store_nbr", "family",
            "dataset", "csv", "table", "column", "schema",
            "sum", "total", "average", "avg", "mean", "median",
            "max", "maximum", "min", "minimum",
            "trend", "plot", "chart", "graph",
            "daily", "monthly", "yearly", "by month", "by year",
            "compare", "comparison", "group by", "aggregate", "aggregation",
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
            # years often appear in dataset questions
            "2013", "2014", "2015", "2016", "2017",
        ]

        self.doc_keywords = [
            "pdf", "document", "docs", "paper", "report", "manual",
            "policy", "guideline", "spec", "specification",
            "cite", "citation", "source", "evidence",
            "section", "page", "paragraph", "chunk",
            "what does the document say", "according to the document",
        ]

        # Intent phrases get a score boost (kept simple).
        self.data_intent_phrases = [
            "show sales", "daily sales", "monthly sales", "yearly sales",
            "which month", "which year",
            "maximum sales", "minimum sales",
            "highest sales", "lowest sales",
            "top month", "top year",
            "trend of sales", "sales trend",
            "plot sales", "chart sales", "graph sales",
            "compare sales",
        ]

        self.doc_intent_phrases = [
            "according to", "from the pdf", "from the document",
            "cite sources", "show sources", "give citations",
        ]

        # Small heuristic patterns for "data-shaped" questions
        self._year_pat = re.compile(r"\b(19|20)\d{2}\b")            # 2016
        self._ymd_pat = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")        # 2016-01-31
        self._month_word_pat = re.compile(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
            re.IGNORECASE,
        )
        self._store_pat = re.compile(r"\bstore\s*\d+\b", re.IGNORECASE)  # store 1

    def _count_matches(self, text: str, keywords: List[str]) -> Tuple[int, List[str]]:
        t = text.lower()
        matched: List[str] = []
        score = 0
        for kw in keywords:
            if kw in t:
                matched.append(kw)
                score += 1
        return score, matched

    def _looks_like_data_question(self, q: str) -> Tuple[bool, List[str]]:
        """
        Fallback heuristic: if no keywords matched, detect typical data query signals.
        Returns (bool, reasons).
        """
        reasons: List[str] = []

        if self._ymd_pat.search(q):
            reasons.append("has_date_YYYY-MM-DD")
        if self._year_pat.search(q):
            reasons.append("has_year")
        if self._month_word_pat.search(q):
            reasons.append("has_month_word")
        if self._store_pat.search(q):
            reasons.append("has_store_pattern")

        # Common analytic verbs even without explicit "sales"
        analytic_verbs = ["max", "maximum", "min", "minimum", "highest", "lowest", "average", "sum", "total", "compare"]
        for v in analytic_verbs:
            if v in q:
                reasons.append(f"has_analytic_verb:{v}")
                break

        return (len(reasons) > 0), reasons

    def route(self, question: str) -> RouteResult:
        q = question.strip().lower()

        data_score, data_matched = self._count_matches(q, self.data_keywords)
        doc_score, doc_matched = self._count_matches(q, self.doc_keywords)

        # Phrase boosts
        for p in self.data_intent_phrases:
            if p in q:
                data_score += 2
                data_matched.append(p)

        for p in self.doc_intent_phrases:
            if p in q:
                doc_score += 2
                doc_matched.append(p)

        # Decision rules
        if data_score > 0 and doc_score > 0:
            route = Route.MIXED
        elif data_score > 0:
            route = Route.DATA
        elif doc_score > 0:
            route = Route.RAG
        else:
            # NEW: fallback heuristic -> DATA before GENERAL
            looks_data, reasons = self._looks_like_data_question(q)
            if looks_data:
                route = Route.DATA
                data_matched.extend(reasons)
                data_score += 1
            else:
                route = Route.GENERAL

        return RouteResult(
            route=route,
            scores={
                Route.DATA: data_score,
                Route.RAG: doc_score,
                Route.MIXED: 0,
                Route.GENERAL: 0,
            },
            matched={
                Route.DATA: data_matched,
                Route.RAG: doc_matched,
                Route.MIXED: [],
                Route.GENERAL: [],
            },
        )

    def explain_route(self, rr: RouteResult) -> str:
        data_score = rr.scores.get(Route.DATA, 0)
        rag_score = rr.scores.get(Route.RAG, 0)
        data_terms = rr.matched.get(Route.DATA, [])[:8]
        rag_terms = rr.matched.get(Route.RAG, [])[:8]

        if rr.route == Route.DATA:
            return f"Routed to DATA (data_score={data_score}, matched={data_terms})"
        if rr.route == Route.RAG:
            return f"Routed to RAG (rag_score={rag_score}, matched={rag_terms})"
        if rr.route == Route.MIXED:
            return f"Routed to MIXED (data_score={data_score}, rag_score={rag_score})"
        return "Routed to GENERAL (no strong data/doc signals)"
