from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.retrieval.service import RetrievalService, RetrievedChunk
from src.llm.ollama_client import OllamaClient, OllamaConfig


@dataclass
class RAGConfig:
    top_k: int = 5
    model: str = "llama3.2:3b"
    min_relevance_score: float = 0.25  # if top score < this, use general mode


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Turn retrieved chunks into a context string for the LLM."""
    parts = []
    for c in chunks:
        parts.append(f"[{c.chunk_id} | score={c.score:.4f} | source={c.source}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


class RAGAnswerer:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.retriever = RetrievalService()
        self.llm = OllamaClient(OllamaConfig(model=self.config.model))

    def answer(self, question: str) -> Dict[str, Any]:
        # 1) Retrieve
        chunks = self.retriever.search(question, top_k=self.config.top_k)
        top_score = chunks[0].score if chunks else 0.0

        # 2) Decide mode
        if top_score < self.config.min_relevance_score:
            mode = "general"
            context = ""
        else:
            mode = "rag"
            context = format_context(chunks)

        # 3) Prompt
        prompt = f"""
You are an AI assistant.

Mode: {mode}

If mode is "rag":
- Answer ONLY using the context below.
- If the answer is not in the context, say exactly: "I don't know based on the document."

If mode is "general":
- Answer from general knowledge.
- Start your answer with: "Not from the document: "

Question:
{question}

Context:
{context}

Return a short answer (3–7 sentences).
""".strip()

        # 4) Generate
        response = self.llm.generate(prompt)

        # 5) Return
        return {
            "answer": response,
            "chunks": chunks,
            "mode": mode,
            "top_score": top_score,
        }
