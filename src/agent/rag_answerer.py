from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.service import RetrievalService, RetrievedChunk
from src.llm.ollama_client import OllamaClient, OllamaConfig


@dataclass
class RAGConfig:
    top_k: int = 5
    model: str = "llama3.2:3b"
    min_doc_score: float = 0.25   # threshold


def format_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[{c.chunk_id} | score={c.score:.4f} | source={c.source}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


class RAGAnswerer:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.retriever = RetrievalService()
        self.llm = OllamaClient(OllamaConfig(model=self.config.model))

    def answer(self, question: str) -> dict:
        chunks = self.retriever.search(question, top_k=self.config.top_k)

        if not chunks:
            return {
                "answer": "I don't know based on the document.",
                "chunks": [],
                "mode": "general",
                "top_score": 0.0,
            }

        top_score = max(c.score for c in chunks)
        mode = "document" if top_score >= self.config.min_doc_score else "general"

        context = format_context(chunks)

        prompt = f"""
You are an AI assistant.
Answer using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the document."

Question:
{question}

Context:
{context}

Return:
- A short answer (3–7 sentences)
- Do NOT hallucinate
""".strip()

        response = self.llm.generate(prompt)

        return {
            "answer": response,
            "chunks": chunks,
            "mode": mode,
            "top_score": top_score,
        }
