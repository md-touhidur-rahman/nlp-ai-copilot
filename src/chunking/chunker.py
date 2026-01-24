from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    source: str
    start_char: int
    end_char: int
    text: str


def _split_into_paragraphs(text: str) -> List[str]:
    # paragraphs separated by blank lines
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Chunk by paragraph packing (keeps semantic structure better than naive splitting).
    Uses character-based limits (token-based later).
    """
    paras = _split_into_paragraphs(text)
    chunks: List[Chunk] = []

    cur = ""
    cur_start = 0
    cursor = 0  # tracks char position in original-ish stream (approx)

    for p in paras:
        # if adding this paragraph would overflow, flush current chunk
        if cur and (len(cur) + 2 + len(p) > max_chars):
            end = cur_start + len(cur)
            chunk_id = f"{Path(source).stem}__{len(chunks):04d}"
            chunks.append(
                Chunk(chunk_id=chunk_id, source=source, start_char=cur_start, end_char=end, text=cur.strip())
            )

            # start next chunk with overlap
            if overlap_chars > 0:
                overlap = cur[-overlap_chars:]
                cur = overlap
                cur_start = max(end - len(overlap), 0)
            else:
                cur = ""
                cur_start = end

        # append paragraph
        if cur:
            cur += "\n\n" + p
        else:
            cur = p
            cur_start = cursor

        cursor += len(p) + 2

    # flush last chunk
    if cur.strip():
        end = cur_start + len(cur)
        chunk_id = f"{Path(source).stem}__{len(chunks):04d}"
        chunks.append(Chunk(chunk_id=chunk_id, source=source, start_char=cur_start, end_char=end, text=cur.strip()))

    return chunks


def save_chunks_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    in_path = Path("data/processed/sample.cleaned.txt")
    out_path = Path("data/processed/sample.chunks.jsonl")

    text = in_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, source="sample.cleaned.txt", max_chars=1800, overlap_chars=200)

    print("Chunks:", len(chunks))
    print("\n--- FIRST CHUNK PREVIEW ---\n")
    print(chunks[0].text[:900])

    save_chunks_jsonl(chunks, out_path)
    print(f"\nSaved to: {out_path}")
