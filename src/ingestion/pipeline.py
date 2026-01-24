from pathlib import Path

from src.ingestion.pdf_loader import load_pdf_text
from src.preprocessing.text_cleaner import clean_pdf_text
from src.chunking.chunker import chunk_text, save_chunks_jsonl


def run_pipeline(pdf_path: Path) -> Path:
    """
    PDF -> extracted text -> cleaned text -> chunks.jsonl
    Returns the path to the chunks file.
    """
    raw_text = load_pdf_text(pdf_path)
    cleaned = clean_pdf_text(raw_text)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = processed_dir / f"{pdf_path.stem}.cleaned.txt"
    cleaned_path.write_text(cleaned, encoding="utf-8")

    chunks = chunk_text(cleaned, source=pdf_path.name, max_chars=1800, overlap_chars=200)
    chunks_path = processed_dir / f"{pdf_path.stem}.chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)

    return chunks_path