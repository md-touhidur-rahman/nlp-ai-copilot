from pathlib import Path
from pypdf import PdfReader


def load_pdf_pages(path: Path) -> list[str]:
    """Return a list of page texts (one string per page)."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def load_pdf_text(path: Path) -> str:
    """Return full PDF text with page markers."""
    pages = load_pdf_pages(path)
    marked = []
    for i, t in enumerate(pages, start=1):
        marked.append(f"\n\n--- PAGE {i} ---\n{t}")
    return "".join(marked).strip()


def save_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    pdf_path = Path("data/raw/sample.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing: {pdf_path.resolve()}")

    text = load_pdf_text(pdf_path)

    print("File:", pdf_path.name)
    print("Chars:", len(text))
    print("\nPreview:\n")
    print(text[:800])

    out_path = Path("data/processed/sample.txt")
    save_text(text, out_path)
    print(f"\nSaved to: {out_path}")
