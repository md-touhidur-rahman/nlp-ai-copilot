import re


def clean_pdf_text(text: str) -> str:
    """
    Clean typical PDF-extracted text issues while preserving paragraphs.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null chars
    text = text.replace("\x00", "")

    # Fix hyphenated line breaks: inter-\nnational -> international
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace single newlines inside paragraphs with spaces
    lines = text.split("\n")
    rebuilt = []
    buffer = []

    def flush_buffer():
        if buffer:
            paragraph = " ".join(buffer)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                rebuilt.append(paragraph)
            buffer.clear()

    for line in lines:
        line = line.strip()
        if line == "":
            flush_buffer()
            rebuilt.append("")  # paragraph break
        else:
            buffer.append(line)

    flush_buffer()

    cleaned = "\n".join(rebuilt)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


if __name__ == "__main__":
    from pathlib import Path

    in_path = Path("data/processed/sample.txt")
    out_path = Path("data/processed/sample.cleaned.txt")

    raw = in_path.read_text(encoding="utf-8")
    cleaned = clean_pdf_text(raw)

    out_path.write_text(cleaned, encoding="utf-8")

    print("Raw chars:", len(raw))
    print("Cleaned chars:", len(cleaned))
    print("\n--- CLEANED PREVIEW ---\n")
    print(cleaned[:1000])
    print(f"\nSaved to: {out_path}")
