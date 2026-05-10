"""Text and code chunking for TriVox Chat."""
import re


def chunk_text(text: str, max_chars: int = 500, overlap: int = 50, prefix: str = "") -> list[str]:
    """Chunk text by paragraphs, respecting max size."""
    if not text.strip():
        return []
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append(prefix + current.strip())
            # Overlap: keep last part
            words = current.split()
            current = " ".join(words[-10:]) + "\n\n" + para if len(words) > 10 else para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(prefix + current.strip())
    return chunks


def chunk_code(text: str, max_chars: int = 800, fname: str = "") -> list[str]:
    """Chunk code by functions/classes, with file context."""
    if not text.strip():
        return []

    chunks = []
    prefix = f"[{fname}] " if fname else ""

    # Try to split by function/class definitions
    pattern = re.compile(r'^((?:def |class |function |const |export |async )\S.*)', re.MULTILINE)
    parts = pattern.split(text)

    current = ""
    for part in parts:
        if not part.strip():
            continue
        if len(current) + len(part) > max_chars and current:
            chunks.append(prefix + current.strip())
            current = part
        else:
            current = current + "\n" + part if current else part

    if current.strip():
        chunks.append(prefix + current.strip())

    # If no functions found, fall back to line-based chunking
    if not chunks:
        lines = text.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) > max_chars and current:
                chunks.append(prefix + current.strip())
                current = line
            else:
                current = current + "\n" + line if current else line
        if current.strip():
            chunks.append(prefix + current.strip())

    return chunks
