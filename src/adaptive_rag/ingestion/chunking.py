"""Character-based chunking that prefers paragraph / sentence boundaries.

Chunks never cross PDF page boundaries, so every chunk keeps an exact page number.
"""

from __future__ import annotations

import re

from adaptive_rag.ingestion.extract import PageText
from adaptive_rag.schemas import Chunk


def _split_points(text: str) -> list[int]:
    """Candidate cut positions: after paragraph breaks and sentence ends."""
    points = {m.end() for m in re.finditer(r"\n\s*\n", text)}
    points |= {m.end() for m in re.finditer(r"[.!?][\"')\]]?\s+", text)}
    return sorted(points)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) character spans covering ``text``."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    n = len(text)
    if n <= chunk_size:
        return [(0, n)] if text.strip() else []
    points = _split_points(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        hard_end = min(start + chunk_size, n)
        end = hard_end
        if hard_end < n:
            # last natural boundary in the second half of the window
            good = [p for p in points if start + chunk_size // 2 <= p <= hard_end]
            if good:
                end = good[-1]
            else:
                space = text.rfind(" ", start + chunk_size // 2, hard_end)
                end = space if space > start else hard_end
        spans.append((start, end))
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # move the overlap start forward to a word boundary
        space = text.find(" ", next_start, end)
        start = space + 1 if 0 <= space < end else next_start
    return [(s, e) for s, e in spans if text[s:e].strip()]


def chunk_pages(
    pages: list[PageText], doc_id: str, filename: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for start, end in chunk_text(page.text, chunk_size, overlap):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{idx}",
                    doc_id=doc_id,
                    filename=filename,
                    text=page.text[start:end].strip(),
                    chunk_index=idx,
                    page=page.page,
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks
