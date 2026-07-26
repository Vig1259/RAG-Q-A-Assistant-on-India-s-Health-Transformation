"""
chunker.py
----------
Splits the cleaned document into chunks of ~200-500 words, each carrying a
human-readable title (e.g. "AB-PMJAY", "Ayushman Arogya Mandirs", "PM-ABHIM").

Strategy:
1. Split the document on "## Heading" markers -> one section per heading.
   These headings already correspond to the natural topical units of the
   PIB backgrounder (Pillar 1: AB-PMJAY, Pillar 2: AAM, Pillar 3: PM-ABHIM,
   Pillar 4: ABDM, NHM sub-programmes, NCDs, AI in healthcare, etc.)
2. If a section is longer than MAX_WORDS, split it further at paragraph
   boundaries into sub-chunks, keeping each sub-chunk between MIN_WORDS and
   MAX_WORDS words and never cutting a sentence/paragraph in half.
3. If a section is shorter than MIN_WORDS, merge it with the next section
   (small "wrap up" sections like the closing paragraph get folded into the
   previous chunk instead of becoming a tiny, low-signal chunk).

Each chunk is stored as a dict: {id, title, text, word_count}.
Output is written to data/index/chunks.json.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List

MIN_WORDS = 200
MAX_WORDS = 500
TARGET_WORDS = 350  # soft target when splitting long sections

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(THIS_DIR), "data")
CLEAN_TXT = os.path.join(DATA_DIR, "raw", "pib_health_transformation.clean.txt")
FALLBACK_CLEAN_MD = os.path.join(DATA_DIR, "raw", "pib_health_transformation.md")
CHUNKS_PATH = os.path.join(DATA_DIR, "index", "chunks.json")


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    word_count: int


def _load_source_text() -> str:
    path = CLEAN_TXT if os.path.exists(CLEAN_TXT) else FALLBACK_CLEAN_MD
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_into_sections(text: str):
    """Split on '## Heading' or '### Heading' markers into (title, body) pairs."""
    pattern = re.compile(r"^#{2,3}\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    sections = []
    if not matches or matches[0].start() > 0:
        # capture any preamble (e.g. the summary paragraph) as its own section
        end = matches[0].start() if matches else len(text)
        preamble = text[:end].strip()
        if preamble:
            sections.append(("Overview / Summary", preamble))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _word_count(s: str) -> int:
    return len(s.split())


def _split_long_section(title: str, body: str) -> List[str]:
    """Break a long section into paragraph-aligned pieces of ~TARGET_WORDS,
    never exceeding MAX_WORDS, never splitting mid-paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces, current, current_words = [], [], 0

    for p in paragraphs:
        p_words = _word_count(p)
        if current_words + p_words > MAX_WORDS and current:
            pieces.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(p)
        current_words += p_words
        if current_words >= TARGET_WORDS:
            pieces.append("\n\n".join(current))
            current, current_words = [], 0

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def build_chunks() -> List[Chunk]:
    text = _load_source_text()
    sections = _split_into_sections(text)

    raw_pieces = []  # (title, body_text)
    for title, body in sections:
        wc = _word_count(body)
        if wc > MAX_WORDS:
            for j, piece in enumerate(_split_long_section(title, body), start=1):
                piece_title = title if j == 1 else f"{title} (cont. {j})"
                raw_pieces.append((piece_title, piece))
        else:
            raw_pieces.append((title, body))

    # Merge any too-short pieces into the following piece so chunks stay
    # semantically meaningful and within the target word-count range.
    merged: List[List] = []  # list of [title, text]
    for title, body in raw_pieces:
        prev_words = _word_count(merged[-1][1]) if merged else None
        combined_words = (prev_words + _word_count(body)) if merged else None
        if merged and prev_words < MIN_WORDS and combined_words <= MAX_WORDS:
            merged[-1][1] = merged[-1][1] + "\n\n" + body
            merged[-1][0] = f"{merged[-1][0]} + {title}"
        else:
            merged.append([title, body])

    chunks = []
    for i, (title, body) in enumerate(merged, start=1):
        chunks.append(Chunk(id=f"chunk_{i:02d}", title=title, text=body, word_count=_word_count(body)))
    return chunks


def save_chunks(chunks: List[Chunk], path: str = CHUNKS_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    return path


def main():
    chunks = build_chunks()
    path = save_chunks(chunks)
    print(f"[chunker] Built {len(chunks)} chunks -> {path}")
    for c in chunks:
        print(f"  {c.id:10s} [{c.word_count:4d} words]  {c.title}")


if __name__ == "__main__":
    main()
