"""
ingest.py
---------
Downloads the PIB backgrounder page ("India's Health Transformation") and
converts it into clean, plain markdown text ready for chunking.

Why this design:
- The PIB page is a plain HTML press release with headings (bold text acting
  as section titles), bullet lists, and inline images. There is no PDF version
  that is easier to parse, so we scrape the HTML and strip markup/boilerplate
  (nav links, image tags, the giant "References" link list, the Hindi footer,
  and a duplicated raw HTML table that PIB renders at the bottom of the page).
- If the network is unavailable (e.g. running in a sandboxed environment),
  ingest.py falls back to the pre-fetched, cleaned copy already committed at
  data/raw/pib_health_transformation.md. This keeps the rest of the pipeline
  runnable end-to-end without a live internet connection.

Usage:
    python -m health_rag.ingest
"""
from __future__ import annotations

import os
import re
import sys

URL = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(THIS_DIR), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHED_CLEAN_MD = os.path.join(RAW_DIR, "pib_health_transformation.md")


def _strip_boilerplate(text: str) -> str:
    """Remove reference-link dumps, Hindi footer, and duplicate tables that
    PIB pages append after the main article body."""
    # Cut everything from the "References" section onward — it is just a
    # long list of outbound links, not article content.
    cut_markers = ["**References**", "References\n", "प्रविष्टि तिथि", "PIB Research"]
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def fetch_and_clean(url: str = URL) -> str:
    """Fetch the live page and return cleaned markdown text.

    Requires `requests` and `beautifulsoup4` (see requirements.txt) and a
    working internet connection. Falls back to the cached copy if either the
    dependency or the network is unavailable, so the pipeline never breaks.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # The article body on PIB pages lives inside a content div with the
        # heading text and paragraphs; images/scripts are dropped, and bold
        # section headers are preserved as "## Header" markers so the
        # chunker can split on them.
        for tag in soup(["script", "style", "img", "nav", "footer"]):
            tag.decompose()

        lines = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li", "b", "strong"]):
            txt = el.get_text(" ", strip=True)
            if not txt:
                continue
            if el.name in ("h1", "h2", "h3") or (el.name in ("b", "strong") and len(txt) < 90):
                lines.append(f"\n## {txt}\n")
            elif el.name == "li":
                lines.append(f"- {txt}")
            else:
                lines.append(txt)

        text = "\n".join(lines)
        text = _strip_boilerplate(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text.split()) < 300:
            # Looks like scraping failed to get real content; use cache.
            raise RuntimeError("Fetched page looked too short — falling back to cache.")

        return text

    except Exception as exc:  # network unavailable, dependency missing, parse failure, etc.
        print(f"[ingest] Live fetch unavailable ({exc}). Using cached copy instead.", file=sys.stderr)
        with open(CACHED_CLEAN_MD, "r", encoding="utf-8") as f:
            return f.read()


def save(text: str, out_path: str | None = None) -> str:
    out_path = out_path or os.path.join(RAW_DIR, "pib_health_transformation.clean.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def main():
    text = fetch_and_clean()
    path = save(text)
    print(f"[ingest] Saved cleaned document to {path} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
