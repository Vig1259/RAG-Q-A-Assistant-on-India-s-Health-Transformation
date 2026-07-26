"""
cli.py
------
Minimal command-line interface.

Usage:
    python -m health_rag.cli                 # interactive loop
    python -m health_rag.cli "your question"  # one-shot
    python -m health_rag.cli --backend gemini "your question"  # force a backend

Shows the answer plus which chunks (titles + a short snippet + similarity
score) were used, so the user can verify the grounding.
"""
from __future__ import annotations

import sys

from .rag import answer_question
from .vector_store import load_or_build


def _print_result(question: str, result) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(f"A ({result.backend_used}):\n{result.answer}")
    print("-" * 70)
    print("Sources used:")
    for chunk, score in result.sources:
        snippet = chunk.text.replace("\n", " ")[:140]
        print(f"  [{score:.3f}] {chunk.title} — \"{snippet}...\"")
    print("=" * 70 + "\n")


def _parse_args(argv):
    backend = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return backend, " ".join(rest)


def main():
    print("Loading index (building it on first run)...")
    store = load_or_build()
    print(f"Ready. {len(store.chunks)} chunks indexed with backend='{store.embedder.name}'.\n")

    if len(sys.argv) > 1:
        backend, question = _parse_args(sys.argv[1:])
        result = answer_question(question, store=store, force_backend=backend)
        _print_result(question, result)
        return

    print("Ask a question about India's Health Transformation (PIB backgrounder).")
    print("Type 'exit' or 'quit' to stop. Prefix with '@gemini ', '@groq ', etc. to force a backend.\n")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            break

        backend = None
        question = raw
        if raw.startswith("@"):
            first_word, _, remainder = raw.partition(" ")
            backend = first_word[1:]
            question = remainder.strip()
            if not question:
                print(f"(usage: @{backend} your question here)")
                continue

        result = answer_question(question, store=store, force_backend=backend)
        _print_result(question, result)


if __name__ == "__main__":
    main()