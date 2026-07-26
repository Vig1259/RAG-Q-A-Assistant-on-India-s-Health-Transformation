"""
vector_store.py
----------------
A minimal, dependency-light vector index:
  - stores chunk embeddings as a single .npy matrix (already L2-normalized)
  - stores chunk metadata (id, title, text) as chunks.json
  - stores the fitted embedder (TF-IDF vocab, or sbert model name) so query
    embeddings are produced with the exact same vectorizer/model
  - cosine similarity == dot product, since vectors are L2-normalized

This is intentionally simple (no external vector DB) since the corpus is a
single ~4,500-word document -> a few dozen chunks. For larger corpora, swap
this module for FAISS/Chroma/pgvector without changing rag.py's interface
(search() -> List[(chunk, score)]).
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

from .chunker import Chunk, build_chunks, CHUNKS_PATH
from .embeddings import get_embedder

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(THIS_DIR), "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
EMB_MATRIX_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
EMBEDDER_STATE_PATH = os.path.join(INDEX_DIR, "embedder.pkl")
META_PATH = os.path.join(INDEX_DIR, "meta.json")


class VectorStore:
    def __init__(self, backend: str | None = None):
        self.backend = backend
        self.embedder = None
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None

    # ---------- build ----------
    def build(self, chunks: List[Chunk] | None = None) -> None:
        self.chunks = chunks or build_chunks()
        self.embedder = get_embedder(self.backend)
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.transform(texts)

    def save(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        np.save(EMB_MATRIX_PATH, self.matrix)
        self.embedder.save(EMBEDDER_STATE_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"backend": self.embedder.name, "n_chunks": len(self.chunks)}, f, indent=2)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f, ensure_ascii=False, indent=2)

    # ---------- load ----------
    def load(self) -> None:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.embedder = get_embedder(meta["backend"])
        self.embedder.load(EMBEDDER_STATE_PATH)
        self.matrix = np.load(EMB_MATRIX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [Chunk(**c) for c in raw]

    def exists(self) -> bool:
        return all(os.path.exists(p) for p in [EMB_MATRIX_PATH, EMBEDDER_STATE_PATH, META_PATH, CHUNKS_PATH])

    # ---------- search ----------
    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        q_vec = self.embedder.transform([query])[0]
        scores = self.matrix @ q_vec  # cosine similarity (both sides L2-normalized)
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


def build_and_save(backend: str | None = None) -> VectorStore:
    store = VectorStore(backend=backend)
    store.build()
    store.save()
    return store


def load_or_build(backend: str | None = None) -> VectorStore:
    store = VectorStore(backend=backend)
    if store.exists():
        store.load()
    else:
        store.build()
        store.save()
    return store


def main():
    store = build_and_save()
    print(f"[vector_store] Indexed {len(store.chunks)} chunks with backend='{store.embedder.name}'.")
    print(f"[vector_store] Embedding matrix shape: {store.matrix.shape}")


if __name__ == "__main__":
    main()
