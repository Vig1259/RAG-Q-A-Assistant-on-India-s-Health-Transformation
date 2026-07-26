"""
embeddings.py
-------------
Pluggable embedding backends so the pipeline can run:
  (a) fully offline, with zero downloads/API keys (TF-IDF backend), and
  (b) with real dense sentence embeddings for better semantic search
      (Sentence-Transformers backend), when the user has internet access.

Both backends implement the same tiny interface:
    fit(texts: List[str]) -> None
    transform(texts: List[str]) -> np.ndarray   # shape (n, d), L2-normalized
    save(path) / load(path)

Backend is chosen via the EMBEDDING_BACKEND environment variable
("tfidf" | "sbert"), defaulting to "tfidf" so `python -m health_rag.cli`
works out of the box with no setup.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class TfidfEmbedder:
    """Offline baseline: TF-IDF vectors + cosine similarity.

    Not a "semantic" embedding in the dense-vector-neural-net sense, but it
    satisfies the same interface and requires no model download or API key,
    which makes the whole pipeline runnable anywhere, instantly. Swap in
    SbertEmbedder for real semantic embeddings (see README).
    """

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        return _l2_normalize(mat)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True


class SbertEmbedder:
    """Real dense semantic embeddings using Sentence-Transformers.

    Requires: pip install sentence-transformers
    Downloads the model (~80MB) on first run; needs internet access once.
    Recommended model: 'all-MiniLM-L6-v2' — small, fast, strong quality for
    short/medium passages, widely used as a RAG default.
    """

    name = "sbert"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> None:
        pass  # no fitting step needed; the model is pre-trained

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(mat.astype("float32"))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_name)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            self.model_name = f.read().strip()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)


def get_embedder(backend: str | None = None):
    backend = backend or os.environ.get("EMBEDDING_BACKEND", "tfidf")
    if backend == "sbert":
        return SbertEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'. Use 'tfidf' or 'sbert'.")
