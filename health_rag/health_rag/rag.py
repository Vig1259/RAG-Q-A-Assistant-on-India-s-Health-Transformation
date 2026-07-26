"""
rag.py
------
Ties retrieval + generation together:

  1. Embed the user's question and retrieve the top-k most similar chunks
     from the VectorStore (semantic search).
  2. Build a strict, grounding-only prompt containing ONLY those chunks as
     context.
  3. Call an LLM to produce a short answer. Three backends are supported,
     tried in this order based on what's configured in the environment:
       - Anthropic API (ANTHROPIC_API_KEY set)          -> claude-*-latest
       - OpenAI API (OPENAI_API_KEY set)                -> gpt-4o-mini
       - Extractive fallback (no key needed, always works)

The extractive fallback never calls any external API: it simply returns the
most relevant sentences from the retrieved chunks, verbatim, so the system
is always runnable and never hallucinates, even with zero configuration.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from .chunker import Chunk
from .vector_store import VectorStore, load_or_build

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file at the project root, if present
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment variables only


def _patch_httpx_proxies_bug():
    """Work around a known compatibility bug: the openai/groq Python SDKs
    (both generated from the same Stainless template) hard-code a `proxies`
    kwarg when constructing their internal httpx.Client, but httpx>=0.28
    removed that parameter entirely, causing:
        TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
    This patches httpx.Client/AsyncClient to silently drop `proxies` if the
    installed httpx version doesn't accept it, so the SDKs work regardless
    of which httpx version ends up installed in a given environment.
    Safe no-op if httpx already supports `proxies`, or isn't installed.
    """
    try:
        import inspect
        import httpx

        for cls in (httpx.Client, httpx.AsyncClient):
            params = inspect.signature(cls.__init__).parameters
            if "proxies" in params:
                continue  # this httpx version still supports it natively

            original_init = cls.__init__

            def patched_init(self, *args, __orig=original_init, **kwargs):
                kwargs.pop("proxies", None)
                __orig(self, *args, **kwargs)

            cls.__init__ = patched_init
    except Exception:
        pass  # best-effort; if this fails, the original error will surface as before


_patch_httpx_proxies_bug()

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file at the project root, if present
except ImportError:
    pass  # python-dotenv not installed -- env vars must be exported manually

TOP_K_DEFAULT = 4

SYSTEM_PROMPT = (
    "You are a question-answering assistant that ONLY uses the provided "
    "context, drawn from a PIB (Press Information Bureau, India) backgrounder "
    "titled 'India's Health Transformation'. "
    "Answer the user's question using ONLY facts present in the context. "
    "If the context does not contain the answer, say so explicitly instead "
    "of guessing. Must return a short, clear answer that uses information from the "
    "PIB document at start or end. Keep it to a few sentences or a short bullet "
    "list), factual, and do not invent any numbers, dates, or scheme names "
    "that are not in the context."
    "If the context does not contain the answer, say so explicitly instead "
    "of guessing. Keep the answer short (a few sentences or a short bullet "
    "list), factual, and do not invent any numbers, dates, or scheme names "
    "that are not in the context."
)


@dataclass
class RAGResult:
    answer: str
    sources: List[Tuple[Chunk, float]]
    backend_used: str


def build_prompt(question: str, sources: List[Tuple[Chunk, float]]) -> str:
    context_blocks = []
    for i, (chunk, score) in enumerate(sources, start=1):
        context_blocks.append(f"[Context {i}: {chunk.title}]\n{chunk.text}")
    context = "\n\n".join(context_blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If it's not answered there, "
        "say the document does not cover it."
    )


def _extractive_fallback(question: str, sources: List[Tuple[Chunk, float]]) -> str:
    """No-LLM fallback: rank sentences across retrieved chunks by lexical
    overlap with the question, and return the best few, verbatim. Guarantees
    zero hallucination since nothing is generated -- it's a direct quote."""
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    scored_sentences = []
    for chunk, chunk_score in sources:
        # Avoid splitting on abbreviations like "Rs." (rupees) which are not
        # sentence boundaries in this document.
        protected = chunk.text.replace("Rs.", "Rs\u2024")
        sentences = re.split(r"(?<=[.!?])\s+", protected)
        sentences = [s.replace("\u2024", ".") for s in sentences]
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean.split()) < 5:
                continue
            s_words = set(re.findall(r"[a-zA-Z]+", s_clean.lower()))
            overlap = len(q_words & s_words)
            scored_sentences.append((overlap + chunk_score, s_clean, chunk.title))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top = scored_sentences[:3] if scored_sentences else []
    if not top:
        return "The document does not appear to cover this question."

    lines = [f"(No LLM API key configured -- showing the most relevant sentences "
             f"directly from the source document.)\n"]
    for _, sentence, title in top:
        lines.append(f"- {sentence} [{title}]")
    return "\n".join(lines)


def _call_anthropic(question: str, sources) -> str:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = build_prompt(question, sources)
    msg = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


def _call_openai(question: str, sources) -> str:
    from openai import OpenAI  # pip install openai
    client = OpenAI()  # reads OPENAI_API_KEY from env
    prompt = build_prompt(question, sources)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _call_groq(question: str, sources) -> str:
    """Groq: free tier, no credit card. Uses Groq's own native SDK (not the
    OpenAI-compatible client) to avoid an unrelated openai-python/httpx
    version-compatibility bug some environments hit.
    Sign up at https://console.groq.com/keys"""
    from groq import Groq  # pip install groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = build_prompt(question, sources)
    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _call_gemini(question: str, sources) -> str:
    from google import genai  # pip install google-genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = build_prompt(question, sources)
    resp = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=400,
            temperature=0.2,
        ),
    )
    return resp.text


def answer_question(
    question: str,
    store: VectorStore | None = None,
    top_k: int = TOP_K_DEFAULT,
    force_backend: str | None = None,
) -> RAGResult:
    store = store or load_or_build()
    sources = store.search(question, top_k=top_k)

    # Priority: explicit function argument > LLM_BACKEND env var > first
    # configured key found, in this fixed order.
    forced = force_backend or os.environ.get("LLM_BACKEND")

    backends = [
        ("anthropic", "ANTHROPIC_API_KEY", _call_anthropic),
        ("openai", "OPENAI_API_KEY", _call_openai),
        ("gemini", "GEMINI_API_KEY", _call_gemini),
        ("groq", "GROQ_API_KEY", _call_groq),
    ]
    if forced:
        backends = [b for b in backends if b[0] == forced]
        if not backends:
            return RAGResult(
                f"Unknown backend '{forced}'. Choose one of: anthropic, openai, gemini, groq.",
                sources,
                "error",
            )

    for name, env_key, fn in backends:
        if os.environ.get(env_key):
            try:
                return RAGResult(fn(question, sources), sources, name)
            except Exception as e:
                print(f"[rag] {name} call failed ({e}); trying next backend.")

    if forced:
        return RAGResult(
            f"'{forced}' was requested but {forced.upper()}_API_KEY is not set in .env.",
            sources,
            "error",
        )
    return RAGResult(_extractive_fallback(question, sources), sources, "extractive-fallback")