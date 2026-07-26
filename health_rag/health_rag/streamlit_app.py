"""
streamlit_app.py
----------------
Optional Streamlit UI — another minimal interface alongside the CLI
(cli.py) and the Flask app (app.py). Same underlying rag.answer_question()
call, so behavior/grounding is identical; this is purely a presentation
layer with a nicer chat-style look and a sidebar showing which backend
(Anthropic / OpenAI / Gemini / extractive fallback) is active.

Run:
    streamlit run health_rag/streamlit_app.py
Then it opens automatically at http://localhost:8501
"""
import os
import sys

# `streamlit run health_rag/streamlit_app.py` executes this file as a
# standalone script (not as part of the health_rag package), so relative
# imports (`from .rag import ...`) would fail here. Add the project root to
# sys.path and import absolutely instead, so the file works both when run
# directly by Streamlit and when imported as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from health_rag.rag import answer_question
from health_rag.vector_store import load_or_build

st.set_page_config(page_title="India's Health Transformation — RAG Q&A", page_icon="🩺")


@st.cache_resource(show_spinner="Loading / building the chunk + embedding index...")
def get_store():
    return load_or_build()


def configured_backends() -> list[str]:
    """Which backends have an API key present in the environment right now."""
    mapping = [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
    ]
    return [name for name, env_key in mapping if os.environ.get(env_key)]


st.title("🩺 India's Health Transformation")
st.caption(
    "Ask questions grounded in the PIB backgrounder "
    "([source](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2)). "
    "Answers use only retrieved chunks from this document."
)

with st.sidebar:
    st.subheader("Index status")
    store = get_store()
    st.write(f"**Chunks indexed:** {len(store.chunks)}")
    st.write(f"**Embedding backend:** `{store.embedder.name}`")

    st.divider()
    st.subheader("Answer backend")
    available = configured_backends()
    if available:
        options = ["(Auto)"] + available
        choice = st.selectbox(
            "Which LLM should generate the answer?",
            options,
            help="'(Auto)' uses the first configured key in priority order: "
                 "anthropic > openai > gemini > groq. Only backends with an "
                 "API key set in your .env file appear here.",
        )
        selected_backend = None if choice == "(Auto)" else choice
    else:
        st.info(
            "No LLM API key found in .env — answers will use the "
            "extractive fallback (verbatim sentences from the source, no LLM)."
        )
        selected_backend = None

    st.divider()
    st.subheader("Try one of these")
    sample_questions = [
        "What are the four pillars of Ayushman Bharat?",
        "How much financial outlay does PM-ABHIM have?",
        "What is ABHA and how does ABDM work?",
        "How has malaria mortality changed in India?",
        "What AI tools does the government use in healthcare?",
        "What does the Eat Right India movement do?",
    ]
    clicked_sample = None
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            clicked_sample = q

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, RAGResult)

question = st.chat_input("Ask a question about India's health transformation...")
question = question or clicked_sample

if question:
    with st.spinner("Searching the document and generating an answer..."):
        result = answer_question(question, store=store, force_backend=selected_backend)
    st.session_state.history.append((question, result))

for q, result in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(result.answer)
        st.caption(f"Answered via: `{result.backend_used}`")
        with st.expander("Sources used"):
            for chunk, score in result.sources:
                st.markdown(f"**[{score:.3f}] {chunk.title}**")
                st.write(chunk.text[:300] + "...")
                st.divider()

if not st.session_state.history:
    st.info("Ask a question above, or click a sample question in the sidebar to get started.")