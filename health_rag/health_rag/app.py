"""
app.py
------
Optional minimal web UI (single page, no build step) as an alternative to
the CLI. Same underlying rag.answer_question() call, so behavior is
identical -- this is purely a presentation layer.

Run:
    python -m health_rag.app
Then open http://localhost:5000
"""
from __future__ import annotations

from flask import Flask, render_template_string, request

from .rag import answer_question
from .vector_store import load_or_build

app = Flask(__name__)
_store = None  # lazy-loaded on first request


def get_store():
    global _store
    if _store is None:
        _store = load_or_build()
    return _store


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>India's Health Transformation — RAG Q&A</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 760px;
           margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    form { display: flex; gap: 8px; margin: 20px 0; }
    input[type=text] { flex: 1; padding: 10px; font-size: 1rem; }
    button { padding: 10px 18px; font-size: 1rem; cursor: pointer; }
    .answer { background: #f4f6f8; border-radius: 8px; padding: 16px; white-space: pre-wrap; }
    .meta { color: #666; font-size: 0.85rem; margin-top: 4px; }
    .sources { margin-top: 16px; }
    .source { border-left: 3px solid #ccc; padding-left: 10px; margin-bottom: 10px; font-size: 0.9rem; }
    .score { color: #888; }
  </style>
</head>
<body>
  <h1>Ask about India's Health Transformation</h1>
  <p class="meta">Grounded in the PIB backgrounder (PRID=2269699). Answers use only retrieved chunks.</p>
  <form method="post">
    <input type="text" name="question" placeholder="e.g. What is PM-ABHIM?" value="{{ question or '' }}" autofocus>
    <button type="submit">Ask</button>
  </form>

  {% if result %}
  <div class="answer"><strong>Answer</strong> <span class="meta">({{ result.backend_used }})</span><br><br>{{ result.answer }}</div>
  <div class="sources">
    <strong>Sources used:</strong>
    {% for chunk, score in result.sources %}
      <div class="source"><span class="score">[{{ '%.3f'|format(score) }}]</span> <strong>{{ chunk.title }}</strong><br>{{ chunk.text[:220] }}...</div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    question = None
    result = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            result = answer_question(question, store=get_store())
    return render_template_string(PAGE, question=question, result=result)


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
