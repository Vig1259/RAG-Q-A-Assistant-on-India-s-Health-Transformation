# Implementation Note

**System:** RAG Q&A over the PIB backgrounder "India's Health Transformation"
**Timebox:** ~7 hours over 2 days

## 1. Embedding model and why

I made embeddings **pluggable** rather than picking a single fixed model,
because the two realistic environments for this assignment (a reviewer's
laptop with internet + API keys vs. a sandboxed/offline grading environment)
have very different constraints:

- **Default: TF-IDF (scikit-learn)** — a sparse, lexical vector space
  (unigrams + bigrams, English stopwords removed). It requires no model
  download, no API key, and no network access, so the pipeline is
  guaranteed to run anywhere, instantly. For a single ~4,500-word document
  with distinctive scheme names ("AB-PMJAY", "PM-ABHIM", "Tele-MANAS"),
  lexical overlap is already a strong signal — most questions repeat scheme
  names or close synonyms of the chunk they need.
- **Optional: `all-MiniLM-L6-v2` (Sentence-Transformers)** — a small (~80MB),
  fast, widely-used dense sentence embedding model that captures actual
  semantic similarity (e.g. matching "how much does the government spend on
  hospital infrastructure" to the PM-ABHIM chunk even without exact word
  overlap). I chose MiniLM specifically because it's the de-facto default
  for small RAG projects: good quality/speed tradeoff, runs on CPU, and
  needs no API key (unlike OpenAI/Cohere embeddings).

Both backends share one interface (`fit`/`transform`) and produce
L2-normalized vectors, so switching backends is a one-line env var change
(`EMBEDDING_BACKEND=sbert`) and doesn't touch the retrieval or RAG code.

## 2. Storage/index for embeddings

Given the corpus is a single small document (15-16 chunks), I used a
minimal flat-file index instead of a vector database:
- `embeddings.npy` — the raw (n_chunks × dim) matrix, L2-normalized so
  cosine similarity reduces to a dot product / matrix multiply.
- `chunks.json` — chunk id, title, text, word count (the retrievable units).
- `embedder.pkl` — the fitted TF-IDF vectorizer (vocabulary + IDF weights)
  or the sbert model name, so query-time embeddings use the exact same
  vector space as the indexed chunks.

This intentionally has the same interface a real vector DB (FAISS, Chroma,
pgvector) would expose — `store.search(query, top_k)` — so swapping in one
of those for a larger corpus wouldn't require changing `rag.py` at all.

## 3. LLM and prompt design

Three interchangeable generation backends, tried in priority order:
1. Anthropic API (`claude-sonnet-4-5`) if `ANTHROPIC_API_KEY` is set.
2. OpenAI API (`gpt-4o-mini`) if `OPENAI_API_KEY` is set.
3. **Extractive fallback** (default, no key required): ranks sentences from
   the retrieved chunks by lexical overlap with the question and returns the
   top few, verbatim, with the source chunk title attached. This is the
   backend the checked-in demo actually runs on, since it guarantees the
   whole pipeline is testable and hallucination-free with zero
   configuration.

**Prompt design:** the system prompt explicitly restricts the model to the
provided context, instructs it to say "the document does not cover this"
rather than guess, and asks for a short answer (a few sentences or a short
bullet list) rather than a long essay. The user prompt labels each retrieved
chunk with its section title (`[Context N: <title>]`) so the model — and a
human reading the prompt — can see exactly which part of the document each
piece of context came from.

## 4. What I had to learn/research

- Confirmed the PIB page's actual HTML structure (heading text is rendered
  as bold `<strong>`/`<p>` text rather than semantic `<h2>`/`<h3>` tags in
  places), which is why `ingest.py` treats short bold runs as headings too.
- Checked that `sentence-transformers`' `all-MiniLM-L6-v2` needs no API key
  and downloads once — relevant because the assignment specifies "any
  embedding model," and I wanted the default path to need zero setup while
  still documenting a real semantic-embedding upgrade path.
- Verified that TF-IDF vectors can be safely L2-normalized so the same
  cosine-similarity code path works for both backends without branching.

## 5. Limitations and what I'd improve with 2 more days

- **TF-IDF is lexical, not semantic.** It will miss paraphrased questions
  that don't share vocabulary with the source chunk (e.g. "how does India
  make sure poor people don't skip cancer treatment" won't match "AB-PMJAY"
  as well as a true embedding model would). Shipping `sbert` as the default
  (once network access for the model download is guaranteed) would fix this.
- **Extractive fallback answers are not fluent.** They're correct, grounded
  quotes, but read as a bullet list of sentences rather than a synthesized
  answer. With guaranteed LLM access, the Anthropic/OpenAI path already
  produces much more natural answers — this is really a "graceful
  degradation" feature, not the intended primary experience.
- **One chunk (the NHM overview) ended up under the 200-word target**
  because merging it with its neighbor would have pushed that neighbor over
  500 words. I'd add a secondary merge pass that also considers merging
  backwards, or slightly relax the max when the alternative is an
  isolated ~100-word chunk.
- **No re-ranking step.** For a bigger document I'd add a cheap re-ranker
  (e.g. cross-encoder) on the top-10 candidates before picking the final
  top-k, since TF-IDF/embedding similarity alone can rank a tangentially
  related chunk above a more relevant one.
- **No answer citations back to exact sentences**, only to chunk titles.
  For a more auditable system I'd have the LLM cite which sentence(s)
  within each chunk it used.
- **No automated eval set.** Given more time I'd write ~15 question/answer
  pairs with expected source chunks and measure retrieval@k accuracy, to
  catch regressions when tuning chunking or switching embedding backends.
