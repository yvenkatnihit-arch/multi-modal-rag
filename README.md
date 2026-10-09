# Multi-modal RAG

Ask questions about a mixed pile of files (text, markdown, HTML, Word, CSV/Excel/JSON tables, PDFs with tables and
figures, scanned pages, photos). Answers use only your documents, cite their sources (file, page or table, modality, and
the picture itself), do exact arithmetic over tables, and decline when the documents don't contain the answer.

Stack: Python · Google Gemini (`gemini-2.5-flash` for language and vision, `gemini-embedding-001`) · ChromaDB ·
rank-bm25 · PyMuPDF · pandas · Streamlit.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env:  GEMINI_API_KEY=...
python check_setup.py           # checks the key, the models, ChromaDB and the libraries
```

## Use it

| Command | What it does |
|---|---|
| `python ingest.py` | Reads `data/<topic>/` into the search index. Re-runnable: unchanged files cost nothing. |
| `python ask.py` | Terminal chat with memory (`/trace`, `/topics`, `/clear`, `/quit`). `python ask.py "question"` for one answer. |
| `streamlit run app.py` | The web UI: chat, sources with pictures, and a trace of every answer. |
| `python -m pytest tests` | The tests (no network, no API cost; any real Gemini call fails loudly). |
| `python -m evaluation.run_eval --retrieval-only` | Scores the search only. **No model calls**, embeddings only. |
| `python -m evaluation.run_eval --no-judge` | Whole pipeline on the question set, without the LLM judges. |
| `python -m evaluation.run_eval` | Everything, including the judges (faithfulness, correctness, answer relevance). Costs real tokens. |
| `python -m evaluation.check_routing` | Scores the topic router and the question rewriter. |

## Where things are

```
data/<topic>/         your files: one folder per topic, a topic.json (name + description) in each
assets/               generated: full tables, image copies, scanned pages, PDF figures, cached vision answers
chroma_db/  logs/     generated: the vector index; one JSON line per question asked

src/
  config.py           settings (imported first by everything)
  core/               shared by both pipelines
    records.py          the one shape all content takes (Record, Chunk, Metadata)
    assets.py           where originals are saved and found; the table manifest
    vector_store.py     ChromaDB: one collection per topic, synced (never rebuilt)
    embeddings.py       text -> vectors          topic_registry.py   which topics exist and what they hold
    llm.py              structured Gemini calls  gemini_client.py    shared client with retries
    images.py           preparing an image to send  table_types.py   what counts as a number or a date
  ingestion/          INGESTION: data -> searchable store
    file_router.py      what kind of file is this?     chunker.py   cut into pieces with stable ids
    parsers/            text_parser · pdf_parser (text, scans, tables, figures) · table_parser · image_parser
    vision.py           Gemini reads an image, with a cache      ingest.py   runs it all, topic by topic
  query/              QUERY: question -> answer
    query_rewriter.py   follow-up -> standalone question        router.py    which topics can answer
    hybrid_search.py    meaning + keywords, fused (RRF)         merger.py    many topics -> one ranked list
    context.py          numbered, labelled evidence block       pictures.py  the original images to show the model
    generator.py        grounded answer, or refusal             table_query.py  exact calculations on full tables
    citations.py        only what the answer really used        pipeline.py  runs the steps, keeps the trace
    observability.py    per-question trace and log

evaluation/           question set, retrieval metrics, generation metrics, runner, results/
scripts/              prepare_data.py (builds data/ from the Kaggle downloads)
tests/                mirrors src/: core/ ingestion/ query/ evaluation/ interface/  (+ test_architecture.py)
app.py  ask.py  ingest.py  check_setup.py     entry points
```

The two pipelines never import each other (`tests/test_architecture.py` enforces it). They meet only in what ingestion
writes (the vector store and `assets/`) and what the query side reads.

## How a question is answered

```
question + history -> rewrite (follow-ups) -> route (which topics?) -> embed -> hybrid search per topic
  -> merge -> numbered evidence (+ original pictures) -> grounded answer (+ exact table calculations) -> citations
```

Three places can decline: the router (nothing fits), the search (nothing found), the answer step (evidence doesn't
support an answer). Every question leaves a trace: timings, decisions, tokens (`logs/queries.jsonl`, `--trace`, the UI).

## Good to know

- **The router only knows the topic descriptions.** If a topic holds something its `topic.json` does not mention, the
  router may decline questions about it. Keep the descriptions up to date when you add files.
- **Cost:** a question costs 2–4 model calls. A full judged evaluation costs hundreds of thousands of tokens;
  `--retrieval-only` and `--no-judge` are the cheap modes.
- Known limits: charts drawn as vector graphics, borderless tables and tables split across pages are not extracted.
