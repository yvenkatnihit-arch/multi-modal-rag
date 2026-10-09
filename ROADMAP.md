# Multi-Modal RAG Roadmap

Core idea: every file type is converted into ONE normalized record (text + metadata).
Chunking, embeddings, search and generation then treat all data the same way.

Legend: [x] done, [ ] to do

## Phase 0: Foundation
- [x] 0.1 Skeleton, requirements.txt, config.py, check_setup.py (venv + API key verified)
- [x] 0.1b Data prepared: 6 topics in data/ built by scripts/prepare_data.py (re-runnable);
      ground-truth facts for generated files in scripts/generated_facts.json
- [x] 0.2 topic_registry: folder name = topic id, plus a description per topic (what kinds of data it holds)

## Phase 1: Normalized record + file_router (build step 1)
- [x] 1.1 Record dataclass: text + metadata {topic_id, source, modality, page, table_id, image_path}
- [x] 1.2 file_router: detect type by extension + content, log and skip unsupported files
- [x] 1.3a text_parser (txt/md/log/html/docx -> one Record per section, heading paths)
- [x] 1.3b basic PDF text parser (page-by-page, strip references)
- [x] 1.3c chunker per modality (1000 chars / 200 overlap); stable chunk IDs (topic/file/page/chunk#)

## Phase 2: Baseline text RAG (fixes the known issues)
- [x] 2.1 vector_store (+ embeddings.py, ingest.py glue): Gemini embeddings, one Chroma collection per topic, idempotent upsert
      (no delete-whole-collection before re-embedding)
- [x] 2.2 hybrid_search: dense + BM25 fused with RRF, BM25 index cached per topic, optional modality filter
- [x] 2.3 router (structured JSON output + safe fallback) and query_rewriter (follow-up -> standalone)
- [x] 2.4 merger: cross-topic RRF that keeps scores, plus dedup
- [x] 2.5 context (labelled chunks, token budget), generator (temp 0, grounded, refuses),
      citations (only chunks actually used)
- [x] 2.6 pipeline.answer(), observability (timing/tokens per question), ask.py terminal chat
- CHECKPOINT: text-only RAG works end to end

## Phase 3: Tables (build step 2)
- NOTE: order changed at the user's request: 3.1 (table_parser) and 4.1 (image_parser) are built BEFORE 2.2 (hybrid_search)
- [x] 3.1 table_parser (CSV/Excel/JSON): schema summary, column stats, row-group chunks with
      header repeated; big tables sampled; full table stored in assets/
- [x] 3.2 table_query tool: sum/count/average computed with pandas on the stored table

## Phase 4: Images (build step 3)
- [x] 4.1 image_parser (+ vision.py, gemini_client.py shared client/retry; analyses cached by file hash): Gemini vision -> caption + OCR text + chart/diagram description; original saved to assets/
- [x] 4.2 OCR fallback for scanned PDF pages with no text layer

## Phase 5: Rich PDFs (build step 4)
- [x] 5.1 pdf_parser: tables via find_tables -> table logic
- [x] 5.2 pdf_parser: embedded images -> image parser

## Phase 6: Multimodal answering (build step 5)
- [x] 6.1 Chunk labels with modality: [text p.3] [table sales.csv] [image fig2.png]   (context.py, since 2.5)
- [x] 6.2 Generator sends relevant images to Gemini as images; text/tables as text      (pictures.py, generator.py)
- [x] 6.3 Citations show file, page/table and modality                                  (citations.py)

## Phase 7: Evaluation + UI (build step 6)
- [x] 7.1 evaluation/testset.json (21 questions, 18 categories) + retrieval_metrics.py (precision/recall) + run_eval.py
- [x] 7.2 generation_metrics.py: faithfulness, correctness, answer relevance (LLM judges; DeepEval deliberately NOT used yet)
- [x] 7.3 Streamlit app.py: chat, sources panel (with pictures), per-question trace view
- [ ] 7.2b DeepEval: DEFERRED ON PURPOSE until you decide (the question set is small to limit API cost)

## After the plan (extra work done along the way)
- [x] Bug fixed: invented catalog value (keyword search: identifiers boosted, plurals, camelCase; generator: partial-answer rule)
- [x] Reorganised by workflow: src/core · src/ingestion · src/query (+ tests mirrored, boundary enforced by test_architecture.py)
- [x] README.md, pyrightconfig.json, complete requirements.txt

## Decisions and ideas still open (not part of the plan)
- [ ] DeepEval: use it or not
- [ ] A full judged evaluation run on all 21 questions (costs real tokens)
- [ ] Auto-generated topic profiles for the router (the descriptions are hand-written and go stale)

## Known issues from the requirements: all fixed
- [x] Unstable chunk_id (global index)          -> Phase 1.3
- [x] Collection deleted before re-embedding    -> Phase 2.1
- [x] BM25 index rebuilt per query              -> Phase 2.2
- [x] Fragile router JSON parsing               -> Phase 2.3
- [x] Cross-topic merge discards scores         -> Phase 2.4
- [x] Citations list every chunk sent to LLM    -> Phase 2.5

## Environment notes
- Every entry point must import src.config before chromadb (it stubs the gRPC telemetry
  module that Windows Application Control blocks).
- Activate the venv with: .\.venv\Scripts\Activate.ps1
