"""The query pipeline: question (+ history) -> answer, sources and a trace.

    rewrite -> route -> embed (once) -> search each topic -> merge -> context -> generate -> citations

This module has no retrieval or answering logic of its own: it calls the stages in order, passes each output
to the next, decides where to stop, and keeps the trace. Four ways out:
    answer   - the normal case
    refusal  - at the router (no topic fits), after search (nothing found), or by the generator (evidence insufficient)
    input    - empty or oversized question, rejected before any API call
    error    - something broke; the user gets a plain message, the details go to the trace and the log
A failure in one topic's search is isolated: the other topics still answer.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src import config
from src.query.citations import Citations, Source, build_citations
from src.query.context import Context, build_context
from src.core.embeddings import embed_query
from src.query.generator import REFUSAL, Answer, generate_answer
from src.query.hybrid_search import HybridSearcher
from src.core.llm import track_usage
from src.query.merger import merge
from src.query.observability import QueryLog, Trace
from src.query.query_rewriter import rewrite_question
from src.query.router import TopicRouter
from src.query.table_query import build_catalog

log = logging.getLogger(__name__)

ERROR_TEXT = "Sorry, something went wrong while answering. Please try again."


@dataclass
class PipelineResult:
    question: str                         # what the user typed
    standalone_question: str              # what the pipeline searched with
    text: str                             # the answer, or the refusal / error message
    answerable: bool
    sources: list[Source] = field(default_factory=list)
    refusal_stage: str | None = None      # "input" | "router" | "retrieval" | "generator" | "error"
    error: str | None = None
    context: Context | None = None        # the chunks the generator saw (for UIs that show them)
    retrieved: list = field(default_factory=list)       # the merged top-k hits before the token budget (for evaluation)
    calculations: list = field(default_factory=list)    # table calculations that were run (evidence for faithfulness)
    topics: list[str] = field(default_factory=list)     # what the router chose
    trace: Trace | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question, "standalone_question": self.standalone_question, "answer": self.text,
            "answerable": self.answerable, "refusal_stage": self.refusal_stage, "error": self.error, "topics": self.topics,
            "sources": [s.display for s in self.sources],
            "trace": self.trace.to_dict() if self.trace else None,
        }


def _hit_summary(h) -> dict:
    return {"id": h.id, "label": h.label, "score": round(h.score, 5), "dense_rank": h.dense_rank,
            "dense_distance": None if h.dense_distance is None else round(h.dense_distance, 3), "bm25_rank": h.bm25_rank}


class RagPipeline:
    def __init__(
        self,
        router=None,
        searcher=None,
        rewrite: Callable = rewrite_question,
        embed: Callable = embed_query,
        generate: Callable = generate_answer,
        catalog: Callable = build_catalog,
        query_log: QueryLog | None = None,
        k: int | None = None,
        max_context_tokens: int | None = None,
        max_distance: float | None = None,
    ):
        self.router = router or TopicRouter()
        self.searcher = searcher or HybridSearcher()
        self._rewrite, self._embed, self._generate, self._catalog = rewrite, embed, generate, catalog
        self.query_log = query_log if query_log is not None else QueryLog()
        self.k = k or config.TOP_K
        self.max_context_tokens = max_context_tokens
        self.max_distance = max_distance

    # ------------------------------------------------------------------ public
    def answer(self, question: str, history: Sequence[dict] | None = None) -> PipelineResult:
        question = (question or "").strip()
        trace = Trace(question)
        with track_usage() as llm_calls:
            try:
                result = self._run(question, list(history or []), trace)
            except Exception as e:                                   # last line of defence: the caller never sees a traceback
                log.exception("pipeline failed")
                result = PipelineResult(question, question, ERROR_TEXT, False, refusal_stage="error",
                                        error=f"{type(e).__name__}: {e}")
        trace.finish(llm_calls)
        result.trace = trace
        self.query_log.write(result.to_dict())
        return result

    # ------------------------------------------------------------------ the stages
    def _stop(self, question, standalone, stage, text=REFUSAL, error=None, context=None) -> PipelineResult:
        return PipelineResult(question, standalone, text, False, refusal_stage=stage, error=error, context=context)

    def _run(self, question: str, history: list[dict], trace: Trace) -> PipelineResult:
        if not question:
            return self._stop(question, question, "input", "Please type a question.")
        if len(question) > config.MAX_QUESTION_CHARS:
            return self._stop(question, question, "input",
                              f"That question is too long (over {config.MAX_QUESTION_CHARS} characters). Please shorten it.")

        with trace.step("rewrite") as info:
            rw = self._rewrite(question, history)
            standalone = rw.question
            info.update(used_llm=rw.used_llm, changed=rw.changed, standalone=standalone, error=rw.error)

        with trace.step("route") as info:
            decision = self.router.route(standalone)
            info.update(topics=decision.topics, reason=decision.reason, fallback=decision.fallback,
                        dropped=list(decision.dropped))
        if not decision.topics:
            return self._stop(question, standalone, "router")
        routed = list(decision.topics)

        with trace.step("embed"):
            qvec = self._embed(standalone)                           # once, shared by every topic

        results, errors = {}, {}
        with trace.step("search") as info:
            for topic in decision.topics:
                try:
                    results[topic] = self.searcher.search(topic, standalone, k=self.k, query_vector=qvec)
                except Exception as e:                               # one broken topic must not sink the others
                    log.exception("search failed for topic %s", topic)
                    errors[topic] = f"{type(e).__name__}: {e}"
            info.update(per_topic={t: [_hit_summary(h) for h in hs] for t, hs in results.items()}, errors=errors)
            if errors and not results:
                raise RuntimeError(f"search failed for every topic: {errors}")

        with trace.step("merge") as info:
            merged = merge(results, k=self.k, max_distance=self.max_distance)
            info.update(candidates=merged.candidates, duplicates=merged.duplicates, too_far=merged.too_far,
                        final=[h.id for h in merged.hits])
        if not merged.hits:
            result = self._stop(question, standalone, "retrieval")
            result.topics = routed
            return result

        with trace.step("context") as info:
            ctx = build_context(merged.hits, self.max_context_tokens)
            info.update(items=[{"number": i.number, "label": i.label, "chunk_id": i.chunk_id, "tokens": i.tokens,
                                "truncated": i.truncated} for i in ctx.items], dropped=ctx.dropped, tokens=ctx.tokens)

        with trace.step("generate") as info:
            tables = self._catalog(decision.topics)                  # the full tables the answer may calculate on
            ans: Answer = self._generate(standalone, ctx, tables=tables)
            info.update(answerable=ans.answerable, used_chunks=ans.used_chunks, refusal_reason=ans.refusal_reason,
                        tables=[t.ref for t in tables], calculations=[c.to_dict() for c in ans.calculations],
                        pictures_shown=ans.pictures_shown,
                        prompt_tokens=ans.usage.prompt_tokens, output_tokens=ans.usage.output_tokens,
                        thinking_tokens=ans.usage.thinking_tokens)

        with trace.step("citations") as info:
            cites: Citations = build_citations(ans, ctx)
            info.update(sources=[s.display for s in cites.sources], invalid=cites.invalid,
                        invalid_calculations=cites.invalid_calculations, uncited=cites.uncited)

        return PipelineResult(
            question, standalone, cites.text or ans.text, ans.answerable, cites.sources,
            refusal_stage=None if ans.answerable else "generator", context=ctx,
            retrieved=list(merged.hits), calculations=list(ans.calculations), topics=list(decision.topics),
        )
