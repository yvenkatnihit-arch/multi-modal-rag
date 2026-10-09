"""Streamlit UI for the multi-modal RAG.

    streamlit run app.py

Chat with memory, the sources behind every answer (with the actual pictures), and a trace of how each answer was made.
"""
import pandas as pd
import streamlit as st

from src import config  # noqa: F401  (first: installs the gRPC telemetry stub before chromadb is imported)
from src.core.assets import resolve_asset
from src.query.observability import step_summary
from src.core.topic_registry import list_topics

EXAMPLES = [
    "What was the total on receipt X51005361900?",
    "Which airline has the most negative tweets, and how many?",
    "What colour are the bars in the snake sightings chart in the 2025 wildlife survey?",
    "How many restaurants offer online delivery?",
    "What budget was approved for the inspection planned for 2 April 2025?",
    "What colour is the skirt in product photo 12839, and what does the catalog list?",
]
CHUNK_PREVIEW_CHARS = 1500

st.set_page_config(page_title="Multi-modal RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the search index...")
def load_pipeline():
    from src.query.pipeline import RagPipeline

    return RagPipeline()


def chunk_counts(pipeline) -> dict[str, int]:
    store = getattr(getattr(pipeline, "searcher", None), "store", None)
    if store is None:
        return {}
    try:
        return {t.id: store.count(t.id) for t in list_topics()}
    except Exception:                                   # the sidebar must never stop the chat from working
        return {}


# ------------------------------------------------------------------ rendering
def render_sources(result) -> None:
    if not result.sources:
        if result.refusal_stage:
            st.caption(f"No sources: declined at the **{result.refusal_stage}** step.")
        return
    st.markdown("**Sources**")
    items = {i.number: i for i in (result.context.items if result.context else [])}
    for s in result.sources:
        st.markdown(f"`{''.join(f'[{r}]' for r in s.refs)}` {s.display}")
        if s.image_path:
            path = resolve_asset(s.image_path)
            if path.exists():
                st.image(str(path), width=280)
        for n in s.numbers:
            item = items.get(n)
            if item is not None:
                with st.expander(f"Show the text of chunk [{n}]"):
                    st.text(item.text[:CHUNK_PREVIEW_CHARS])


def render_trace(result) -> None:
    t = result.trace
    if t is None:
        return
    tokens = sum(t.tokens.values())
    with st.expander(f"Trace · {t.total_seconds:.1f}s · {t.llm_calls} model call(s) · {tokens:,} tokens"):
        if result.standalone_question != result.question:
            st.markdown(f"**Searched for:** {result.standalone_question}")
        if result.topics:
            st.markdown("**Topics chosen by the router:** " + ", ".join(f"`{x}`" for x in result.topics))
        st.markdown("**Steps**")
        st.dataframe(pd.DataFrame([{"step": s.name, "seconds": round(s.seconds, 2), "what happened": step_summary(s)}
                                   for s in t.steps]), hide_index=True, width="stretch")
        if result.retrieved:
            st.markdown("**Retrieved chunks** (best first; rank 1 = best in that search)")
            st.dataframe(pd.DataFrame([{
                "chunk": h.label, "topic": h.topic_id, "meaning rank": h.dense_rank, "keyword rank": h.bm25_rank,
                "distance": None if h.dense_distance is None else round(h.dense_distance, 3), "score": round(h.score, 4),
            } for h in result.retrieved]), hide_index=True, width="stretch")
        if result.calculations:
            st.markdown("**Table calculations** (computed over the full tables)")
            for c in result.calculations:
                st.code(c.to_text(), language=None)
        if result.error:
            st.error(result.error)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        result = m.get("result")
        if result is not None:
            render_sources(result)
            if st.session_state.get("show_trace", True):
                render_trace(result)


# ------------------------------------------------------------------ the page
pipeline = load_pipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🔎 Multi-modal RAG")
    st.caption("Ask about text, tables, PDFs, scans and images. Answers use only the documents, and say where they came from.")
    st.toggle("Show the trace under each answer", value=True, key="show_trace")
    if st.button("Clear the conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try one**")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example{i}", width="stretch"):
            st.session_state.pending = example
    st.markdown("**What it knows about**")
    counts = chunk_counts(pipeline)
    for t in list_topics():
        with st.expander(f"{t.name} · {counts[t.id]} chunks" if t.id in counts else t.name):
            st.caption(t.description)

st.header("Ask your documents")
for message in st.session_state.messages:
    render_message(message)

typed = st.chat_input("Ask a question, or a follow-up like \"and in Mombasa?\"")
question = typed or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            result = pipeline.answer(question, history)
    st.session_state.messages.append({"role": "assistant", "content": result.text, "result": result})
    st.rerun()
