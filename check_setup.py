"""Verifies the environment: folders, API key, LLM call, embedding call, Chroma."""
import sys

from src import config


def step(name, fn):
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False
    return True


def main():
    config.ensure_dirs()
    if not config.GEMINI_API_KEY or "paste_your_key" in config.GEMINI_API_KEY:
        print("[FAIL] GEMINI_API_KEY missing in .env")
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    def llm():
        r = client.models.generate_content(
            model=config.LLM_MODEL, contents="Reply with the single word: ready"
        )
        return r.text.strip()

    def embed():
        r = client.models.embed_content(model=config.EMBED_MODEL, contents="hello")
        return f"vector of {len(r.embeddings[0].values)} dims"

    def chroma():
        import chromadb

        c = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return f"chromadb {chromadb.__version__}, {len(c.list_collections())} collections"

    def libs():
        import pymupdf, pandas, rank_bm25, langchain_text_splitters  # noqa: F401

        return f"pymupdf {pymupdf.__version__}, pandas {pandas.__version__}"

    results = [step("LLM", llm), step("Embeddings", embed), step("ChromaDB", chroma), step("Libraries", libs)]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
