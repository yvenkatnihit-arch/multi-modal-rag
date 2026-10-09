"""Terminal chat with the multi-modal RAG system.

    python ask.py                       interactive chat (remembers the conversation)
    python ask.py "your question"       answer one question and exit
    python ask.py --trace               also print the per-step trace after every answer

Commands inside the chat:  /topics  /trace  /clear  /quit
"""
import argparse
import sys

from src import config  # noqa: F401  (first: installs the gRPC telemetry stub before chromadb is imported)
from src.query.observability import format_trace

HELP = "Commands: /topics (what I know about)  /trace (show or hide the trace)  /clear (forget the conversation)  /quit"


def format_result(result, show_trace: bool = False) -> str:
    lines = [result.text]
    if result.sources:
        lines += ["", "Sources:"]
        lines += ["  " + "".join(f"[{r}]" for r in s.refs) + " " + s.display for s in result.sources]
    if show_trace and result.trace:
        lines += ["", format_trace(result.trace)]
    return "\n".join(lines)


def chat(pipeline, read=input, write=print, show_trace: bool = False, topics=None) -> list[dict]:
    """The interactive loop. Returns the final history (handy for tests)."""
    history: list[dict] = []
    write("Multi-modal RAG. " + HELP)
    while True:
        try:
            line = read("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "/q"):
            break
        if line.lower() == "/clear":
            history.clear()
            write("(conversation cleared)")
            continue
        if line.lower() == "/trace":
            show_trace = not show_trace
            write(f"(trace {'on' if show_trace else 'off'})")
            continue
        if line.lower() == "/topics":
            for t in (topics or []):
                write(f"  {t.id:<18} {t.description[:100]}...")
            continue
        if line.startswith("/"):
            write(HELP)
            continue

        result = pipeline.answer(line, history)
        write("\n" + format_result(result, show_trace))
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": result.text}]
    return history


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):                    # Windows consoles are not UTF-8 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question", nargs="?", help="ask one question and exit")
    ap.add_argument("--trace", action="store_true", help="show the per-step trace")
    args = ap.parse_args(argv)

    from src.query.pipeline import RagPipeline
    from src.core.topic_registry import list_topics

    pipeline = RagPipeline()
    if args.question:
        print(format_result(pipeline.answer(args.question), args.trace))
        return 0
    chat(pipeline, show_trace=args.trace, topics=list_topics())
    return 0


if __name__ == "__main__":
    sys.exit(main())
