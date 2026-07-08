"""
End-to-end runner: fetch -> preprocess -> index -> extract-entities -> build-graph -> query.
Can run individual stages or the full pipeline.
"""
import argparse
import os
import sys
from pathlib import Path


def cmd_fetch(args):
    from src.fetch_filings import fetch
    for ticker in args.ticker:
        fetch(
            ticker=ticker,
            filing_types=args.types,
            num_filings=args.limit,
            out_dir=Path(args.raw_dir),
        )


def cmd_preprocess(args):
    from src.preprocess import process_companies
    process_companies(
        raw_dir=Path(args.raw_dir),
        companies=args.ticker,
        out_path=Path(args.chunks),
    )


def cmd_index(args):
    from src.index_documents import upsert_chunks
    upsert_chunks(Path(args.chunks), args.index)


def cmd_extract_entities(args):
    from src.extract_entities import extract_entities
    extract_entities(Path(args.chunks), Path(args.entities), companies=args.ticker)


def cmd_build_graph(args):
    from src.knowledge_graph import build_graph
    build_graph(Path(args.entities), clear=args.clear)


def cmd_query(args):
    from src.rag_query import answer, format_graph_context
    use_graph = {"auto": None, "on": True, "off": False}[args.use_graph]
    result = answer(
        args.question,
        company=args.company,
        filing_type=args.filing_type,
        use_graph=use_graph,
    )
    print("\n=== ANSWER ===")
    print(result["answer"])
    if result.get("graph_facts"):
        print("\n=== GRAPH FACTS ===")
        print(format_graph_context(result["graph_facts"]))
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  [{s['filing_type']}] chunk {s['chunk_index']} (score={s['score']}) — {s['source']}")


def cmd_all(args):
    cmd_fetch(args)
    cmd_preprocess(args)
    cmd_index(args)
    if not os.getenv("NEO4J_URI"):
        print("\nWARNING: NEO4J_URI not set — skipping extract-entities and build-graph.")
        print("Set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env and run:")
        print("  python main.py extract-entities && python main.py build-graph")
        return
    cmd_extract_entities(args)
    cmd_build_graph(args)


DEFAULTS = {
    "ticker": ["AAPL"],
    "raw_dir": "data/raw",
    "chunks": "data/processed/chunks.json",
    "entities": "data/processed/entities.json",
    "index": "10k-filings",
    "types": ["10-K", "10-Q"],
    "limit": 3,
}


def main():
    parser = argparse.ArgumentParser(description="10-K/10-Q RAG pipeline")
    sub = parser.add_subparsers(dest="cmd")

    def add_common(p):
        p.add_argument("--ticker", nargs="+", default=None, help="Stock ticker(s)")
        p.add_argument("--raw-dir", default=None)
        p.add_argument("--chunks", default=None)
        p.add_argument("--entities", default=None)
        p.add_argument("--index", default=None)
        p.add_argument("--types", nargs="+", default=None)
        p.add_argument("--limit", type=int, default=None)

    for name, help_text in [
        ("fetch", "Download SEC filings"),
        ("preprocess", "Parse and chunk filings"),
        ("index", "Embed and upsert to Pinecone"),
        ("extract-entities", "LLM entity/relationship extraction -> entities.json"),
        ("all", "Run fetch + preprocess + index + extract-entities + build-graph"),
    ]:
        p = sub.add_parser(name, help=help_text)
        add_common(p)

    g_parser = sub.add_parser("build-graph", help="Load entities.json into Neo4j")
    add_common(g_parser)
    g_parser.add_argument("--clear", action="store_true", help="Wipe graph before loading")

    q_parser = sub.add_parser("query", help="Ask a question")
    q_parser.add_argument("--question", required=True)
    q_parser.add_argument("--company", default=None)
    q_parser.add_argument("--filing-type", default=None, choices=["10-K", "10-Q"])
    q_parser.add_argument("--use-graph", default="auto", choices=["auto", "on", "off"])

    args = parser.parse_args()

    for key, value in DEFAULTS.items():
        if getattr(args, key, None) in (None, []):
            setattr(args, key, value)
    if not hasattr(args, "clear"):
        args.clear = False

    dispatch = {
        "fetch": cmd_fetch,
        "preprocess": cmd_preprocess,
        "index": cmd_index,
        "extract-entities": cmd_extract_entities,
        "build-graph": cmd_build_graph,
        "query": cmd_query,
        "all": cmd_all,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
