"""
RAG query pipeline: embed question -> retrieve chunks from Pinecone -> GPT-4o answer.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 10
GRAPH_TOP_K = 5  # smaller vector context when graph facts are also in the prompt

SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about SEC filings (10-K and 10-Q reports).

Rules:
- Answer ONLY from the provided context chunks. Do not use prior knowledge.
- If a "Knowledge Graph Facts" block is provided, treat it as ground truth for entity/relationship questions (people, competitors, ownership, headquarters, products) — prefer it over inference from prose.
- Always cite which filing section or source the information comes from.
- If the answer is not present in the context, say: "This information is not found in the provided filings."
- Be precise with numbers and dates.
"""

# Keyword heuristic -> graph query types. Deliberately simple; no LLM router.
_GRAPH_INTENTS = {
    "board": ["board of directors", "board member", "director"],
    "executives": ["executive", "officer", "ceo", "cfo", "coo", "chief executive",
                   "chief financial", "chief operating", "leadership team"],
    "competitors": ["competitor", "compete", "competes", "competition", "rival"],
    "shareholders": ["shareholder", "stockholder", "institutional investor",
                     "ownership", "owns", "stake"],
    "headquarters": ["headquarter", "headquarters", "hq", "principal executive office",
                     "address", "located", "where is"],
    "products": ["product", "products", "brand", "brands", "services offered",
                 "what does", "sell"],
}


def _classify_graph_intent(question: str) -> set[str]:
    q = question.lower()
    return {intent for intent, kws in _GRAPH_INTENTS.items() if any(kw in q for kw in kws)}


def _resolve_company(kg, session, question: str) -> str | None:
    """Find which issuer the question is about by matching ticker or legal name
    against the question text."""
    q = question.lower()
    for issuer in kg.query_issuers(session):
        ticker = (issuer.get("ticker") or "").lower()
        legal = (issuer.get("legal_name") or "").lower()
        # "Apple Inc." -> match on the leading word(s) before the suffix
        legal_short = legal.split(" inc")[0].split(" corp")[0].strip()
        if (ticker and ticker in q.split()) or (legal_short and legal_short in q):
            return issuer["ticker"] or issuer["name"]
    return None


def _fetch_graph_facts(intents: set[str], question: str, company: str = None) -> dict:
    """Query Neo4j for each matched intent. Returns {} if Neo4j is unconfigured/down."""
    if not os.getenv("NEO4J_URI"):
        return {}
    from src import knowledge_graph as kg
    try:
        driver = kg.get_driver()
    except Exception as e:
        print(f"(knowledge graph unavailable: {e})")
        return {}
    query_fns = {
        "board": kg.query_board,
        "executives": kg.query_executives,
        "competitors": kg.query_competitors,
        "shareholders": kg.query_shareholders,
        "headquarters": kg.query_headquarters,
        "products": kg.query_products,
    }
    facts = {}
    try:
        with driver.session() as session:
            target = company or _resolve_company(kg, session, question)
            if not target:
                return {}
            for intent in intents:
                rows = query_fns[intent](session, target)
                if rows:
                    facts[intent] = rows
    finally:
        kg.close_driver(driver)
    return facts


def format_graph_context(facts: dict) -> str:
    lines = ["Knowledge Graph Facts (structured, extracted from the filings):"]
    for intent, rows in facts.items():
        lines.append(f"\n{intent.capitalize()}:")
        for r in rows:
            if intent in ("board", "executives"):
                title = f" — {r['title']}" if r.get("title") else ""
                lines.append(f"  - {r['name']}{title} ({r['org']})")
            elif intent == "headquarters":
                lines.append(f"  - {r['org']}: {r['address']}")
            elif intent == "shareholders":
                pct = f" ({r['pct_owned']}% owned)" if r.get("pct_owned") is not None else ""
                lines.append(f"  - {r['name']}{pct} -> {r['org']}")
            else:
                lines.append(f"  - {r['name']} ({r['org']})")
    return "\n".join(lines)


def embed_query(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return response.data[0].embedding


def retrieve(index, query_embedding: list[float], top_k: int = TOP_K, filter: dict = None) -> list[dict]:
    kwargs = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
    if filter:
        kwargs["filter"] = filter
    results = index.query(**kwargs)
    return results.matches


def format_context(matches: list) -> str:
    parts = []
    for m in matches:
        meta = m.metadata
        header = f"[{meta.get('filing_type', 'Filing')} | {meta.get('company', '')} | chunk {meta.get('chunk_index', '')}]"
        parts.append(f"{header}\n{meta.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def answer(question: str, company: str = None, filing_type: str = None, use_graph: bool = None) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "10k-filings"))

    # Graph routing: use_graph=None -> keyword heuristic; True/False -> forced
    graph_facts = {}
    if use_graph is not False:
        intents = _classify_graph_intent(question) if use_graph is None else set(_GRAPH_INTENTS)
        if intents:
            graph_facts = _fetch_graph_facts(intents, question, company=company)

    filter_dict = {}
    if company:
        filter_dict["company"] = {"$eq": company}
    if filing_type:
        filter_dict["filing_type"] = {"$eq": filing_type}

    query_embedding = embed_query(client, question)
    top_k = GRAPH_TOP_K if graph_facts else TOP_K
    matches = retrieve(index, query_embedding, top_k=top_k, filter=filter_dict or None)

    if not matches and not graph_facts:
        return {"answer": "No relevant documents found in index.", "sources": [], "graph_facts": {}}

    context_parts = []
    if graph_facts:
        context_parts.append(format_graph_context(graph_facts))
    if matches:
        context_parts.append(format_context(matches))
    context = "\n\n===\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from SEC filings:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    sources = [
        {
            "source": m.metadata.get("source", ""),
            "filing_type": m.metadata.get("filing_type", ""),
            "chunk_index": m.metadata.get("chunk_index"),
            "score": round(m.score, 4),
        }
        for m in matches
    ]

    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "graph_facts": graph_facts,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--company", default=None)
    parser.add_argument("--filing-type", default=None, choices=["10-K", "10-Q"])
    parser.add_argument("--use-graph", default="auto", choices=["auto", "on", "off"])
    args = parser.parse_args()

    use_graph = {"auto": None, "on": True, "off": False}[args.use_graph]
    result = answer(args.question, company=args.company, filing_type=args.filing_type, use_graph=use_graph)
    print("\n=== ANSWER ===")
    print(result["answer"])
    if result.get("graph_facts"):
        print("\n=== GRAPH FACTS ===")
        print(format_graph_context(result["graph_facts"]))
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  [{s['filing_type']}] chunk {s['chunk_index']} (score={s['score']}) — {s['source']}")
