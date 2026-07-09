import json
import os
import re
from pathlib import Path

import streamlit as st

from src.rag_query import answer

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
ENTITIES_PATH = Path("data/processed/entities.json")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "10k-filings")

st.title("10-K / 10-Q Analyst")


@st.cache_data
def list_companies() -> list[str]:
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    return sorted({c["company"] for c in chunks})


def ingest(tickers: list[str], filing_types: list[str], limit: int) -> None:
    from src.fetch_filings import fetch
    from src.index_documents import upsert_chunks
    from src.preprocess import process_companies

    with st.status(f"Ingesting {', '.join(tickers)}…", expanded=True) as status:
        for ticker in tickers:
            st.write(f"Fetching {ticker} from SEC EDGAR…")
            fetch(ticker=ticker, filing_types=filing_types, num_filings=limit, out_dir=RAW_DIR)

        st.write("Preprocessing filings into chunks…")
        process_companies(raw_dir=RAW_DIR, companies=tickers, out_path=CHUNKS_PATH)

        st.write("Embedding and indexing in Pinecone…")
        upsert_chunks(CHUNKS_PATH, INDEX_NAME)

        if os.getenv("NEO4J_URI"):
            from src.extract_entities import extract_entities
            from src.knowledge_graph import build_graph

            st.write("Extracting entities (LLM)…")
            extract_entities(CHUNKS_PATH, ENTITIES_PATH, companies=tickers)
            st.write("Building knowledge graph…")
            build_graph(ENTITIES_PATH)
        else:
            st.warning("NEO4J_URI not set — skipped entity extraction and graph build.")

        status.update(label=f"Ingested {', '.join(tickers)}", state="complete")


with st.sidebar:
    companies = list_companies()
    company_choice = st.selectbox("Company", ["All companies"] + companies)
    graph_mode = st.selectbox("Knowledge graph", ["Auto", "On", "Off"])

    st.divider()
    st.markdown("**Add company**")
    ticker_input = st.text_input("Ticker(s)", placeholder="NVDA or AAPL, MSFT")
    with st.expander("Ingest options"):
        filing_types = st.multiselect("Filing types", ["10-K", "10-Q"], default=["10-K"])
        limit = st.number_input("Filings per type", min_value=1, max_value=10, value=1)
    if st.button("Ingest") and ticker_input:
        tickers = [t.upper() for t in re.split(r"[,\s]+", ticker_input.strip()) if t]
        try:
            ingest(tickers, filing_types or ["10-K"], int(limit))
        except Exception as e:
            st.error(f"Ingest failed: {e}")
        else:
            st.cache_data.clear()
            st.rerun()

company = None if company_choice == "All companies" else company_choice
use_graph = {"Auto": None, "On": True, "Off": False}[graph_mode]

question = st.text_input("Question", placeholder="What are Apple's main risk factors?")

if st.button("Ask") and question:
    with st.spinner("Searching filings..."):
        result = answer(question, company=company, use_graph=use_graph)

    st.markdown("### Answer")
    st.write(result["answer"])

    if result.get("graph_facts"):
        with st.expander("Knowledge Graph Facts"):
            for intent, rows in result["graph_facts"].items():
                st.markdown(f"**{intent.capitalize()}**")
                for r in rows:
                    if intent in ("board", "executives"):
                        title = f" — {r['title']}" if r.get("title") else ""
                        st.write(f"- {r['name']}{title} ({r['org']})")
                    elif intent == "headquarters":
                        st.write(f"- {r['org']}: {r['address']}")
                    else:
                        st.write(f"- {r['name']} ({r['org']})")

    with st.expander("Sources"):
        for s in result["sources"]:
            st.write(f"**{s['filing_type']}** · chunk {s['chunk_index']} · score {s['score']}")
            st.caption(s["source"])
