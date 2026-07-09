import json
import os
import re
from pathlib import Path

import streamlit as st

# Streamlit Cloud provides credentials via st.secrets; the pipeline modules read
# os.environ. Bridge them before any src import triggers an env lookup.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets file — local dev uses .env via dotenv

from src.rag_query import answer

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
ENTITIES_PATH = Path("data/processed/entities.json")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "10k-filings")

st.set_page_config(page_title="10-K / 10-Q Analyst", page_icon="§", layout="wide")

# ---------------------------------------------------------------------------
# Ledger/terminal styling — a filing sits somewhere between a legal document
# and a data feed, so the UI borrows a masthead + monospace-figures register
# rather than a generic chat skin.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #0D1013;
        --panel: #161A1F;
        --line: #2A2F37;
        --text: #E8E6DE;
        --muted: #8B9099;
        --gold: #D4A24C;
        --good: #5FA776;
        --bad: #C1584F;
    }

    html, body, [class*="css"] { color: var(--text); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

    h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif !important; letter-spacing: 0.01em; }

    /* Masthead */
    .masthead { border-bottom: 1px solid var(--line); padding-bottom: 0.9rem; margin-bottom: 1.4rem; }
    .masthead h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .masthead .strip {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--gold);
    }

    /* Section labels */
    .section-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.2rem 0 0.5rem 0;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.35rem;
    }
    .section-label .mark { color: var(--gold); }

    /* Query prompt */
    div[data-testid="stTextInput"] input {
        font-family: "IBM Plex Mono", monospace !important;
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 2px !important;
    }
    div[data-testid="stTextInput"] input:focus { border-color: var(--gold) !important; }

    button[kind], .stButton button {
        font-family: "Inter", sans-serif !important;
        border-radius: 2px !important;
        border: 1px solid var(--gold) !important;
    }

    /* Memo card for each Q&A turn */
    .memo { border: 1px solid var(--line); border-radius: 2px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; background: var(--panel); }
    .memo-q {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.85rem;
        color: var(--gold);
        margin-bottom: 0.7rem;
    }
    .memo-q .tag { color: var(--muted); margin-right: 0.5rem; }
    .memo-a { font-family: "Source Serif 4", Georgia, serif; font-size: 1.02rem; line-height: 1.55; color: var(--text); }

    /* Ledger-style rows for sources */
    .ledger-row {
        display: flex; justify-content: space-between; gap: 1rem;
        font-family: "IBM Plex Mono", monospace; font-size: 0.8rem;
        padding: 0.4rem 0; border-bottom: 1px solid var(--line);
    }
    .ledger-row:last-child { border-bottom: none; }
    .ledger-row .score { color: var(--gold); white-space: nowrap; }
    .ledger-row .path { color: var(--muted); }

    hr { border-color: var(--line) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if os.getenv("APP_PASSWORD") and not st.session_state.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw:
        if pw == os.environ["APP_PASSWORD"]:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


@st.cache_data
def list_companies() -> list[str]:
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        return sorted({c["company"] for c in chunks})
    # Cloud disk is ephemeral: chunks.json can vanish on restart while the graph
    # persists — fall back to issuer nodes so the dropdown stays usable.
    if os.getenv("NEO4J_URI"):
        from src import knowledge_graph as kg
        try:
            driver = kg.get_driver()
            try:
                with driver.session() as session:
                    issuers = kg.query_issuers(session)
            finally:
                kg.close_driver(driver)
            return sorted({i["ticker"] or i["name"] for i in issuers})
        except Exception:
            pass
    return []


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


if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.markdown('<div class="section-label"><span class="mark">§</span> Filters</div>', unsafe_allow_html=True)
    companies = list_companies()
    if not companies:
        st.info("No companies yet. Add a ticker below to get started.")
    company_choice = st.selectbox("Company", ["All companies"] + companies, label_visibility="collapsed")
    graph_mode = st.selectbox("Knowledge graph", ["Auto", "On", "Off"])

    if st.session_state["history"] and st.button("Clear conversation"):
        st.session_state["history"] = []
        st.rerun()

    st.markdown('<div class="section-label"><span class="mark">§</span> Add company</div>', unsafe_allow_html=True)
    ticker_input = st.text_input("Ticker(s)", placeholder="NVDA or AAPL, MSFT", label_visibility="collapsed")
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

st.markdown(
    f"""
    <div class="masthead">
        <h1>10-K / 10-Q Analyst</h1>
        <div class="strip">{company or "ALL FILINGS"} — GRAPH: {graph_mode.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_NODE_COLOR = {"Organization": "#D4A24C", "Person": "#5FA776", "Product": "#8B9099", "RiskFactor": "#C1584F"}


_INTENT_REL = {
    "board": ("Person", "board member of", False),
    "executives": ("Person", "executive of", False),
    "competitors": ("Organization", "competes with", True),
    "shareholders": ("Organization", "shareholder of", False),
    "products": ("Product", "product of", False),
    "risk_factors": ("RiskFactor", "exposed to", False),
}


def _render_cross_company_risk_viz(rows: list) -> None:
    """Category-centric star: one RiskFactor node, an org leaf per exposed company.
    Different shape from the org-centric views below — there's no single company here."""
    from streamlit_agraph import Config, Edge, Node, agraph

    if not rows:
        return
    category = rows[0]["category"]
    nodes = {category: Node(id=category, label=category, size=22, color=_NODE_COLOR["RiskFactor"], font={"color": "#E8E6DE"})}
    edges = []
    for r in rows:
        org = r["org"]
        if org not in nodes:
            nodes[org] = Node(id=org, label=org, size=14, color=_NODE_COLOR["Organization"], font={"color": "#E8E6DE"})
        nodes[org].title = r.get("summary")
        edges.append(Edge(source=org, target=category, title="exposed to"))

    config = Config(height=420, width=860, directed=True, physics=True, hierarchical=False)
    config.physics["barnesHut"] = {"springLength": 170, "springConstant": 0.02, "damping": 0.5, "avoidOverlap": 1}
    with st.expander("§ Knowledge graph visual", expanded=True):
        agraph(nodes=list(nodes.values()), edges=edges, config=config)


def render_graph_viz(graph_facts: dict) -> None:
    """Builds the visual straight from the already-fetched, intent-scoped graph_facts
    (not a fresh full-neighborhood query) so it only ever shows what was actually asked."""
    from streamlit_agraph import Config, Edge, Node, agraph

    if "risk_factors_cross_company" in graph_facts:
        _render_cross_company_risk_viz(graph_facts["risk_factors_cross_company"])
        return

    nodes = {}
    seen_edges = set()
    edges = []
    hq_notes = {}

    org_counts: dict[str, int] = {}
    for rows in graph_facts.values():
        for r in rows:
            if r.get("org"):
                org_counts[r["org"]] = org_counts.get(r["org"], 0) + 1
    if not org_counts:
        return
    center = max(org_counts, key=org_counts.get)

    def add_node(name: str, label: str, size: int) -> None:
        if name not in nodes:
            nodes[name] = Node(
                id=name, label=name, size=size, color=_NODE_COLOR.get(label, "#8B9099"),
                font={"color": "#E8E6DE"},
            )
        elif size > nodes[name].size:
            nodes[name].size = size

    def add_edge(a: str, b: str, rel: str) -> None:
        key = (a, b, rel)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append(Edge(source=a, target=b, title=rel))  # title = hover tooltip, not a canvas label

    for intent, rows in graph_facts.items():
        if intent == "headquarters":
            for r in rows:
                hq_notes[r["org"]] = r.get("address")
                add_node(r["org"], "Organization", 22 if r["org"] == center else 14)
            continue
        spec = _INTENT_REL.get(intent)
        if not spec:
            continue
        neighbor_label, rel, org_to_org = spec
        for r in rows:
            org = r.get("org")
            org_size = 22 if org == center else 14
            add_node(org, "Organization", org_size)
            add_node(r["name"], neighbor_label, 14)
            if org_to_org:
                add_edge(org, r["name"], rel)
            else:
                add_edge(r["name"], org, rel)

    if not nodes:
        return
    for org, addr in hq_notes.items():
        if org in nodes:
            nodes[org].title = f"HQ: {addr}"

    config = Config(height=480, width=860, directed=True, physics=True, hierarchical=False)
    config.physics["barnesHut"] = {"springLength": 170, "springConstant": 0.02, "damping": 0.5, "avoidOverlap": 1}

    with st.expander("§ Knowledge graph visual", expanded=True):
        agraph(nodes=list(nodes.values()), edges=edges, config=config)


_INTENT_TITLES = {"risk_factors_cross_company": "Companies exposed to this risk"}


def render_graph_facts(graph_facts: dict) -> None:
    with st.expander("§ Knowledge graph facts"):
        for intent, rows in graph_facts.items():
            st.markdown(f"**{_INTENT_TITLES.get(intent, intent.replace('_', ' ').capitalize())}**")
            for r in rows:
                if intent in ("board", "executives"):
                    title = f" — {r['title']}" if r.get("title") else ""
                    st.write(f"- {r['name']}{title} ({r['org']})")
                elif intent == "headquarters":
                    st.write(f"- {r['org']}: {r['address']}")
                elif intent == "risk_factors":
                    st.write(f"- **{r['name']}**: {r['summary']} ({r['org']})")
                elif intent == "risk_factors_cross_company":
                    st.write(f"- {r['org']}: {r['summary']}")
                else:
                    st.write(f"- {r['name']} ({r['org']})")


def render_sources(sources: list) -> None:
    with st.expander(f"§ Sources ({len(sources)})"):
        rows = "".join(
            f'<div class="ledger-row"><span>{s["filing_type"]} · chunk {s["chunk_index"]} '
            f'<span class="path">{s["source"]}</span></span><span class="score">{s["score"]:.2f}</span></div>'
            for s in sorted(sources, key=lambda x: x["score"], reverse=True)
        )
        st.markdown(rows, unsafe_allow_html=True)


st.markdown('<div class="section-label"><span class="mark">§</span> Query the filings</div>', unsafe_allow_html=True)
with st.form("query_form", clear_on_submit=True):
    qcol, bcol = st.columns([6, 1])
    with qcol:
        question = st.text_input(
            "Question", placeholder="What are Apple's main risk factors?", label_visibility="collapsed"
        )
    with bcol:
        ask_clicked = st.form_submit_button("Run query", use_container_width=True)

if ask_clicked and question:
    if not companies:
        st.warning("Add a company in the sidebar before asking a question.")
    else:
        with st.spinner("Searching filings…"):
            result = answer(question, company=company, use_graph=use_graph)

        st.session_state["history"].append(
            {
                "question": question,
                "answer": result["answer"],
                "graph_facts": result.get("graph_facts"),
                "sources": result.get("sources"),
            }
        )

history = st.session_state["history"]
for i, turn in reversed(list(enumerate(history, start=1))):
    st.markdown(
        f"""
        <div class="memo">
            <div class="memo-q"><span class="tag">Q{i}</span>{turn['question']}</div>
            <div class="memo-a">{turn['answer']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if turn.get("graph_facts"):
        render_graph_facts(turn["graph_facts"])
        render_graph_viz(turn["graph_facts"])
    if turn.get("sources"):
        render_sources(turn["sources"])
