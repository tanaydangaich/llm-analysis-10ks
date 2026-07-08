# LLM Analysis of 10-K Filings — Hybrid RAG + Knowledge Graph

Q&A system for SEC filings combining vector retrieval (RAG) with a Neo4j knowledge graph. Ask natural-language questions about a public company's 10-K/10-Q filings and get answers grounded in the actual documents — with structured graph facts for entity questions (board members, executives, headquarters, products) that plain top-k vector retrieval handles poorly.

Based on: *LLM Analysis of 10-K and 10-Q Filings: RAG Results* (IJRTI, 2024). This repo implements both halves of the paper: the vector-RAG pipeline and the knowledge-graph layer.

---

## Why a knowledge graph on top of RAG?

Vector retrieval works well for narrative questions ("What are Apple's risk factors?") but is unreliable for precise entity facts. A question like "Who is on Apple's board?" depends on the signature-page chunk happening to rank in the top-k — and when it doesn't, the LLM hallucinates or mixes filing years.

The graph layer extracts entities and relationships once at index time, stores them as structured facts in Neo4j, and injects them into the prompt as ground truth when a question matches an entity intent. Answers become exact and reproducible:

| Question | Vector-only | Hybrid (graph) |
|---|---|---|
| "Who is on Apple's board?" | 10 names mixing 3 filing years | Exact 7 directors + titles from the filing |
| "Where is Apple headquartered?" | Depends on retrieval luck | "One Apple Park Way, Cupertino, California 95014" |
| "Who are Apple's competitors?" | Narrative summary | No fabricated list (modern 10-Ks name no competitors — verified) |

---

## Architecture

```
fetch (multi-ticker) → preprocess → chunks.json
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
              index_documents.py                  extract_entities.py
              (Pinecone vectors)                  (gpt-4o-mini structured
                    │                              extraction → entities.json)
                    │                                       │
                    │                                       ▼
                    │                              knowledge_graph.py
                    │                              (Neo4j MERGE upserts)
                    ▼                                       ▼
              rag_query.answer() ──── hybrid routing ──── Neo4j graph
              (keyword heuristic: vector-only, or
               graph facts + vector context blended
               into one GPT-4o prompt)
```

**Graph schema** — nodes: `Person`, `Organization` (ticker, legal_name, role, headquarters_address), `Product`, `Filing`. Relationships: `BOARD_MEMBER_OF`, `EXECUTIVE_OF` (title), `COMPETES_WITH`, `SHAREHOLDER_OF` (pct_owned), `PRODUCT_OF`, `SUBSIDIARY_OF`, `PARTNERS_WITH`, `MENTIONED_IN` (provenance).

---

## Stack

- **Answer LLM** — GPT-4o · **Extraction LLM** — gpt-4o-mini (structured output / forced tool calls)
- **Embeddings** — OpenAI `text-embedding-3-small`
- **Vector DB** — Pinecone · **Graph DB** — Neo4j (Desktop, local)
- **Filing source** — SEC EDGAR via `sec-edgar-downloader`
- **UI** — Streamlit

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Neo4j** — install [Neo4j Desktop](https://neo4j.com/download/), create and start a local database. Default URI is `bolt://localhost:7687`.

**3. Keys and credentials**
```bash
cp .env.example .env
```
Fill in `.env`:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=10k-filings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password from Neo4j Desktop>
```

---

## Usage

**Full pipeline, multiple companies:**
```bash
python main.py fetch --ticker AAPL MSFT GOOGL --types 10-K --limit 1
python main.py preprocess --ticker AAPL MSFT GOOGL
python main.py index
python main.py extract-entities --ticker AAPL MSFT GOOGL
python main.py build-graph
```
Or all at once: `python main.py all --ticker AAPL MSFT GOOGL --types 10-K --limit 1` (graph steps skip gracefully if `NEO4J_URI` is unset).

**Query from the terminal:**
```bash
python main.py query --question "Who is on Apple's board of directors?"
python main.py query --question "Who is Microsoft's CFO?"
python main.py query --question "What are Apple's main risk factors?"   # pure vector — no graph facts
```
Graph-routed answers print a `=== GRAPH FACTS ===` block. Force behavior with `--use-graph {auto,on,off}` (default `auto`).

**Launch the UI:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Sidebar has an Auto/On/Off knowledge-graph toggle; graph-backed answers show a "Knowledge Graph Facts" expander.

**Inspect the graph** in Neo4j Browser:
```cypher
MATCH (o:Organization {ticker:'AAPL'})-[r]-(n) WHERE NOT n:Filing RETURN o, r, n LIMIT 50
MATCH (p:Person)-[r]->(o:Organization) RETURN p, r, o
```

---

## CLI reference

| Command | What it does |
|---------|-------------|
| `fetch` | Download filings from SEC EDGAR |
| `preprocess` | Parse HTML, clean, chunk into 500-token pieces |
| `index` | Embed chunks and upload to Pinecone |
| `extract-entities` | LLM entity/relationship extraction → `entities.json` |
| `build-graph` | Load `entities.json` into Neo4j (idempotent; `--clear` wipes first) |
| `query` | Ask a question (`--use-graph auto\|on\|off`) |
| `all` | fetch + preprocess + index + extract-entities + build-graph |

**Common flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `AAPL` | One or more stock tickers |
| `--types` | `10-K 10-Q` | Filing types to fetch |
| `--limit` | `3` | Filings per type |
| `--entities` | `data/processed/entities.json` | Extraction output path |

---

## Design notes

- **Extraction windows** — 500-token chunks are too small for reliable entity extraction (a director list can straddle a boundary), so chunks are regrouped into ~3000-token windows per filing. A low-signal heuristic skips inline-XBRL/cover-page tag soup before spending LLM calls.
- **No embeddings in Neo4j** — Pinecone already does vector search; the graph does the one thing Pinecone can't: structured traversal.
- **Idempotent loads** — all graph writes are Cypher `MERGE`, so re-running `build-graph` never duplicates nodes.
- **Keyword intent router** — a simple keyword heuristic (no LLM router) decides when to query the graph; `--use-graph` overrides it.
- **Known finding** — `COMPETES_WITH` edges are sparse-to-empty because modern 10-Ks deliberately avoid naming competitors (verified directly against MSFT's FY2025 filing). Competitor questions fall back to narrative vector context without fabricating a list.
- **Accepted gaps** — light name canonicalization only (no full coreference resolution: "Tim Cook" vs "Timothy D. Cook" can coexist).

---

## Project structure

```
├── app.py                    # Streamlit UI (graph toggle + facts expander)
├── main.py                   # CLI entrypoint
├── PLAN.md                   # Implementation plan for the graph layer
├── src/
│   ├── fetch_filings.py      # Download from SEC EDGAR
│   ├── preprocess.py         # Parse, clean, chunk (multi-company)
│   ├── index_documents.py    # Embed + upsert to Pinecone
│   ├── extract_entities.py   # LLM entity/relationship extraction
│   ├── knowledge_graph.py    # Neo4j upserts + read helpers
│   └── rag_query.py          # Hybrid graph+vector query pipeline
├── data/
│   ├── raw/                  # Downloaded SEC filings (gitignored)
│   └── processed/            # chunks.json (gitignored), entities.json
├── demo/                     # Demo video
├── requirements.txt
└── .env.example
```
