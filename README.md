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

**2. Neo4j** — either works:
- **Local**: install [Neo4j Desktop](https://neo4j.com/download/), create and start a database. URI is `bolt://localhost:7687`, user `neo4j`.
- **Cloud**: create a free [AuraDB](https://console.neo4j.io) instance. URI is `neo4j+s://<id>.databases.neo4j.io`. **Careful:** newer Aura instances use the *instance ID as the username*, not `neo4j` — copy `NEO4J_USERNAME` from the credentials file Aura downloads at creation, and put it under `NEO4J_USER` (the name this codebase reads). Resetting the password invalidates the creation-file password.

**3. Keys and credentials**
```bash
cp .env.example .env
```
Fill in `.env`:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=10k-filings
NEO4J_URI=bolt://localhost:7687      # or neo4j+s://<id>.databases.neo4j.io
NEO4J_USER=neo4j                     # Aura: the instance ID, not neo4j
NEO4J_PASSWORD=<password>
```

---

## Usage

**Easiest: everything from the UI**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. From the sidebar you can:
- **Add company** — type ticker(s) (e.g. `NVDA` or `AAPL, MSFT`), pick filing types and count under "Ingest options", hit **Ingest**. Runs the full pipeline (fetch → preprocess → index → extract entities → build graph) with per-stage progress; the new company appears in the dropdown when done. Graph steps are skipped with a warning if `NEO4J_URI` is unset. Ingestion takes a few minutes per 10-K (EDGAR download + LLM extraction).
- **Company dropdown** — scope questions to one indexed company ("All companies" = no filter).
- **Knowledge graph toggle** — Auto/On/Off; graph-backed answers show a "Knowledge Graph Facts" expander.

**Or from the CLI:**
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
Graph-routed answers print a `=== GRAPH FACTS ===` block. Force behavior with `--use-graph {auto,on,off}` (default `auto`); scope to one company with `--company AAPL`.

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

## Deployment (Streamlit Community Cloud)

The app runs on Streamlit Community Cloud, wired to this repo: **every push to `main` auto-redeploys** (~1–2 min; check "Manage app" in the running app for build logs and the active commit).

How the pieces split:
- **Repo** — code + the pre-indexed demo dataset (`data/processed/chunks.json` is committed for this reason).
- **Dashboard Secrets (TOML)** — all credentials; `app.py` bridges `st.secrets` into `os.environ` at startup so the pipeline modules work unchanged. Set `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `APP_PASSWORD`.
- **Pinecone + AuraDB** — persistent stores; the container connects at runtime.
- **Container disk is ephemeral** — tickers ingested through the UI persist in Pinecone/Aura, but the updated `chunks.json` is lost on restart. The company dropdown falls back to graph issuer nodes when the file is stale.

Access: set `APP_PASSWORD` in secrets to gate the whole UI (skip it locally for no gate), and set the app's sharing to *public* in the dashboard — the password is the barrier, not Streamlit's viewer login. Deploy settings: repo `main`, entrypoint `app.py`, Python 3.11+.

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
├── app.py                    # Streamlit UI (company picker, in-app ingest, graph toggle)
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
