# Knowledge Graph Layer for the 10-K/10-K RAG Pipeline

## Context

The repo currently implements only half of the paper it's based on (IJRTI2401091,
"LLM Analysis of 10-K and 10-Q Filings: RAG Results"): the vector-RAG pipeline
(fetch → preprocess → index → query via Pinecone + GPT-4o). The paper's stated
"main part of this research" — a **Neo4j knowledge graph** layered on top, used to
answer entity/relationship questions (board members, competitors, HQ, ownership)
that plain top-k vector retrieval handles poorly — is entirely unimplemented
(confirmed via grep: zero references to Neo4j/entities/relationships anywhere in
`src/`, `main.py`, `app.py`, `requirements.txt`).

Goal of this plan: close that gap — add entity/relationship extraction, a Neo4j
graph, hybrid graph+vector query routing, and multi-company data so the graph has
real cross-company edges to traverse — while reusing the existing pipeline
structure rather than rearchitecting it.

**Decisions locked in with the user:**
- Neo4j hosting: **Neo4j Desktop** (local, GUI browser for inspecting the graph — matches the paper's own screenshots). Default connection `bolt://localhost:7687`, user `neo4j`, password set during Desktop DB creation.
- Multi-company scope: add **AAPL + MSFT + GOOGL** (10-K only) so `COMPETES_WITH`-style edges have something real to connect — both are named/implicated in Apple's own antitrust/competitive risk-factor language.

**Grounding facts from re-reading the actual codebase and data (not just the paper):**
- The paper's own Neo4j demo (PDF pg. 599) is crude — it dumps raw chunk text into a generic `Other` node with an `info` property. That's redundant with Pinecone and adds no traversal value. This plan does **not** replicate that; it builds the proper entity-relationship graph shown in the paper's pg. 601 figure (AAPL at center, People/Orgs/Products as colored nodes) instead.
- The paper also attaches embeddings to Neo4j nodes (pg. 600) — skipped here too. Pinecone already does vector search; duplicating embeddings into Neo4j is sync overhead for no new capability. Neo4j is used purely for structured graph traversal, the one thing Pinecone can't do.
- `data/processed/chunks.json` (already generated, 327 AAPL chunks) was inspected directly: chunk 12 contains the exact HQ address ("One Apple Park Way, Cupertino, California 95014") the paper says GPT-4-alone hallucinates; chunk 108 contains the full signature page with Tim Cook, Kevan Parekh, Chris Kondo, and 6 named directors with titles. **This data is already fetched** — the gap is that vector retrieval doesn't reliably surface it, not that it's missing. Making it a structured graph fact removes reliance on retrieval luck.
- Chunks 0–~15 of every filing are inline-XBRL/cover-page tag soup, not prose — must be filtered before spending LLM calls on entity extraction.
- No `docker` CLI on this machine, no `neo4j` python driver installed yet, no `.env.example` despite README referencing it.

---

## Architecture / data flow

```
fetch (multi-ticker) → preprocess (multi-ticker) → chunks.json
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                                       ▼
                              index_documents.py                  extract_entities.py
                              (Pinecone, unchanged)                (windows chunks, LLM
                                    │                                structured extraction)
                                    │                                       │
                                    │                                       ▼
                                    │                              entities.json (new)
                                    │                                       │
                                    │                                       ▼
                                    │                              knowledge_graph.py
                                    │                              (Neo4j MERGE upserts)
                                    │                                       │
                                    ▼                                       ▼
                              rag_query.answer() ──── hybrid routing ──── Neo4j graph
                              (keyword heuristic decides: vector-only,
                               graph-only, or both — merged into one
                               GPT-4o prompt)
```

---

## New / modified files

**`src/extract_entities.py` (new)**
- `_group_chunks_into_windows(chunks, window_tokens=3000)` — groups chunks by `source`, concatenates in `chunk_index` order into larger windows (500-token chunks are too small/arbitrarily cut for reliable entity extraction — a director list can straddle a chunk boundary).
- `_is_low_signal(text)` — heuristic to skip XBRL tag-soup windows (chunks 0–~15) before spending an LLM call.
- `extract_from_window(client, window)` — one OpenAI call per window using **structured output / forced tool calling** (not free-text parsing) against a small fixed schema.
- `extract_entities(chunks_path, out_path, companies=None)` — public entry point; loads `chunks.json`, filters, windows, filters low-signal windows, extracts, writes `data/processed/entities.json`.
- Model: `gpt-4o-mini` for extraction (cheap, NER-appropriate; ~50-60 calls per company at current chunk volume) — keep `gpt-4o` reserved for final answer synthesis.
- Schema — node types: `person`, `organization` (with `role`: issuer/competitor/institutional_investor/supplier_partner/subsidiary/other), `product`. Relationship types: `BOARD_MEMBER_OF`, `EXECUTIVE_OF` (+ `title`), `COMPETES_WITH`, `SHAREHOLDER_OF` (+ `pct_owned`), `PRODUCT_OF`, `SUBSIDIARY_OF`, `PARTNERS_WITH`. Plus a scalar `headquarters_address` fact extracted from cover-page-like windows.
- Output is a flat JSON array (same convention as `chunks.json`) — a human-inspectable QA checkpoint before anything loads into Neo4j.

**`src/knowledge_graph.py` (new)**
- `get_driver()` / `close_driver(driver)` — reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` from `.env` via the existing `load_dotenv()` pattern.
- Idempotent upserts via Cypher `MERGE` (never `CREATE`): `upsert_organization`, `upsert_person`, `upsert_product`, `upsert_filing`, and typed link helpers (`link_board_member`, `link_executive`, `link_competitor`, `link_shareholder`, `link_product`, `link_subsidiary`, `link_partner`, `set_headquarters`, `link_mentioned_in` for provenance).
- `_canonicalize(name)` — light normalization only (trim, strip Inc./Corp./LLC, title-case). Full coreference resolution ("Tim Cook" vs "Timothy D. Cook") is an explicit accepted MVP gap, not solved here.
- `load_graph_records(driver, records)` — dispatches `entities.json` records to the right upsert/link call, batched via `session.execute_write`.
- `clear_graph(driver)` — `MATCH (n) DETACH DELETE n`, opt-in via `--clear` flag only.
- Read helpers for hybrid retrieval: `query_competitors`, `query_board`, `query_executives`, `query_shareholders`, `query_headquarters`, `query_products`, `query_entity_neighborhood(name, hops=1)`.

**`src/rag_query.py` (modify)**
- `_classify_graph_intent(question) -> set[str]` — cheap keyword→query-type heuristic (competitor/rival → competitors; board/director → board; executive/officer/ceo/cfo → executives; shareholder/investor → shareholders; headquarter/hq/address → headquarters; product/brand → products). No LLM router, no agent framework.
- `format_graph_context(facts) -> str` — renders graph facts as a distinct "Knowledge Graph Facts" block.
- `answer(question, company=None, filing_type=None, use_graph=None)` — `use_graph=None` runs the heuristic; explicit `True`/`False` bypasses it (needed for CLI/UI toggle and isolated testing). When graph facts are found, still run vector retrieval too (drop `top_k` 10→5 to control prompt size) and blend both into one GPT-4o call.
- `SYSTEM_PROMPT` gains one line: treat Knowledge Graph Facts as ground truth for entity/relationship questions, preferred over inference from prose.
- Return value gains optional `"graph_facts"` key alongside existing `"sources"`.

**`src/preprocess.py` (modify)**
- Add `process_companies(raw_dir, companies: list[str], out_path)` — loops the existing per-file logic across companies, accumulates in memory, writes `chunks.json` **once** at the end (avoids overwrite footguns from calling `process_directory` N times against the same path). `process_directory()` stays as-is for single-company callers.

**`main.py` (modify)**
- `--ticker` becomes `nargs="+"` (default `["AAPL"]`) at top level and on `fetch`/`preprocess`/`all` subparsers; `cmd_fetch` loops the existing `fetch()` per ticker; `cmd_preprocess` calls new `process_companies()`.
- New subcommand `extract-entities` → `src.extract_entities.extract_entities` (flags `--chunks`, `--entities` default `data/processed/entities.json`, `--company`/companies).
- New subcommand `build-graph` → `src.knowledge_graph` (flags `--entities`, `--clear` opt-in).
- `query` subcommand gains `--use-graph {auto,on,off}` (default `auto`), passed to `answer()`.
- `cmd_all` extended to `fetch → preprocess → index → extract-entities → build-graph`, with a clear warning + graceful skip of the last two steps if `NEO4J_URI` isn't set (no hard crash for users who haven't done Phase 0 yet).

**`app.py` (modify, Phase 6 / optional polish)**
- Sidebar `st.selectbox("Knowledge graph", ["Auto", "On", "Off"])` → `use_graph` param.
- New `st.expander("Knowledge Graph Facts")` rendering `result.get("graph_facts")`, separate from the existing "Sources" expander (unchanged).

**`requirements.txt`** — add `neo4j>=5.20.0`.

**`.env`** — add `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=<set in Neo4j Desktop>`.

---

## Explicit non-goals (deviations from the paper, by design)

- No node-embeddings in Neo4j (redundant with Pinecone).
- No literal replication of the paper's "dump raw chunk text into a generic node" pattern (adds nothing over Pinecone).
- No full entity-resolution/coreference system — name canonicalization is light-touch only.
- No LLM-based intent classifier for graph routing unless the keyword heuristic proves insufficient in testing (Phase 4) — start simple.
- No mandatory graph-visualization UI (pyvis/streamlit-agraph) — flagged as optional Phase 6 stretch only.

---

## Phases

**Phase 0 — Setup (blocking)**
- Add `neo4j` to `requirements.txt`, install it.
- Create a local database in Neo4j Desktop, start it, add `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` to `.env`.

**Phase 1 — Graph plumbing only (proves connectivity)**
- Build `src/knowledge_graph.py` (connection + upsert/link helpers).
- Manually upsert 1-2 test records; confirm round-trip in Neo4j Browser (`MATCH (n) RETURN n LIMIT 25`).

**Phase 2 — Entity extraction, AAPL only, not yet loaded**
- Build `src/extract_entities.py`; run `python main.py extract-entities --ticker AAPL`.
- Inspect `entities.json` against known ground truth: ~9 board members + titles from the signature page, HQ = "One Apple Park Way, Cupertino, California 95014", plausible product names. Confirm the low-signal filter actually skips XBRL chunks (0–~15).

**Phase 3 — Load graph, single company**
- Wire `main.py build-graph`; run against Phase 2 output.
- Verify in Neo4j Browser: `MATCH (o:Organization {ticker:'AAPL'}) RETURN o`; `MATCH (p:Person)-[:BOARD_MEMBER_OF]->(:Organization {ticker:'AAPL'}) RETURN p.name, p.title` returns ~9 matching names; HQ address matches exactly.
- Idempotency check: re-run `build-graph`, confirm `MATCH (n) RETURN count(n)` unchanged.

**Phase 4 — Hybrid retrieval, still single company**
- Extend `rag_query.answer()` per the design above.
- Run CLI test queries (see Verification) confirming graph-routed questions are precise/non-hallucinated and non-graph questions are unaffected by the change.

**Phase 5 — Multi-company expansion**
- `python main.py fetch --ticker AAPL MSFT GOOGL --types 10-K --limit 1`, then `preprocess`, `index`, `extract-entities`, `build-graph` over the combined dataset.
- Verify: `MATCH (a:Organization {ticker:'AAPL'})-[:COMPETES_WITH]-(c) RETURN c.name`; test "Does Microsoft compete with Apple?"
- Expect sparse competitor edges (Apple's 10-K competition language is narrative, not a named list) — document this as a finding, not a bug to chase.

**Phase 6 — UI polish (optional/stretch)**
- Streamlit toggle + graph-facts expander; optional graph visualization view.

---

## Verification

**Neo4j Browser, per phase:** node/edge counts (`MATCH (n) RETURN count(n)`, `MATCH ()-[r]->() RETURN type(r), count(*)`), idempotency re-run, visual subgraph (`MATCH (o:Organization {ticker:'AAPL'})-[r]-(n) RETURN o,r,n LIMIT 50`).

**CLI smoke tests** (`python main.py query --question "..."`), run before/after each phase:
- Pure-vector regression (unaffected by changes): *"What was Apple's total revenue in fiscal 2025?"*, *"What are Apple's main risk factors?"* — `graph_facts` should be empty, answer quality unchanged.
- Graph-routed, single company: *"Who is on Apple's board of directors?"* — expect the ~9 verified names/titles, no hallucination. *"Where is Apple headquartered?"* — expect the exact verified address.
- Graph-routed, sparse case: *"Who are Apple's competitors?"* — deliverable is "no fabricated list," not necessarily a rich one.
- Cross-company (Phase 5+): *"Does Microsoft compete with Apple?"*
- Toggle check: same question with `--use-graph off` vs `--use-graph on` to confirm the flag changes behavior.
