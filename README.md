# LLM Analysis of 10-K Filings (RAG)

Q&A system for SEC filings using Retrieval-Augmented Generation. Ask natural language questions about any public company's 10-K or 10-Q filings and get answers grounded in the actual documents.

Based on: *LLM Analysis of 10-K and 10-Q Filings: RAG Results* (IJRTI, 2024)

---

## How it works

1. **Fetch** — download filings from SEC EDGAR
2. **Preprocess** — extract and chunk text into ~500 token pieces
3. **Index** — embed chunks with OpenAI and store in Pinecone
4. **Query** — embed your question, retrieve top 10 relevant chunks, GPT-4o answers

This is RAG (Retrieval-Augmented Generation). Instead of feeding a 419-page document to an LLM, only the most relevant paragraphs are retrieved and sent as context.

---

## Stack

- **LLM** — GPT-4o
- **Embeddings** — OpenAI `text-embedding-3-small`
- **Vector DB** — Pinecone
- **Filing source** — SEC EDGAR via `sec-edgar-downloader`
- **UI** — Streamlit

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add API keys**
```bash
cp .env.example .env
```
Fill in `.env`:
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

---

## Usage

**Run the full pipeline for a company:**
```bash
python main.py fetch --ticker AAPL --types 10-K --limit 3
python main.py preprocess --ticker AAPL
python main.py index
```

**Launch the UI:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Type a question, click Ask.

**Or query from the terminal:**
```bash
python main.py query --question "What was Apple's total revenue in fiscal 2025?"
```

---

## CLI reference

| Command | What it does |
|---------|-------------|
| `fetch` | Download filings from SEC EDGAR |
| `preprocess` | Parse HTML, clean, chunk into 500-token pieces |
| `index` | Embed chunks and upload to Pinecone |
| `query` | Ask a question and get an answer |
| `all` | Run fetch + preprocess + index in one go |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `AAPL` | Stock ticker |
| `--types` | `10-K 10-Q` | Filing types to fetch |
| `--limit` | `3` | Number of filings per type |

---

## Example questions

- What was Apple's total revenue in fiscal 2025?
- What are Apple's main risk factors?
- How does Apple describe its competitive landscape?
- What services does Apple offer?
- How much cash does Apple have on hand?

---

## Project structure

```
├── app.py                  # Streamlit UI
├── main.py                 # CLI entrypoint
├── src/
│   ├── fetch_filings.py    # Download from SEC EDGAR
│   ├── preprocess.py       # Parse, clean, chunk
│   ├── index_documents.py  # Embed + upsert to Pinecone
│   └── rag_query.py        # RAG query pipeline
├── data/
│   ├── raw/                # Downloaded SEC filings
│   └── processed/          # chunks.json
├── requirements.txt
└── .env.example
```
