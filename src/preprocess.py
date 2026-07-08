"""
Parse SEC filing HTML/PDF, clean text, and chunk into ~500-token pieces with overlap.
Outputs data/processed/chunks.json with metadata per chunk.
"""
import json
import re
import uuid
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    return text.strip()


def _chunk(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_text = ENCODING.decode(tokens[start:end])
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_tokens - overlap
    return chunks


def parse_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return _clean(soup.get_text(separator=" "))


def parse_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return _clean(" ".join(pages))


def extract_primary_doc_from_submission(path: Path) -> str:
    """Extract the primary filing HTML from a SEC full-submission.txt SGML container."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Find the first <DOCUMENT> block with SEQUENCE 1 (the main filing)
    doc_blocks = re.split(r"<DOCUMENT>", content)[1:]  # skip preamble
    for block in doc_blocks:
        seq_match = re.search(r"<SEQUENCE>(\d+)", block)
        if seq_match and int(seq_match.group(1)) == 1:
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL)
            if text_match:
                inner = text_match.group(1).strip()
                # If it looks like HTML/XBRL, parse it; otherwise return plain text
                if "<html" in inner.lower() or "<xbrl" in inner.lower() or "<body" in inner.lower():
                    return parse_html(inner)
                return _clean(inner)
    # Fallback: strip all SGML tags and return plain text
    plain = re.sub(r"<[^>]+>", " ", content)
    return _clean(plain)


def process_file(path: Path, company: str, filing_type: str) -> list[dict]:
    suffix = path.suffix.lower()
    if path.name == "full-submission.txt":
        text = extract_primary_doc_from_submission(path)
    elif suffix in (".htm", ".html"):
        text = parse_html(path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".pdf":
        text = parse_pdf(path)
    else:
        return []

    chunks = _chunk(text)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "source": str(path),
            "company": company,
            "filing_type": filing_type,
            "chunk_index": i,
            "token_count": _count_tokens(chunk),
        })
    return records


def _collect_company_chunks(raw_dir: Path, company: str) -> list[dict]:
    all_chunks = []
    for filing_type in ["10-K", "10-Q"]:
        pattern = raw_dir / "sec-edgar-filings" / company / filing_type
        if not pattern.exists():
            continue
        for html_file in sorted(pattern.rglob("full-submission.txt")) + sorted(pattern.rglob("*.htm")):
            print(f"Processing {html_file}...")
            chunks = process_file(html_file, company, filing_type)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks")
    return all_chunks


def _write_chunks(all_chunks: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\nTotal chunks: {len(all_chunks)} -> {out_path}")


def process_directory(raw_dir: Path, company: str, out_path: Path) -> None:
    _write_chunks(_collect_company_chunks(raw_dir, company), out_path)


def process_companies(raw_dir: Path, companies: list[str], out_path: Path) -> None:
    """Process several companies, write chunks.json once at the end."""
    all_chunks = []
    for company in companies:
        all_chunks.extend(_collect_company_chunks(raw_dir, company))
    _write_chunks(all_chunks, out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with downloaded filings")
    parser.add_argument("--ticker", default="DCTK", help="Ticker (used as folder name)")
    parser.add_argument("--out", default="data/processed/chunks.json")
    args = parser.parse_args()

    process_directory(
        raw_dir=Path(args.raw_dir),
        company=args.ticker,
        out_path=Path(args.out),
    )
