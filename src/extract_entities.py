"""
Extract entities and relationships from filing chunks via LLM structured output.
Writes data/processed/entities.json — a flat record list consumed by knowledge_graph.py.
"""
import json
import os
import re
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EXTRACT_MODEL = "gpt-4o-mini"
WINDOW_TOKENS = 3000
ENCODING = tiktoken.get_encoding("cl100k_base")

EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_entities",
        "description": "Record entities and relationships found in an SEC filing excerpt.",
        "parameters": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Full name of the person"},
                            "relationship": {
                                "type": "string",
                                "enum": ["BOARD_MEMBER_OF", "EXECUTIVE_OF", "NONE"],
                                "description": "Relationship to the filing company. Directors -> BOARD_MEMBER_OF; officers (CEO, CFO, VP, controller) -> EXECUTIVE_OF. A person can be both; pick the role stated in this excerpt.",
                            },
                            "title": {"type": "string", "description": "Exact title as stated, e.g. 'Chief Executive Officer', 'Director'"},
                        },
                        "required": ["name", "relationship"],
                    },
                },
                "organizations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {
                                "type": "string",
                                "enum": ["competitor", "institutional_investor", "supplier_partner", "subsidiary", "other"],
                            },
                            "relationship": {
                                "type": "string",
                                "enum": ["COMPETES_WITH", "SHAREHOLDER_OF", "SUBSIDIARY_OF", "PARTNERS_WITH", "NONE"],
                                "description": "Relationship to the filing company, only if explicitly stated or clearly implied in this excerpt.",
                            },
                            "pct_owned": {"type": "number", "description": "Ownership percentage if stated (SHAREHOLDER_OF only)"},
                        },
                        "required": ["name", "role", "relationship"],
                    },
                },
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named products/services/brands of the filing company (e.g. iPhone, Mac, Azure). Not generic categories.",
                },
                "headquarters_address": {
                    "type": "string",
                    "description": "The filing company's principal executive office street address, only if stated verbatim in this excerpt.",
                },
                "issuer_legal_name": {
                    "type": "string",
                    "description": "The filing company's exact legal name (e.g. 'Apple Inc.'), only if stated in this excerpt.",
                },
            },
            "required": ["people", "organizations", "products"],
        },
    },
}

SYSTEM_PROMPT = """You extract structured entities from SEC filing text for a knowledge graph.

Rules:
- Only record what is explicitly stated in the excerpt. Never infer from outside knowledge.
- The filing company (issuer) is given; do not list it as an organization.
- Skip generic references ("our suppliers", "certain competitors") — named entities only.
- People: only those affiliated with the issuer (directors, officers, signatories).
- Return empty arrays when the excerpt has nothing relevant (financial tables, legal boilerplate)."""


def _filing_year(source: str) -> int:
    """Filing year from the SEC accession number in the source path,
    e.g. .../0000320193-23-000106/... -> 2023. 0 if underivable."""
    m = re.search(r"-(\d{2})-\d{6}", source)
    return 2000 + int(m.group(1)) if m else 0


def _group_chunks_into_windows(chunks: list[dict], window_tokens: int = WINDOW_TOKENS) -> list[dict]:
    """Group chunks by source file, concatenate in chunk_index order into larger windows.

    500-token chunks are too small for reliable entity extraction (a director list can
    straddle a boundary); bigger windows also cut LLM call count."""
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c)

    windows = []
    for source, group in by_source.items():
        group.sort(key=lambda c: c["chunk_index"])
        buf, buf_tokens, start_idx = [], 0, None
        for c in group:
            if start_idx is None:
                start_idx = c["chunk_index"]
            buf.append(c["text"])
            buf_tokens += c.get("token_count") or len(ENCODING.encode(c["text"]))
            if buf_tokens >= window_tokens:
                windows.append({
                    "text": " ".join(buf),
                    "source": source,
                    "company": group[0]["company"],
                    "filing_type": group[0]["filing_type"],
                    "chunk_range": (start_idx, c["chunk_index"]),
                })
                buf, buf_tokens, start_idx = [], 0, None
        if buf:
            windows.append({
                "text": " ".join(buf),
                "source": source,
                "company": group[0]["company"],
                "filing_type": group[0]["filing_type"],
                "chunk_range": (start_idx, group[-1]["chunk_index"]),
            })
    return windows


def _is_low_signal(text: str) -> bool:
    """Skip inline-XBRL / cover-page tag soup (typically the first ~15 chunks of a filing).

    Heuristics: very low ratio of alphabetic words to tokens, or dominated by
    xbrl/iso4217-style identifiers."""
    sample = text[:4000]
    if re.search(r"(iso4217|xbrli|us-gaap|dei:|xmlns)", sample, re.IGNORECASE):
        # XBRL identifiers present — check whether prose still dominates
        words = re.findall(r"[A-Za-z]{3,}", sample)
        sentences = re.findall(r"[a-z],?\s+[a-z]+\s+[a-z]+\s+[a-z]+", sample)
        if len(sentences) < 20:
            return True
    words = re.findall(r"\b[a-zA-Z]{2,}\b", sample)
    if len(words) < 100:
        return True
    # Ratio of dictionary-ish lowercase words to all tokens; tag soup is mostly
    # camelCase identifiers, numbers, and punctuation.
    lower_words = [w for w in words if w.islower()]
    if len(lower_words) / max(len(words), 1) < 0.35:
        return True
    return False


def extract_from_window(client: OpenAI, window: dict) -> list[dict]:
    """One LLM call per window; returns flat entity records."""
    company = window["company"]
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Filing company (issuer): {company}\n"
                    f"Filing type: {window['filing_type']}\n\n"
                    f"Excerpt:\n{window['text']}"
                ),
            },
        ],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_entities"}},
        temperature=0,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return []
    try:
        data = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError:
        return []

    records = []
    meta = {
        "issuer": company,
        "source": window["source"],
        "filing_type": window["filing_type"],
        "filing_year": _filing_year(window["source"]),
    }

    for p in data.get("people", []) or []:
        if not p.get("name"):
            continue
        rec = {"type": "person", "name": p["name"], **meta}
        if p.get("relationship") and p["relationship"] != "NONE":
            rec["relationship"] = p["relationship"]
        if p.get("title"):
            rec["title"] = p["title"]
        records.append(rec)

    for o in data.get("organizations", []) or []:
        if not o.get("name"):
            continue
        rec = {"type": "organization", "name": o["name"], "role": o.get("role", "other"), **meta}
        if o.get("relationship") and o["relationship"] != "NONE":
            rec["relationship"] = o["relationship"]
        if o.get("pct_owned") is not None:
            rec["pct_owned"] = o["pct_owned"]
        records.append(rec)

    for prod in data.get("products", []) or []:
        if prod and isinstance(prod, str):
            records.append({"type": "product", "name": prod, **meta})

    hq = data.get("headquarters_address")
    if hq and isinstance(hq, str) and hq.strip():
        records.append({"type": "headquarters", "address": hq.strip(), **meta})

    legal = data.get("issuer_legal_name")
    if legal and isinstance(legal, str) and legal.strip():
        records.append({"type": "issuer_legal_name", "name": legal.strip(), **meta})

    return records


_TITLE_WORDS = {
    "officer", "executive", "chief", "chair", "chairman", "chairperson", "head",
    "heads", "counsel", "president", "committee", "member", "compliance",
    "audit", "accounting", "assurance", "secretary", "treasurer", "controller",
    "evp", "svp", "cvp", "vp",
}


def _looks_like_title_not_name(name: str) -> bool:
    """Extraction sometimes returns a role ('Chief Executive Officer') as a person
    name. Real names never contain these words."""
    return any(w in _TITLE_WORDS for w in re.findall(r"[a-z]+", name.lower()))


def _clean_records(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        if "name" in rec:
            # normalize curly apostrophes so "O'Brien"/"O’Brien" dedupe together
            rec["name"] = rec["name"].replace("’", "'").replace("‘", "'")
        if rec["type"] == "person" and _looks_like_title_not_name(rec["name"]):
            continue
        out.append(rec)
    return out


def _dedupe(records: list[dict]) -> list[dict]:
    """Drop exact duplicates across windows. Keyed per filing year so the same
    person/org in different filings stays as separate year-scoped records;
    within a year keep the record with the most fields (e.g. one with a title)."""
    best: dict[tuple, dict] = {}
    hq_seen: set[tuple] = set()
    legal_seen: set[str] = set()
    out = []
    for rec in records:
        year = rec.get("filing_year", 0)
        if rec["type"] == "headquarters":
            key = (rec["issuer"], year)
            if key in hq_seen:
                continue
            hq_seen.add(key)
            out.append(rec)
            continue
        if rec["type"] == "issuer_legal_name":
            if rec["issuer"] in legal_seen:
                continue
            legal_seen.add(rec["issuer"])
            out.append(rec)
            continue
        key = (rec["type"], rec["name"].lower(), rec["issuer"], rec.get("relationship"), year)
        if key not in best or len(rec) > len(best[key]):
            best[key] = rec
    out.extend(best.values())
    return out


def extract_entities(chunks_path: Path, out_path: Path, companies: list[str] = None) -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(chunks_path) as f:
        chunks = json.load(f)
    if companies:
        wanted = {c.upper() for c in companies}
        chunks = [c for c in chunks if c["company"].upper() in wanted]
    if not chunks:
        print("No chunks matched — run preprocess first (or check --ticker).")
        return

    windows = _group_chunks_into_windows(chunks)
    live = [w for w in windows if not _is_low_signal(w["text"])]
    print(f"{len(chunks)} chunks -> {len(windows)} windows, {len(live)} after low-signal filter")

    all_records = []
    issuers_seen = set()
    for i, w in enumerate(live):
        if w["company"] not in issuers_seen:
            issuers_seen.add(w["company"])
            all_records.append({
                "type": "issuer", "name": w["company"], "ticker": w["company"],
                "issuer": w["company"], "source": w["source"], "filing_type": w["filing_type"],
            })
        try:
            recs = extract_from_window(client, w)
        except Exception as e:
            print(f"  window {i + 1}/{len(live)} FAILED: {e}")
            continue
        all_records.extend(recs)
        print(f"  window {i + 1}/{len(live)} [{w['company']} chunks {w['chunk_range'][0]}-{w['chunk_range'][1]}]: {len(recs)} records")

    # Merge: preserve existing records for companies NOT re-extracted this run,
    # so incremental runs never silently drop earlier companies.
    extracted = {w["company"] for w in live}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        kept = [r for r in existing if r["issuer"] not in extracted]
        if kept:
            print(f"Keeping {len(kept)} existing records for other companies")
        all_records = kept + all_records

    deduped = _dedupe(_clean_records(all_records))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"\n{len(all_records)} raw -> {len(deduped)} deduped records -> {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract entities from filing chunks")
    parser.add_argument("--chunks", default="data/processed/chunks.json")
    parser.add_argument("--entities", default="data/processed/entities.json")
    parser.add_argument("--ticker", nargs="+", default=None, help="Limit to these companies")
    args = parser.parse_args()
    extract_entities(Path(args.chunks), Path(args.entities), companies=args.ticker)
