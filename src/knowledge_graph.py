"""
Neo4j knowledge graph layer: idempotent upserts of extracted entities/relationships
and read helpers for hybrid graph+vector retrieval.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

_ORG_SUFFIX_RE = re.compile(
    r"[,.]?\s+(Inc|Incorporated|Corp|Corporation|LLC|L\.L\.C|Ltd|Limited|Co|Company|PLC|plc|LP|L\.P)\.?$"
)


def get_driver():
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def close_driver(driver):
    driver.close()


def _canonicalize(name: str) -> str:
    """Light normalization: trim, strip corporate suffixes, normalize whitespace.
    Full coreference resolution is out of scope."""
    name = re.sub(r"\s+", " ", name or "").strip()
    name = _ORG_SUFFIX_RE.sub("", name).strip().rstrip(",.")
    return name


# ---------------------------------------------------------------------------
# Write helpers (all MERGE-based, idempotent)
# ---------------------------------------------------------------------------

def upsert_organization(tx, name: str, ticker: str = None, role: str = None, legal_name: str = None):
    name = _canonicalize(name)
    if not name:
        return
    tx.run(
        """
        MERGE (o:Organization {name: $name})
        SET o.ticker = coalesce($ticker, o.ticker),
            o.role = coalesce($role, o.role),
            o.legal_name = coalesce($legal_name, o.legal_name)
        """,
        name=name, ticker=ticker, role=role, legal_name=legal_name,
    )


def upsert_person(tx, name: str):
    name = _canonicalize(name)
    if not name:
        return
    tx.run("MERGE (p:Person {name: $name})", name=name)


def upsert_product(tx, name: str):
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name:
        return
    tx.run("MERGE (pr:Product {name: $name})", name=name)


def upsert_filing(tx, company: str, filing_type: str, source: str):
    tx.run(
        """
        MERGE (f:Filing {source: $source})
        SET f.company = $company, f.filing_type = $filing_type
        """,
        company=company, filing_type=filing_type, source=source,
    )


def link_board_member(tx, person: str, org: str, title: str = None):
    tx.run(
        """
        MERGE (p:Person {name: $person})
        MERGE (o:Organization {name: $org})
        MERGE (p)-[r:BOARD_MEMBER_OF]->(o)
        SET r.title = coalesce($title, r.title)
        """,
        person=_canonicalize(person), org=_canonicalize(org), title=title,
    )


def link_executive(tx, person: str, org: str, title: str = None):
    tx.run(
        """
        MERGE (p:Person {name: $person})
        MERGE (o:Organization {name: $org})
        MERGE (p)-[r:EXECUTIVE_OF]->(o)
        SET r.title = coalesce($title, r.title)
        """,
        person=_canonicalize(person), org=_canonicalize(org), title=title,
    )


def link_competitor(tx, org_a: str, org_b: str):
    a, b = _canonicalize(org_a), _canonicalize(org_b)
    if not a or not b or a == b:
        return
    tx.run(
        """
        MERGE (a:Organization {name: $a})
        MERGE (b:Organization {name: $b})
        MERGE (a)-[:COMPETES_WITH]->(b)
        """,
        a=a, b=b,
    )


def link_shareholder(tx, holder: str, org: str, pct_owned: float = None):
    tx.run(
        """
        MERGE (h:Organization {name: $holder})
        MERGE (o:Organization {name: $org})
        MERGE (h)-[r:SHAREHOLDER_OF]->(o)
        SET r.pct_owned = coalesce($pct, r.pct_owned)
        """,
        holder=_canonicalize(holder), org=_canonicalize(org), pct=pct_owned,
    )


def link_product(tx, product: str, org: str):
    product = re.sub(r"\s+", " ", product or "").strip()
    if not product:
        return
    tx.run(
        """
        MERGE (pr:Product {name: $product})
        MERGE (o:Organization {name: $org})
        MERGE (pr)-[:PRODUCT_OF]->(o)
        """,
        product=product, org=_canonicalize(org),
    )


def link_subsidiary(tx, subsidiary: str, parent: str):
    tx.run(
        """
        MERGE (s:Organization {name: $sub})
        MERGE (p:Organization {name: $parent})
        MERGE (s)-[:SUBSIDIARY_OF]->(p)
        """,
        sub=_canonicalize(subsidiary), parent=_canonicalize(parent),
    )


def link_partner(tx, org_a: str, org_b: str):
    a, b = _canonicalize(org_a), _canonicalize(org_b)
    if not a or not b or a == b:
        return
    tx.run(
        """
        MERGE (a:Organization {name: $a})
        MERGE (b:Organization {name: $b})
        MERGE (a)-[:PARTNERS_WITH]->(b)
        """,
        a=a, b=b,
    )


def set_headquarters(tx, org: str, address: str):
    tx.run(
        """
        MERGE (o:Organization {name: $org})
        SET o.headquarters_address = $address
        """,
        org=_canonicalize(org), address=address.strip(),
    )


def link_mentioned_in(tx, node_label: str, node_name: str, source: str):
    """Provenance edge: entity -> Filing it was extracted from."""
    if node_label not in ("Person", "Organization", "Product"):
        return
    name = node_name if node_label == "Product" else _canonicalize(node_name)
    tx.run(
        f"""
        MERGE (n:{node_label} {{name: $name}})
        MERGE (f:Filing {{source: $source}})
        MERGE (n)-[:MENTIONED_IN]->(f)
        """,
        name=name, source=source,
    )


def clear_graph(driver):
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
    print("Graph cleared.")


# ---------------------------------------------------------------------------
# Loading entities.json into the graph
# ---------------------------------------------------------------------------

def _load_record(tx, rec: dict):
    rtype = rec.get("type")
    issuer = rec.get("issuer", "")
    source = rec.get("source")

    if rtype == "person":
        upsert_person(tx, rec["name"])
        rel = rec.get("relationship")
        if rel == "BOARD_MEMBER_OF":
            link_board_member(tx, rec["name"], issuer, rec.get("title"))
        elif rel == "EXECUTIVE_OF":
            link_executive(tx, rec["name"], issuer, rec.get("title"))
        if source:
            link_mentioned_in(tx, "Person", rec["name"], source)

    elif rtype == "organization":
        upsert_organization(tx, rec["name"], ticker=rec.get("ticker"), role=rec.get("role"))
        rel = rec.get("relationship")
        if rel == "COMPETES_WITH":
            link_competitor(tx, rec["name"], issuer)
        elif rel == "SHAREHOLDER_OF":
            link_shareholder(tx, rec["name"], issuer, rec.get("pct_owned"))
        elif rel == "SUBSIDIARY_OF":
            link_subsidiary(tx, rec["name"], issuer)
        elif rel == "PARTNERS_WITH":
            link_partner(tx, rec["name"], issuer)
        if source:
            link_mentioned_in(tx, "Organization", rec["name"], source)

    elif rtype == "product":
        link_product(tx, rec["name"], issuer)
        if source:
            link_mentioned_in(tx, "Product", rec["name"], source)

    elif rtype == "headquarters":
        set_headquarters(tx, issuer, rec["address"])

    elif rtype == "issuer":
        upsert_organization(tx, rec["name"], ticker=rec.get("ticker"), role="issuer")

    elif rtype == "issuer_legal_name":
        # Issuer nodes are keyed by ticker; attach the legal name as a property so
        # questions phrased with the common name ("Apple") can resolve to the ticker.
        upsert_organization(tx, issuer, ticker=issuer, role="issuer", legal_name=rec["name"])


def load_graph_records(driver, records: list[dict], batch_size: int = 50):
    loaded = 0
    with driver.session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            def _write_batch(tx, batch=batch):
                for rec in batch:
                    _load_record(tx, rec)

            session.execute_write(_write_batch)
            loaded += len(batch)
            print(f"  loaded {loaded}/{len(records)} records")
    print(f"Done: {loaded} records loaded into Neo4j.")


def build_graph(entities_path: Path, clear: bool = False):
    with open(entities_path) as f:
        records = json.load(f)
    driver = get_driver()
    try:
        if clear:
            clear_graph(driver)
        load_graph_records(driver, records)
        with driver.session() as session:
            n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"Graph now has {n} nodes, {r} relationships.")
    finally:
        close_driver(driver)


# ---------------------------------------------------------------------------
# Read helpers for hybrid retrieval
# ---------------------------------------------------------------------------

def query_board(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (p:Person)-[r:BOARD_MEMBER_OF]->(o:Organization)
        WHERE o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
           OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company)
        RETURN DISTINCT p.name AS name, r.title AS title, o.name AS org
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_executives(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization)
        WHERE o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
           OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company)
        RETURN DISTINCT p.name AS name, r.title AS title, o.name AS org
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_competitors(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (o:Organization)-[:COMPETES_WITH]-(c:Organization)
        WHERE o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
           OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company)
        RETURN DISTINCT c.name AS name, o.name AS org
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_shareholders(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (h:Organization)-[r:SHAREHOLDER_OF]->(o:Organization)
        WHERE o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
           OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company)
        RETURN DISTINCT h.name AS name, r.pct_owned AS pct_owned, o.name AS org
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_headquarters(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (o:Organization)
        WHERE (o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
               OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company))
          AND o.headquarters_address IS NOT NULL
        RETURN o.name AS org, o.headquarters_address AS address
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_products(session, company: str) -> list[dict]:
    result = session.run(
        """
        MATCH (pr:Product)-[:PRODUCT_OF]->(o:Organization)
        WHERE o.ticker = $company OR toLower(o.name) CONTAINS toLower($company)
           OR toLower(coalesce(o.legal_name, '')) CONTAINS toLower($company)
        RETURN DISTINCT pr.name AS name, o.name AS org
        """,
        company=company,
    )
    return [dict(rec) for rec in result]


def query_issuers(session) -> list[dict]:
    """All issuer organizations — used to resolve which company a question is about."""
    result = session.run(
        """
        MATCH (o:Organization {role: 'issuer'})
        RETURN o.name AS name, o.ticker AS ticker, o.legal_name AS legal_name
        """
    )
    return [dict(rec) for rec in result]


def query_entity_neighborhood(session, name: str, hops: int = 1) -> list[dict]:
    result = session.run(
        f"""
        MATCH (n)-[r*1..{int(hops)}]-(m)
        WHERE toLower(n.name) CONTAINS toLower($name)
          AND NOT m:Filing
        UNWIND r AS rel
        RETURN DISTINCT startNode(rel).name AS from, type(rel) AS rel, endNode(rel).name AS to
        LIMIT 100
        """,
        name=name,
    )
    return [dict(rec) for rec in result]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load extracted entities into Neo4j")
    parser.add_argument("--entities", default="data/processed/entities.json")
    parser.add_argument("--clear", action="store_true", help="Wipe graph before loading")
    args = parser.parse_args()
    build_graph(Path(args.entities), clear=args.clear)
