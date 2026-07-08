"""
Embed chunks with OpenAI text-embedding-3-small and upsert into Pinecone.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100
METRIC = "cosine"


def get_or_create_index(pc: Pinecone, index_name: str) -> object:
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [r.embedding for r in response.data]


def upsert_chunks(chunks_path: Path, index_name: str) -> None:
    with open(chunks_path) as f:
        chunks = json.load(f)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = get_or_create_index(pc, index_name)

    print(f"Upserting {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(client, texts)

        vectors = [
            (
                c["id"],
                emb,
                {k: v for k, v in c.items() if k != "id"},
            )
            for c, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors)
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    stats = index.describe_index_stats()
    print(f"\nIndex '{index_name}': {stats.total_vector_count} vectors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/chunks.json")
    parser.add_argument("--index", default=os.getenv("PINECONE_INDEX_NAME", "10k-filings"))
    args = parser.parse_args()

    upsert_chunks(Path(args.chunks), args.index)
