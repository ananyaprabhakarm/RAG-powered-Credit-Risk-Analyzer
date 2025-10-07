from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    REGULATIONS_DIR,
    POLICIES_DIR,
    CASES_DIR,
    ARTIFACTS_DIR,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
)


def _read_text_files(directories: List[Path]) -> List[Dict]:
    documents: List[Dict] = []
    for directory in directories:
        if not directory.exists():
            continue
        for p in sorted(directory.rglob("*.txt")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = p.relative_to(directory.parent)
            documents.append({
                "id": str(rel),
                "source": str(rel),
                "text": text.strip(),
            })
    return documents


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def build_index() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")

    docs = _read_text_files([REGULATIONS_DIR, POLICIES_DIR, CASES_DIR])
    if not docs:
        raise RuntimeError("No .txt documents found under data/. Add sample files and retry.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False)
    embeddings = _normalize(embeddings.astype(np.float32))

    # Init Pinecone and ensure index exists
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    dim = embeddings.shape[1]
    existing = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=dim, metric="cosine", spec=spec)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Upsert vectors with metadata
    vectors = []
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        vectors.append({
            "id": doc["id"],
            "values": emb.tolist(),
            "metadata": {
                "source": doc["source"],
                "text": doc["text"][:4000],
            },
        })
    # Batch to avoid payload limits
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start:start+batch_size])

    # Persist local docstore for full text/citations
    DOCSTORE_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2))

    print(f"Indexed {len(docs)} documents → Pinecone index '{PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    build_index()


