from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from src.config import (
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

class Retriever:
    def __init__(self) -> None:
        if not DOCSTORE_PATH.exists():
            raise RuntimeError(
                "Docstore not found. Run ingestion: `python -m src.rag.ingest`"
            )
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.docs: List[Dict] = json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        q_emb = _normalize(q_emb)
        res = self.index.query(vector=q_emb[0].tolist(), top_k=top_k, include_metadata=True)
        results: List[Dict] = []
        for match in res.matches or []:
            meta = match.metadata or {}
            text = meta.get("text", "")
            preview = text[:200].replace("\n", " ")
            results.append({
                "source": meta.get("source", match.id),
                "preview": preview,
                "score": float(match.score or 0.0),
                "text": text,
            })
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


