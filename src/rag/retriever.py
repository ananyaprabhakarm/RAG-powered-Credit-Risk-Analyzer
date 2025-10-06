from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    INDEX_PATH,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


class Retriever:
    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not DOCSTORE_PATH.exists():
            raise RuntimeError(
                "Index or docstore not found. Run ingestion: `python -m src.rag.ingest`"
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.docs: List[Dict] = json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        q_emb = _normalize(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            preview = doc["text"][:200].replace("\n", " ")
            results.append({
                "source": doc.get("source", str(idx)),
                "preview": preview,
                "score": float(score),
                "text": doc["text"],
            })
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


