from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    REGULATIONS_DIR,
    POLICIES_DIR,
    CASES_DIR,
    ARTIFACTS_DIR,
    INDEX_PATH,
    EMBEDDINGS_PATH,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
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

    docs = _read_text_files([REGULATIONS_DIR, POLICIES_DIR, CASES_DIR])
    if not docs:
        raise RuntimeError("No .txt documents found under data/. Add sample files and retry.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False)
    embeddings = _normalize(embeddings.astype(np.float32))

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)
    DOCSTORE_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2))

    print(f"Indexed {len(docs)} documents → {INDEX_PATH}")


if __name__ == "__main__":
    build_index()


