import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"
POLICIES_DIR = DATA_DIR / "policies"
CASES_DIR = DATA_DIR / "cases"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
DOCSTORE_PATH = ARTIFACTS_DIR / "docstore.json"

# Embedding model
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Risk model thresholds
RISK_HIGH_THRESHOLD = 0.65
RISK_MEDIUM_THRESHOLD = 0.4

# LLM config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


