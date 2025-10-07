import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
DATA_DIR = PROJECT_ROOT / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"
POLICIES_DIR = DATA_DIR / "policies"
CASES_DIR = DATA_DIR / "cases"

# Artifacts (docstore kept locally; vector index goes to Pinecone)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DOCSTORE_PATH = ARTIFACTS_DIR / "docstore.json"

# Embedding model
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Risk model thresholds
RISK_HIGH_THRESHOLD = 0.65
RISK_MEDIUM_THRESHOLD = 0.4

# LLM config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Pinecone config
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "credit-risk-regs")


