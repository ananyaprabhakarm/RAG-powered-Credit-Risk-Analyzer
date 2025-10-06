## Credit Risk Analyzer (ML + RAG)

A minimal, end-to-end scaffold that combines a simple risk model (structured data) with Retrieval-Augmented Generation (RAG) over regulations, internal policies, and past cases to generate human-readable, regulation-aware credit decisions.

### Features
- Risk score from borrower profile (rule-based starter; pluggable ML later)
- RAG over local knowledge base using FAISS + sentence-transformers
- Natural language explanation citing retrieved snippets
- Streamlit UI for quick demo

### Project Structure
```
src/
  app/streamlit_app.py        # Streamlit UI
  risk_model/model.py         # Simple probability model + categorization
  rag/ingest.py               # Build FAISS index from txt/PDF (txt starter)
  rag/retriever.py            # Retriever wrapper
  explain/generator.py        # Explanation generator (templated + optional OpenAI)
  utils/text.py               # Small helpers
  config.py                   # Paths, model names, constants
data/
  regulations/*.txt
  policies/*.txt
  cases/*.txt
artifacts/
  faiss_index.bin
  embeddings.npy
  docstore.json
```

### Quickstart
1) Create and activate a virtual environment
```bash
python -m venv .venv && source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) (Optional) Set OpenAI key for LLM-based explanations
```bash
export OPENAI_API_KEY=your_key
```

4) Ingest sample documents (build FAISS index)
```bash
python -m src.rag.ingest
```

5) Run the UI
```bash
streamlit run src/app/streamlit_app.py
```

### Notes
- By default, explanations use a deterministic template combining model features and retrieved snippets. If `OPENAI_API_KEY` is present, the generator will use the LLM to craft more fluent explanations.
- Replace the simple rule-based model with your trained classifier by swapping implementations in `src/risk_model/model.py`.
- Add your PDFs or policies by converting to text (`.txt`) and placing them under `data/` to start. PDF parsing can be added later.

### License
MIT


