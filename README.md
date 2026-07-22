# 🏥 X-Ray Manual RAG Assistant (MedTech RAG)

## 🚀 V2 Development in Progress
Active development on hybrid search, chunking strategies, and enterprise resilience patterns.  
See the [`feature/phase-0.5-markdown-recursive-chunking`](https://github.com/sagarmishra0112/MedTech-RAG/tree/feature/phase-0.5-markdown-recursive-chunking)


A full-stack Retrieval-Augmented Generation (RAG) system built on a real-world scanned medical equipment manual (Allengers X-Ray Generator). The pipeline runs from raw PDF ingestion all the way to a live conversational chat interface.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Streamlit Chat UI  (port 8501)             │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP POST /query
┌─────────────────────▼───────────────────────────────────┐
│              FastAPI Backend  (port 8000)               │
│                src/api.py — Orchestration Layer         │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐  ┌────────────▼──────────────────┐
│   Vector Retrieval  │  │       LLM Generation          │
│   src/embedding.py  │  │       src/llm.py              │
│   ChromaDB (local)  │  │   OpenAI · Anthropic · Gemini │
│   HuggingFace Emb.  │  │   (Factory Pattern)           │
└─────────────────────┘  └───────────────────────────────┘
           ▲
┌──────────┴─────────────────────────────────────────────┐
│             Offline Data Pipeline (runs once)          │
│    ingestion.py → preprocessing.py → chunking.py       │
└────────────────────────────────────────────────────────┘
```

---

## 🧠 Engineering Decisions

### 1. Decoupled Ingestion & Preprocessing
`ingestion.py` runs once and saves three intermediate artifacts to disk (`fulltext.txt`, `tables.json`, `full_markdown.md`). `preprocessing.py` reads those artifacts independently. Changing cleaning logic does not require re-running the expensive LlamaParse API call.

### 2. Human-in-the-Loop Data Audit
After parsing, a manual audit of `tables.json` found:
- **Page 11** — A UI diagram misclassified as a data table. Added to exclusion list.
- **Page 12** — Merged cells with missing row headers. Fixed with a forward-fill rule.

### 3. Dual Table Representation (V1 Redundancy)
Markdown tables lack surrounding prose, which hurts semantic search scores. The pipeline retains flattened table data inside `clean_text.txt` where surrounding paragraph context provides the semantic anchor for retrieval. V2 will replace this with LLM-generated table summaries.

### 4. Factory Pattern — LLM & Embedding Layers
Both `llm.py` and `embedding.py` use a Factory Pattern. The LLM provider is swapped by changing a single string argument:
```python
llm = get_llm("openai")      # GPT-4o-mini
llm = get_llm("anthropic")   # Claude 3 Haiku
llm = get_llm("google")      # Gemini 1.5 Flash
```
Default embedding uses HuggingFace `all-MiniLM-L6-v2` — no API key needed to run.

### 5. Graceful Degradation
If the LLM fails to load (missing key, package error, or network issue), the FastAPI server does not crash. It degrades to serving raw retrieved context with a warning banner. Users always get a response; developers see the full exception in the server terminal only.

### 6. Layered Error Handling
Inner service layers (`llm.py`, `embedding.py`) throw specific technical exceptions. The orchestration layer (`api.py`) catches them at the boundary and makes the system-level decision — no raw stack traces ever reach the end user.

---

## ⚠️ Known V1 Limitations

| Query | Result |
|---|---|
| `"calibration settings"` | ✅ Returns correct data |
| `"50 mA 100 kVp"` | ✅ Returns matching row |
| `"Is 50 mA correct at 80 kVp?"` | ❌ Cannot find answer |
| `"Exposure for medium current?"` | ❌ Cannot find answer |

**Root cause:** The `all-MiniLM-L6-v2` model cannot bridge the semantic gap between a conversational query and a sparse numerical table row. Full analysis in [`DEV_LOG.md`](./DEV_LOG.md).

**V2 Roadmap:**
- [ ] Upgrade to `BAAI/bge-large-en-v1.5` or OpenAI `text-embedding-3-large`
- [ ] Hybrid Search (BM25 + Semantic)
- [ ] LLM Query Rewriting before retrieval
- [ ] Table Summarization (Metadata Enrichment)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| PDF Parsing | LlamaParse (Agentic Tier) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embedding | HuggingFace `all-MiniLM-L6-v2` (local, no API key needed) |
| Vector Store | ChromaDB (local) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| LLM Generation | OpenAI GPT-4o-mini (swappable via Factory Pattern) |

---

## 🚀 Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/sagarmishra0112/X-Ray-manual-RAG.git
cd X-Ray-manual-RAG
python -m venv ragvenv
.\ragvenv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add environment variables (.env file)
# OPENAI_API_KEY=sk-xxx   ← optional, app works without it
# llamacloud_key=llx-xxx  ← only needed to re-run ingestion

# 4. Start the backend (Terminal 1)
python -m uvicorn src.api:app --reload

# 5. Start the frontend (Terminal 2)
streamlit run src/ui.py
```

> **Note:** Intermediate pipeline artifacts (`data/parsed_output/`) are included in the repo. The embedding and retrieval steps can be run immediately without a LlamaParse key. The source PDF is publicly available — download it and place it at `data/Allengers_100.pdf` to re-run ingestion from scratch.

| Interface | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| Swagger API Docs | http://127.0.0.1:8000/docs |

---

## 📁 Project Structure

```
X-Ray-manual-RAG/
├── src/
│   ├── ingestion.py       # LlamaParse PDF parsing
│   ├── preprocessing.py   # Cleaning, heuristics, markdown conversion
│   ├── chunking.py        # Text & table chunking
│   ├── embedding.py       # Embedding + ChromaDB (Factory Pattern)
│   ├── llm.py             # LLM generation (Factory Pattern — 4 providers)
│   ├── api.py             # FastAPI orchestration layer
│   ├── ui.py              # Streamlit chat frontend
│   └── schemas.py         # Pydantic request/response schemas
├── data/parsed_output/    # Intermediate pipeline artifacts
├── DEV_LOG.md             # Engineering decision log
└── README.md
```

---

*V1 Baseline — April 2026*
