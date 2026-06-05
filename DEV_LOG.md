
# FinanceRAG - Engineering Dev Log

This log tracks key architectural decisions, problems encountered, and solutions implemented. 
It serves as a high-level record of the project's evolution, separate from detailed code learning notes.

## 2026-05-29: V2 Phase 0.5 - Heuristic Noise Filtering
**Problem:** Garbage OCR and diagram noise (e.g., spatial layout artifacts) were polluting the vector space.
- **Decision:** Instead of a manual exclusion list for every single diagram page, implemented a simple heuristic filter in `preprocessing.py`.
- **Action:** Split the text into blocks and calculated the ratio of alphabetical characters to total characters. Blocks with `ratio < 0.4` and `length > 20` are classified as noise and discarded.
- **Result:** Automatically cleans up wiring diagrams, spatial lists, and random numbers without needing human-in-the-loop review for this personal project scale.

## 2026-02-21: Architecture Decision - Contextual Table Retrieval (V1 vs V2)
**Observation:** Markdown tables lack surrounding context. E.g., The Page 35 table contains calibration data (mA, KVP), but lacks the word "calibrate". A simple vector search for "How to calibrate" will likely fail to retrieve this table.
- **Decision:** For **V1**, we rely on the flattened table data residing inside `clean_text.txt` (which contains surrounding semantic context) as a fallback mechanism. We plan to solve that in V2 (Advanced RAG techniques)
- **V2 Roadmap:** Implement **Table Summarization (Metadata Enrichment)**. We will use an LLM to write a descriptive summary for every table, creating a highly searchable hybrid chunk.

## 2026-02-21: V1 Text & Table Preprocessing Completed
**Problem:** Raw `fulltext.txt` contained repeating headers ("Allengers 100..."), footers ("Page X"), and massive whitespace gaps, which would pollute RAG chunks. `tables.json` contained fake tables and missing merged-cell data.
- **Decision:** Built `preprocessing.py`.
- **Action:** 
    1. **Regex Cleaning:** Stripped headers, footers, and normalized whitespace in `fulltext.txt`.
    2. **Heuristic Cleaning:** Dropped Page 11 (fake table) and applied a forward-fill rule for Page 12 (merged cells).
    3. **Markdown Conversion:** Converted structured JSON tables into `processed_tables.md`.
- **Result:** Two clean artifacts ready for chunking.

## 2026-02-19: Preprocessing & Data Cleaning
**Problem:** False Positive Table Extraction
- **Observation:** LlamaParse identified a UI diagram on **Page 11** as a table.
- **Root Cause:** The diagram had grid-like visual structure (lines, aligned text) that confused the OCR/parser model.
- **Decision:** Implement a **Manual Exclusion List** in `preprocessing.py`.
- **Action:** 
    1. Audited `tables.json`.
    2. Identified Page 11 as a false positive.
    3. Plan to add `if table['page'] in [11]: continue` to the processing script.
- **Lesson:** Automated parsing of complex PDFs (medical manuals) is ~90% accurate. For a "Gold Standard" RAG dataset, a human-in-the-loop audit is necessary.

## 2026-02-18: Initial Architecture (Ingestion Phase)
**Problem:** Various issues during first script run (`ing.py`).
- **Issues & Fixes:**
    1. **Silent Failure:** Script ran but did nothing.
       - *Fix:* Added `asyncio.run(main())` entry point.
    2. **Duplicate Downloads:** Same image downloaded multiple times.
       - *Fix:* Added deduplication logic (`if filename in downloaded: continue`).
    3. **Hardcoded Paths:** Script failed when run from root dir.
       - *Fix:* Implemented `os.path.join(SCRIPT_DIR, ...)` logic.
    4. **Data Format:** Need structured data for tables, not just text.
       - *Decision:* Extracted tables as **JSON** (structured) instead of Markdown.

## 2026-02-18: Data Persistence Architecture
**Problem:** Data lost between pipeline steps.
- **Observation:** `ingestion.py` was printing results but not saving them in a usable format for `preprocessing.py`.
- **Decision:** Decoupled Ingestion and Preprocessing.
- **Action:** 
    - `ingestion.py` now saves intermediate artifacts: `fulltext.txt`, `tables.json`, `full_markdown.md`.
    - `preprocessing.py` loads these artifacts.
- **Benefit:** Allows restarting the pipeline at any stage without re-running expensive/slow PDF parsing.

## 2026-02-18: Environment Management
**Problem:** `ModuleNotFoundError: No module named 'httpx'`
- **Root Cause:** VS Code terminal default shell was not using the active virtual environment (`ragvenv`).
- **Action:** Enforced activation via `ragvenv\Scripts\Activate.ps1` before running scripts.
- **Lesson:** Always verify `sys.executable` or active venv before debugging import errors.

## 2026-03-09: Architecture Decision - Embedding Model and Vector Store
**Problem:** Need to embed our generated chunks into a vector space and store them for retrieval. The system needs to be flexible enough for both a free local fallback and a high-performance production environment.
- **Decision:** Implemented a Factory/Strategy pattern in `embedding.py` to support multiple Embedding Models (HuggingFace vs OpenAI) and Vector Stores (ChromaDB vs Pinecone). 
- **Action:** Created `embedding.py` with feature flags (`--model`, `--store`) to allow toggling between choices via CLI arguments. Defaulting to `huggingface` + `chroma` for V1.
- **Rejected Alternatives:** 
  - *Hardcoding OpenAI APIs globally:* Rejected because it forces recruiters or anyone cloning the repository to supply an API key and spend money just to test the code.
  - *Using Pinecone/Weaviate for V1 default:* Rejected because they require database provisioning and cloud credentials. ChromaDB runs completely locally as a pseudo-SQLite database, perfect for a self-contained prototype.
- **Result:** A robust, environment-agnostic embedding pipeline ready for retrieval.

## 2026-03-17: API and UI Development
**Problem:** The core RAG pipeline (Ingest -> Clean -> Chunk -> Embed) is complete. Now we need a way to serve these answers to the user and a clean interface to interact with it, avoiding heavy frameworks for a simple prototype.
- **Decision:** Built a backend API using **FastAPI** (`src/api.py`) and a frontend Chat UI using **Streamlit** (`src/ui.py`).
- **Action:**
    1. **FastAPI Lifespan:** Used `@asynccontextmanager` to load the ChromaDB vector store into memory *once* when the server starts, rather than loading it on every user question.
    2. **Pydantic Schemas:** Created `schemas.py` to strictly enforce what the API receives (`QueryRequest`) and returns (`QueryResponse`).
    3. **Streamlit UI:** Created a minimalistic 40-line `ui.py` that mimics ChatGPT. It connects to the FastAPI backend via the `requests` library.
- **Rejected Alternatives:**
    - *Gradio:* Rejected because Streamlit's `st.chat_message` components are much better suited for conversational AI interfaces out-of-the-box.
- **Issue encountered:** HuggingFace `sentence-transformers` threw a `TypeError: Pooling.__init__() missing 1 required positional argument` error during embedding.
    - *Root Cause:* Incompatibility between older `sentence-transformers` and newer `transformers` library versions.
        - *Fix:* Upgraded via `pip install -U sentence-transformers transformers`.
- **Result:** A fully functional, decoupled enterprise RAG application. The backend runs on port 8000 and the frontend on 8501.

## 2026-03-17: Architecture Decision - LLM Generation & Graceful Degradation
**Problem:** The pipeline successfully retrieved context (Search Engine), but lacked a Generative AI layer to synthesize that data into human-readable answers. 
- **Decision (Strategy Pattern):** We refused to hard-couple the project to OpenAI. Built an LLM factory in `src/llm.py` that allows hot-swapping between 4 different providers: OpenAI (GPT-4o), Anthropic (Claude 3), Google (Gemini 1.5), and Local LLMs (Ollama/HuggingFace).
- **Decision (Graceful Degradation):** What happens if the OpenAI API is down or the recruiter reviewing the code doesn't have an API key? Standard apps throw a `500 Internal Server Error`.
  - *Action:* Built a `try/except` block inside the FastAPI lifecycle manager. If the model fails to load or no API key is present, the API gracefully degrades. Instead of crashing, it bypasses the LLM and serves the raw chunked text to the user with a warning: `⚠️ [Generation Model Offline - Displaying Raw Extracted Context]`.
- **Why this matters for V1:** It prevents vendor lock-in, reduces costs during testing, and guarantees the application will never "crash" due to a third-party LLM service outage.

## 2026-03-22: Known Failure Mode — V1 Retrieval Semantic Gap

**Observation:** The V1 RAG pipeline performs well on direct, keyword-rich queries but fails on conversational, inferential, or paraphrased questions. This was discovered during live testing.

**Real Failure Examples (Observed on Allengers_100.pdf):**

| Query Type | Example Query | Result |
|---|---|---|
| ✅ Works | `"calibration"` | Returns correct overload/calibration settings |
| ✅ Works | `"50 mA 100 kVp"` | Returns matching table row |
| ❌ Fails | `"So 50 mA is correct at 80 kVp?"` | "I cannot find the answer in the provided documents." |
| ❌ Fails | `"Is the exposure setting right for medium current?"` | "I cannot find the answer in the provided documents." |

**Root Cause Analysis (5 Contributing Factors):**

1. **Weak Embedding Model:** `all-MiniLM-L6-v2` is a 22MB micro-model. It handles keyword proximity well but fails to bridge the semantic gap between a conversational question and a structured table row. The query `"Is 50 mA correct at 80 kVp?"` and the chunk `"50 | 80 | 95"` are scored as mathematically distant vectors even though they refer to the same fact.

2. **Tabular Data Stored as Raw Text:** Calibration tables are stored as flat, sparse numerical chunks (e.g., `"50 100 115 | 50 80 95"`). These chunks have almost no natural language for the embedding model to interpret semantically. Without surrounding prose, they are nearly invisible to a semantic search.

3. **`top_k=3` Too Restrictive:** Only the top 3 chunks are retrieved. If the correct chunk scores 4th or 5th in cosine similarity (which easily happens with a weak model), the LLM never sees the relevant data and correctly reports it cannot answer.

4. **No Query Rewriting / Expansion:** The raw user query goes directly into ChromaDB as-is. A single phrasing of a question may not match any stored chunk. In later versions we will implement query rewriting to rewrite the query into multiple search-friendly variants before retrieval, dramatically improving recall.

5. **Pure Semantic Search (No Keyword Fallback):** Exact numbers (`80`, `50 mA`) are not guaranteed to score high in semantic search. To overcome this we will implement Hybrid search (semantic + BM25 keyword) to ensure that exact numerical matches are always captured even when semantic similarity fails.


**V2 Roadmap — Planned Improvements:**

- [ ] **Upgrade Embedding Model:** Replace `all-MiniLM-L6-v2` with OpenAI `text-embedding-3-large` or `BAAI/bge-large-en-v1.5` for significantly better semantic understanding, especially on technical and numerical content.
- [ ] **Table Summarization (Metadata Enrichment):** Use an LLM during preprocessing to write a natural language summary for every table (e.g., *"This table shows mA/kVp calibration parameters and the corresponding correct mAs values."*). Inject this summary as the chunk text alongside the raw table data. This makes tables semantically searchable.
- [ ] **Hybrid Search (BM25 + Semantic):** Add BM25 keyword-based retrieval alongside ChromaDB's semantic search. Merge and re-rank results. This guarantees exact numbers and technical codes are always found even when semantic similarity scores are low.
- [ ] **Query Rewriting:** Before hitting ChromaDB, pass the user's question through a fast LLM call to rewrite it into 3 search-optimized variants. Run all 3 searches and deduplicate results. This dramatically improves recall for conversational and paraphrased queries.
- [ ] **Increase `top_k`:** Raise from `k=3` to `k=5` or `k=7` and add a relevance score threshold filter to discard low-confidence chunks while keeping more potentially relevant ones.
