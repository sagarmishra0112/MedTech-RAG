import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.embedding import get_embedding_model, get_vector_store
from src.llm import get_llm, generate_answer, rewrite_query
from src.agent import run_agent
from dotenv import load_dotenv

from src.schemas import QueryRequest, QueryResponse, AgentQueryRequest, AgentQueryResponse
from langsmith import traceable

# Global variable to hold our database in memory
vector_store = None
llm = None

# --- NEW: Lifespan Manager ---
# This runs exactly once BEFORE the server starts taking requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, llm
    load_dotenv()
    print("🚀 Loading AI Models & ChromaDB...")
    embedding_choice = os.getenv("EMBEDDING_MODEL", "huggingface")
    embeddings = get_embedding_model(embedding_choice)
    vector_store = get_vector_store("chroma", embeddings)
    
    # Try to load the LLM. 
    # Change "openai" below to "google", "anthropic", or "local" to change providers!
    try:
        llm = get_llm("openai")
    except Exception as e:
        print(f"⚠️ Generation Model Offline. (No API key or package missing). Error: {e}")
        llm = None
        
    print("✅ System loaded and ready!")
    yield # Server runs here 
    print("🛑 Shutting down server...")

# 1. Create the 'app' instance, using the lifespan
app = FastAPI(title="MedTech RAG API", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "MedTech RAG is alive!"}

@app.get("/status")
def get_status():
    return {
        "status": "online",
        "pipeline": "chromadb_connected",
    }

# 2. Add the Query Endpoint
@app.post("/query", response_model=QueryResponse)
@traceable(name="MedTech RAG Query Endpoint")
def query_rag(request: QueryRequest):
    # This is the Logic Layer! We use our loaded vector_store
    print(f"Searching for: {request.question}")
    
    # Run the actual similarity search on Chroma
    # Phase 1: rewrite the query first for better retrieval (silently skips if no LLM)
    search_query = rewrite_query(llm, request.question) if llm else request.question
    results = vector_store.similarity_search(search_query, k=request.top_k)
    
    # Format the results for the user
    # Combine all matched texts into one context string
    combined_context = "\n\n".join([doc.page_content for doc in results])
    
    # --- NEW: The Generation Phase ---
    if llm:
        print("🧠 Synthesizing final answer using LLM...")
        final_answer = generate_answer(llm, request.question, combined_context)
    else:
        print("📥 LLM Offline. Returning raw retrieved context instead.")
        final_answer = "⚠️ [Generation Model Offline - Displaying Raw Extracted Context]:\n\n" + combined_context
    
    # Format human-readable source locations using chunk metadata
    sources = []
    for doc in results:
        meta = doc.metadata
        source_type = meta.get("source", "Unknown")
        if source_type == "unstructured_text":
            headers = [meta.get(h) for h in ["Header 1", "Header 2", "Header 3"] if meta.get(h)]
            header_str = " > ".join(headers) if headers else "Unstructured Text"
            page = meta.get("page")
            sources.append(f"{header_str} (Page {page})" if page is not None else header_str)
        elif source_type == "markdown_table":
            page = meta.get("page")
            sources.append(f"Table (Page {page})" if page is not None else "Table")
        else:
            sources.append(source_type)
            
    # Return it! FastAPI will check this against QueryResponse to strip leaks
    return {
        "answer": final_answer,
        "sources": sources
    }


# ── Phase 2: Agentic endpoint ──────────────────────────────────────────────

@app.post("/agent-query", response_model=AgentQueryResponse)
@traceable(name="MedTech Agentic RAG Endpoint")
def agent_query_rag(request: AgentQueryRequest):
    """
    Agentic RAG endpoint.
    Unlike /query (single retrieve→generate), this endpoint runs a
    ReAct agent loop that:
      1. Decides which tool to call (text search / table search / summary)
      2. Inspects the results, calls more tools if needed
      3. Synthesizes a grounded final answer
      4. Self-grades the answer and retries if the score is too low

    Response includes `steps_taken` (tool calls made) and
    `grade` (self-critique score 1–5).
    """
    if not llm:
        return AgentQueryResponse(
            answer="⚠️ Agent is offline — no LLM loaded. Check your API key in .env.",
            sources=[],
            steps_taken=0,
            grade=0,
        )

    print(f"\n🤖 Agentic query received: {request.question}")
    result = run_agent(
        question=request.question,
        vector_store=vector_store,
        llm=llm,
    )
    return AgentQueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        steps_taken=result["steps_taken"],
        grade=result["grade"],
    )
