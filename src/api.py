import os
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from src.embedding import get_embedding_model, get_vector_store
from src.llm import get_llm, generate_answer
from dotenv import load_dotenv


from src.schemas import QueryRequest, QueryResponse
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
    results = vector_store.similarity_search(request.question, k=request.top_k)
    
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

 