import os

# Base classes from LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(model_choice: str):
    """
    Factory function to initialize the Language Model for Generation.
    Supports OpenAI, Anthropic (Claude), Google (Gemini), and Local models.
    """
    if model_choice == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please 'pip install langchain-openai' to use OpenAI.")
        print("[LLM] Initializing OpenAI LLM (gpt-4o-mini)...")
        # Requires OPENAI_API_KEY in .env
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif model_choice == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please 'pip install langchain-anthropic' to use Claude.")
        print("[LLM] Initializing Anthropic LLM (claude-3-haiku)...")
        # Requires ANTHROPIC_API_KEY in .env
        return ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
        
    elif model_choice == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please 'pip install langchain-google-genai' to use Gemini.")
        print("[LLM] Initializing Google LLM (gemini-1.5-flash)...")
        # Requires GOOGLE_API_KEY in .env
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
    elif model_choice == "local":
        # ----------------------------------------------------------------
        # Points at ANY OpenAI-compatible server running on a rented GPU.
        # Spin up Ollama or vLLM on RunPod / Vast.ai / Lambda Labs,
        # then set these two env vars in your .env:
        #
        #   LOCAL_LLM_BASE_URL=http://<your-gpu-ip>:8000/v1
        #   LOCAL_LLM_MODEL=llama3.1:70b   (or qwen2.5:72b, etc.)
        #
        # vLLM exposes an OpenAI-compatible API, so ChatOpenAI works
        # with zero changes — just override the base_url.
        # ----------------------------------------------------------------
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please 'pip install langchain-openai' to use local vLLM.")
        base_url = os.getenv("LOCAL_LLM_BASE_URL")
        model_name = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
        if not base_url:
            raise ValueError(
                "LOCAL_LLM_BASE_URL is not set in your .env file.\n"
                "Set it to your rented GPU's vLLM/Ollama endpoint, e.g.:\n"
                "  LOCAL_LLM_BASE_URL=http://<gpu-ip>:8000/v1"
            )
        print(f"[LLM] Initializing Local LLM via vLLM/Ollama at {base_url} (model: {model_name})...")
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),  # vLLM may not need a real key
            temperature=0,
        )
        
    elif model_choice == "groq":
        # ----------------------------------------------------------------
        # Groq — free tier, OpenAI-compatible API, very fast inference.
        # Models: llama-3.1-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it
        # Get a free key at: https://console.groq.com
        # Add to .env:  GROQ_API_KEY=gsk_...
        # ----------------------------------------------------------------
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please 'pip install langchain-openai' to use Groq.")
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not set in your .env file.")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        print(f"[LLM] Initializing Groq LLM ({groq_model}) - free tier...")
        return ChatOpenAI(
            model=groq_model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown LLM model: {model_choice}")

@traceable(name="Rewrite Query")
def rewrite_query(llm: BaseChatModel, question: str) -> str:
    """
    Phase 1 — Query Rewriting.
    Rewrites a raw user question into a concise, retrieval-optimized search
    query before it hits ChromaDB.  This dramatically improves recall for
    vague, conversational, or multi-intent questions.

    Example:
        "what's the fuse thing on the back panel?"
        → "rear panel fuse rating and location specifications"
    """
    system_prompt = (
        "You are a search-query optimizer for a medical X-ray equipment service manual. "
        "Rewrite the user's question into a short, precise search query "
        "that will retrieve the most relevant technical information from a vector database. "
        "Output ONLY the rewritten query — no explanation, no punctuation at the end."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    try:
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        print(f"[Query] Rewritten: '{question}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        # Graceful fallback — use original question if rewriting fails
        print(f"⚠️ Query rewriting failed ({e}), using original question.")
        return question


@traceable(name="Generate RAG Answer")
def generate_answer(llm, question: str, context: str) -> str:

    """
    Takes the retrieved context and the user's question, and asks the LLM to generate a final answer.
    """
    system_prompt = (
        "You are an expert technical assistant for an enterprise medical equipment manufacturer. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not contained in the context, say 'I cannot find the answer in the provided documents.' "
        "Do not hallucinate.\n\n"
        "Context:\n"
        f"{context}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Generate the response  
    response = llm.invoke(messages)
    return response.content
    
