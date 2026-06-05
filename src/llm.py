import os

# Base classes from LangChain
from langchain_core.messages import HumanMessage, SystemMessage

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
        print("🤖 Initializing OpenAI LLM (gpt-4o-mini)...")
        # Requires OPENAI_API_KEY in .env
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    elif model_choice == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please 'pip install langchain-anthropic' to use Claude.")
        print("🟣 Initializing Anthropic LLM (claude-3-haiku)...")
        # Requires ANTHROPIC_API_KEY in .env
        return ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
        
    elif model_choice == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please 'pip install langchain-google-genai' to use Gemini.")
        print("🔵 Initializing Google LLM (gemini-1.5-flash)...")
        # Requires GOOGLE_API_KEY in .env
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
    elif model_choice == "local":
        print("🏠 Initializing Local LLM (Placeholder for Ollama/Llama.cpp)...")
        # For a truly local pipeline, you would use ChatOllama or HuggingFacePipeline here
        # E.g., return ChatOllama(model="llama3")
        raise NotImplementedError("Local LLM integration via Ollama not yet configured. Use 'openai' for V1.")
        
    else:
        raise ValueError(f"Unknown LLM model: {model_choice}")

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
    
