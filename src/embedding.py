import os
import argparse
from dotenv import load_dotenv

# We can import our chunking functions to get the fresh data directly
# This saves us from having to parse the txt file back into a list!
from src import chunking

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None  # Will handle this gracefully if not installed

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "..", "data", "chroma_db")

def get_embedding_model(model_choice):
    """Factory function to swap out the underlying embedding model."""
    if model_choice == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("Please 'pip install langchain-openai' to use OpenAI embeddings.")
        print("🤖 Initializing OpenAI Embeddings (text-embedding-3-small)...")
        # Ensure OPENAI_API_KEY is in your .env file
        return OpenAIEmbeddings(model="text-embedding-3-small")
        
    elif model_choice == "huggingface":
        print("🤗 Initializing local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        # This will download the model to your local machine the first time it runs
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding model: {model_choice}")

def get_vector_store(store_choice, embeddings_model):
    """Factory function to swap out the underlying vector database."""
    if store_choice == "chroma":
        print(f"🗄️ Initializing local ChromaDB at {os.path.abspath(DB_DIR)}...")
        return Chroma(
            persist_directory=DB_DIR, 
            embedding_function=embeddings_model
        )
    elif store_choice == "pinecone":
        print("🌲 Pinecone Vector Store selected. (Placeholder for V2)")
        # To strictly switch to Pinecone, you would import PineconeVectorStore
        # and initialize it with your PINECONE_API_KEY here.
        raise NotImplementedError("Pinecone not yet fully configured. Use 'chroma' for V1.")
    else:
        raise ValueError(f"Unknown vector store: {store_choice}")

def main(args):
    print("🚀 Starting Embedding & Vector Storage Pipeline...")
    load_dotenv() # Load API keys from .env if needed
    
    # 1. Get the data pieces using our previous module
    print("\n--- 1. Fetching Chunks ---")
    text_chunks = chunking.chunk_text()
    table_chunks = chunking.chunk_tables()
    
    # LangChain databases expect "Document" objects, not raw strings
    # So we wrap our strings and add some helpful metadata
    documents = []
    
    for i, t in enumerate(text_chunks):
        doc = Document(page_content=t, metadata={"source": "unstructured_text", "chunk_index": i})
        documents.append(doc)
        
    for i, t in enumerate(table_chunks):
        doc = Document(page_content=t, metadata={"source": "markdown_table", "chunk_index": i})
        documents.append(doc)
        
    print(f"✅ Packaged {len(documents)} total chunks into Document objects.")
    
    # 2. Initialize Models and Storage based on User Choice
    print(f"\n--- 2. Connecting to External Services ---")
    embeddings = get_embedding_model(args.model)
    vector_store = get_vector_store(args.store, embeddings)
    
    # 3. Embed and Store
    print(f"\n--- 3. Pushing to Vector Database ---")
    print("Working... (If using local HF, this might take a moment to compute embeddings)")
    
    # Reset/clear previous db (optional, good for dev)
    if args.store == "chroma" and os.path.exists(DB_DIR):
        print("⚠️  Warning: Appending to existing Chroma database.")
        
    vector_store.add_documents(documents)
    print("✅ All documents successfully embedded and stored!")
    print(f"🎯 Ready for Contextual Retrieval.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks into a Vector DB.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["huggingface", "openai"], 
        default="huggingface",
        help="Which AI model to use to create the number vectors."
    )
    parser.add_argument(
        "--store", 
        type=str, 
        choices=["chroma", "pinecone"], 
        default="chroma",
        help="Where to save the vectors."
    )
    
    args = parser.parse_args()
    main(args)
