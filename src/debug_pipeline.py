import os
import sys
from dotenv import load_dotenv
from src.embedding import get_embedding_model, get_vector_store
from src.llm import get_llm, generate_answer

# Add root folder to sys.path so imports work smoothly if run from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_retrieval_only():
    """
    Tests only the VectorDB and Embedding layer. Bypasses the LLM completely.
    """
    print("\n" + "="*50)
    print("🛠️  RETRIEVAL-ONLY ISOLATION MODE")
    print("="*50)
    
    load_dotenv()
    print("[1/2] Loading Embedding Model...")
    embeddings = get_embedding_model("huggingface")
    
    print("[2/2] Connecting to ChromaDB...")
    vector_store = get_vector_store("chroma", embeddings)
    print("✅ Database Connected!\n")
    
    while True:
        question = input("Enter a test query (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        try:
            top_k = int(input("How many chunks do you want to retrieve? (e.g., 3): "))
        except ValueError:
            top_k = 3
            
        print(f"\n🔍 Searching vector store for: '{question}'...")
        results = vector_store.similarity_search(question, k=top_k)
        
        print("\n" + "-"*40)
        print(f"🎯 FOUND {len(results)} CHUNKS:")
        print("-"*40)
        
        for i, doc in enumerate(results, 1):
            print(f"\nCHUNK {i}:")
            print(f"Metadata Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content:\n{doc.page_content.strip()}")
            print("-" * 20)


def test_generation_only():
    """
    Tests only the LLM generation capabilities. Bypasses ChromaDB.
    """
    print("\n" + "="*50)
    print("🧠 GOLDEN CONTEXT (LLM ISOLATION) MODE")
    print("="*50)
    
    load_dotenv()
    print("[1/1] Loading LLM...")
    try:
        llm = get_llm("openai")
        print("✅ LLM Loaded Successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load LLM. Check your API keys. Error: {e}")
        return
        
    print("Testing the LLM's comprehension with a Perfect 'Golden' Context.\n")
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\nPaste your perfect 'Golden' context below.")
        print("When you are done pasting, hit ENTER on an empty line twice to submit it:")
        
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
            
        context = "\n".join(lines)
        
        if not context.strip():
            print("⚠️ No context provided. Try again.")
            continue
            
        print("\n🧠 Sending Question + Golden Context to LLM...")
        answer = generate_answer(llm, question, context)
        
        print("\n" + "-"*40)
        print("🤖 LLM OUTPUT:")
        print("-"*40)
        print(answer)
        print("\n" + "="*50 + "\n")


def main():
    print("\n" + "="*50)
    print("🚀 MedTech RAG Diagnostic & Debugging Tool")
    print("="*50)
    print("Which component do you want to test in isolation?")
    print("[1] Retrieval Phase (Test ChromaDB + Embeddings)")
    print("[2] Generation Phase (Test LLM with Golden Context)")
    print("[3] Exit")
    
    choice = input("\nEnter your choice (1, 2, or 3): ")
    
    if choice == '1':
        test_retrieval_only()
    elif choice == '2':
        test_generation_only()
    else:
        print("Exiting tool.")

if __name__ == "__main__":
    main()
