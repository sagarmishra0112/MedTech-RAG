import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Import our RAG pieces
from src.embedding import get_embedding_model, get_vector_store
from src.llm import get_llm, generate_answer

# Import Ragas metrics
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# Load environment variables (OPENAI_API_KEY is required for Ragas evaluation natively)
load_dotenv(override=True)

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "eval", "golden_dataset.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval", "ragas_results.csv")

def main():
    print("🚀 Initializing Evaluation Pipeline...")
    
    # 1. Check for OpenAI Key (Ragas needs an LLM to act as the "Judge")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: Ragas requires an OPENAI_API_KEY to act as the evaluator/judge.")
        print("Please add it to your .env file before running this script.")
        # We can theoretically swap in local models for Ragas, but standard is OpenAI for benchmarking
    
    # 2. Load the RAG System (Vector Store & LLM)
    print("Loading RAG Components...")
    embeddings = get_embedding_model("huggingface")
    vector_store = get_vector_store("chroma", embeddings)
    
    try:
        # We need an LLM to generate the answers we're going to evaluate
        llm = get_llm("openai")
    except Exception as e:
        print(f"❌ Failed to load LLM for generation. Cannot evaluate answers. {e}")
        return

    # 3. Load the Golden Dataset
    print(f"Loading Golden Dataset from {GOLDEN_DATASET_PATH}...")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"Loaded {len(golden_data)} questions. Generating answers...")
    
    # Ragas expects a dataset with these exact column names:
    # question, answer (generated), contexts (retrieved), ground_truth
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # 4. Generate Predictions
    for idx, item in enumerate(golden_data):
        print(f"Processing question {idx+1}/{len(golden_data)}: {item['question']}")
        
        # A) Retrieve Context (Search Engine)
        results = vector_store.similarity_search(item['question'], k=3)
        contexts = [doc.page_content for doc in results]
        combined_context = "\n\n".join(contexts)
        
        # B) Generate Answer (LLM)
        generated_answer = generate_answer(llm, item['question'], combined_context)
        
        # C) Store in Ragas format
        data_for_ragas["question"].append(item["question"])
        data_for_ragas["answer"].append(generated_answer)
        data_for_ragas["contexts"].append(contexts)  # Ragas wants the list of strings
        data_for_ragas["ground_truth"].append(item["ground_truth"])
        
    print("✔️ All answers generated!")
    
    # 5. Convert to Hugging Face Dataset format (required by Ragas)
    dataset = Dataset.from_dict(data_for_ragas)
    
    # 6. Run Ragas Evaluation!
    print("⚖️ Running Ragas Evaluation (This may take a few minutes)...")
    metrics = [
        context_precision,  # Did we retrieve the right context at the top?
        context_recall,     # Did we retrieve all the necessary context?
        faithfulness,       # Is the answer faithful to the context (no hallucinations)?
        answer_relevancy,   # Does the answer actually address the question?
    ]
    
    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        print("\n📊 --- Evaluation Results ---")
        print(evaluation_result)
        
        # 7. Save Results
        df = evaluation_result.to_pandas()
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        df.to_csv(RESULTS_PATH, index=False)
        print(f"\n💾 Detailed results (with scores per question) saved to: {RESULTS_PATH}")
        
    except Exception as e:
        print(f"❌ Ragas evaluation failed: {e}")

if __name__ == "__main__":
    main()
