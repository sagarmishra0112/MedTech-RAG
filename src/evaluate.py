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

# Shared judge factory — same module used by api.py for runtime self-correction
from src.judge import build_judge_llm

# Load environment variables
load_dotenv(override=True)

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "eval", "golden_dataset.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval", "ragas_results.csv")


def main():
    print("🚀 Initializing Evaluation Pipeline...")

    # 1. Load the RAG System (Vector Store & Generator LLM)
    print("Loading RAG Components...")
    embedding_choice = os.getenv("EMBEDDING_MODEL", "huggingface")
    embeddings = get_embedding_model(embedding_choice)
    vector_store = get_vector_store("chroma", embeddings)

    try:
        # Generator LLM — answers the questions (stays as gpt-4o-mini)
        generator_llm = get_llm("openai")
    except Exception as e:
        print(f"❌ Failed to load Generator LLM. Cannot evaluate answers. {e}")
        return

    # 2. Build the Judge LLM (separate provider = unbiased grading)
    judge_llm = build_judge_llm()

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

    # 4. Generate Predictions  (Generator LLM does the answering)
    for idx, item in enumerate(golden_data):
        print(f"  [{idx+1}/{len(golden_data)}] {item['question']}")

        # A) Retrieve Context
        results = vector_store.similarity_search(item['question'], k=3)
        contexts = [doc.page_content for doc in results]
        combined_context = "\n\n".join(contexts)

        # B) Generate Answer
        generated_answer = generate_answer(generator_llm, item['question'], combined_context)

        # C) Store in Ragas format
        data_for_ragas["question"].append(item["question"])
        data_for_ragas["answer"].append(generated_answer)
        data_for_ragas["contexts"].append(contexts)   # Ragas wants a list of strings
        data_for_ragas["ground_truth"].append(item["ground_truth"])

    print("✔️  All answers generated!")

    # 5. Convert to Hugging Face Dataset format (required by Ragas)
    dataset = Dataset.from_dict(data_for_ragas)

    # 6. Run Ragas Evaluation — judge_llm grades, generator_llm generated
    print("\n⚖️  Running Ragas Evaluation with cross-provider Judge LLM...")
    print("   (This may take a few minutes)\n")

    metrics = [
        context_precision,   # Did we retrieve the right context at the top?
        context_recall,      # Did we retrieve all the necessary context?
        faithfulness,        # Is the answer faithful to the context (no hallucinations)?
        answer_relevancy,    # Does the answer actually address the question?
    ]

    try:
        # llm= here is the JUDGE — it overrides all internal metric LLM calls.
        # The generator was already used above; Ragas never touches it for grading.
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,       # ← Claude grades, GPT generates: true cross-provider eval
        )
        print("\n📊 --- Evaluation Results ---")
        print(evaluation_result)

        # 7. Save Results
        df = evaluation_result.to_pandas()
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        df.to_csv(RESULTS_PATH, index=False)
        print(f"\n💾 Detailed results saved to: {RESULTS_PATH}")

    except Exception as e:
        print(f"❌ Ragas evaluation failed: {e}")


if __name__ == "__main__":
    main()
