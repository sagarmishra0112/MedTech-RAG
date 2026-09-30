"""
src/evaluate_v3_agent.py — V3 Agentic RAG RAGAS Evaluation
=============================================================
Run the same 20-question golden dataset through the LangGraph ReAct agent
(run_agent) instead of classic single-shot RAG (generate_answer).

This gives us a direct V2 Classic RAG vs V3 Agent comparison on identical questions,
with the same Claude judge — so any score difference is purely from the agent architecture.

Usage:
    ragvenv\\Scripts\\python.exe -m src.evaluate_v3_agent

Output:
    data/eval/ragas_results_v3_agent.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Ensure UTF-8 output on Windows console to prevent charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# RAG system components
from src.embedding import get_embedding_model, get_vector_store
from src.llm import get_llm
from src.agent import run_agent
from src.judge import build_judge_llm, build_judge_llm_raw

# RAGAS
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

load_dotenv(override=True)
# Must be AFTER load_dotenv — otherwise .env LANGCHAIN_TRACING_V2=true overwrites it
os.environ["LANGCHAIN_TRACING_V2"] = "false"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "eval", "golden_dataset.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval", "ragas_results_v4_agent.csv")
LATENCY_SUMMARY_PATH = os.path.join(DATA_DIR, "eval", "v4_latency_summary.json")


def main():
    print("=" * 60)
    print("[V4] AGENT RAGAS EVALUATION + LATENCY BENCHMARK (CROSS-GRADER)")
    print(f"   Generator : OpenAI / gpt-4o-mini  (via run_agent)")
    print("   Judge     : Claude Haiku 4.5       (cross-provider, unbiased)")
    print("=" * 60)

    # ── 1. Load infrastructure ──────────────────────────────────────
    print("\nLoading vector store and models...")
    embedding_choice = os.getenv("EMBEDDING_MODEL", "openai")
    embeddings = get_embedding_model(embedding_choice)
    vector_store = get_vector_store("chroma", embeddings)

    generator_provider = os.getenv("GENERATOR_LLM", "openai")
    try:
        generator_llm = get_llm(generator_provider)
        print(f"[OK] Generator LLM loaded: {generator_provider}")
    except Exception as e:
        print(f"[ERROR] Could not load Generator LLM ({generator_provider}): {e}")
        return

    # Judge for RAGAS evaluation (wraps model in LangchainLLMWrapper)
    try:
        ragas_judge = build_judge_llm()
        print("[OK] Judge LLM loaded for RAGAS grading.")
    except Exception as e:
        print(f"[ERROR] Could not load Judge LLM: {e}")
        return

    # Raw judge for runtime self-correction inside run_agent()
    try:
        runtime_judge = build_judge_llm_raw()
        print("[OK] Runtime judge loaded for in-agent grading.")
    except Exception as e:
        print(f"[WARN] Runtime judge unavailable - agent will self-grade. {e}")
        runtime_judge = None

    # ── 2. Load golden dataset ──────────────────────────────────────
    print(f"\nLoading golden dataset from {GOLDEN_DATASET_PATH}...")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    print(f"   {len(golden_data)} questions loaded.\n")

    # ── 3. Run all questions through the V3 agent with latency timing
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    correction_log = []   # Track retries, latency, steps per question

    for idx, item in enumerate(golden_data):
        q = item["question"]
        print(f"\n[{idx+1:02d}/{len(golden_data)}] {q}")

        t0 = time.perf_counter()
        try:
            result = run_agent(
                question=q,
                vector_store=vector_store,
                llm=generator_llm,
                judge_llm=runtime_judge,
                max_grade_retries=2,
            )
            latency_s = round(time.perf_counter() - t0, 3)
        except Exception as e:
            latency_s = round(time.perf_counter() - t0, 3)
            print(f"   [WARN] Agent failed ({latency_s}s): {e} - skipping")
            continue

        answer = result["answer"]
        raw_snippets = result.get("context_snippets", [])
        contexts = raw_snippets if raw_snippets else (result["sources"] if result["sources"] else ["No context retrieved"])

        correction_log.append({
            "question": q,
            "grade": result["grade"],
            "correction_attempts": result["correction_attempts"],
            "confidence": result["confidence"],
            "steps_taken": result["steps_taken"],
            "latency_s": latency_s,
            "judge_provider": result["judge_provider"],
        })

        print(f"   Grade: {result['grade']}/5 | Retries: {result['correction_attempts']} | Latency: {latency_s:.2f}s | Confidence: {result['confidence']}")

        data_for_ragas["question"].append(q)
        data_for_ragas["answer"].append(answer)
        data_for_ragas["contexts"].append(contexts)
        data_for_ragas["ground_truth"].append(item["ground_truth"])

    print(f"\n[DONE] All {len(data_for_ragas['question'])} answers generated through V3 agent.")

    # ── 4. Self-correction & Latency Summary ────────────────────────
    total_retries = sum(r["correction_attempts"] for r in correction_log)
    questions_retried = sum(1 for r in correction_log if r["correction_attempts"] > 0)
    high_conf = sum(1 for r in correction_log if r["confidence"] == "high")
    low_conf  = sum(1 for r in correction_log if r["confidence"] == "low")
    latencies = [r["latency_s"] for r in correction_log]

    print("\n" + "=" * 60)
    print("RUNTIME SELF-CORRECTION SUMMARY")
    print("=" * 60)
    print(f"   Questions answered    : {len(correction_log)}")
    print(f"   Questions retried     : {questions_retried} ({round(100*questions_retried/max(len(correction_log),1))}%)")
    print(f"   Total retry events    : {total_retries}")
    print(f"   High confidence       : {high_conf}")
    print(f"   Low confidence        : {low_conf}")
    print("=" * 60)

    latency_metrics = {}
    if latencies:
        mean_lat = round(float(np.mean(latencies)), 2)
        median_lat = round(float(np.median(latencies)), 2)
        p90_lat = round(float(np.percentile(latencies, 90)), 2)
        p95_lat = round(float(np.percentile(latencies, 95)), 2)
        min_lat = round(float(min(latencies)), 2)
        max_lat = round(float(max(latencies)), 2)

        latency_metrics = {
            "total_questions": len(latencies),
            "mean_latency_s": mean_lat,
            "median_p50_latency_s": median_lat,
            "p90_latency_s": p90_lat,
            "p95_latency_s": p95_lat,
            "min_latency_s": min_lat,
            "max_latency_s": max_lat,
        }

        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK METRICS")
        print("=" * 60)
        print(f"   Mean Latency          : {mean_lat}s")
        print(f"   Median (P50) Latency  : {median_lat}s")
        print(f"   P90 Latency           : {p90_lat}s")
        print(f"   P95 Latency           : {p95_lat}s")
        print(f"   Min / Max Latency     : {min_lat}s / {max_lat}s")
        print("=" * 60)

        with open(LATENCY_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(latency_metrics, f, indent=2)
        print(f"Latency summary saved -> {LATENCY_SUMMARY_PATH}")

    # Save correction & latency log
    correction_df = pd.DataFrame(correction_log)
    correction_csv = os.path.join(DATA_DIR, "eval", "v4_correction_log.csv")
    correction_df.to_csv(correction_csv, index=False)
    print(f"Correction + Latency log saved -> {correction_csv}")

    # ── 5. RAGAS evaluation (Claude grades all 4 metrics) ──────────
    if not data_for_ragas["question"]:
        print("[ERROR] No answers to evaluate. Exiting.")
        return

    dataset = Dataset.from_dict(data_for_ragas)

    print("\nRunning RAGAS evaluation (Claude judging all 4 metrics)...")
    print("   This will take ~3-5 minutes for 20 questions.\n")

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    try:
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_judge,
        )

        print("\n--- V3 AGENT RAGAS RESULTS ---")
        print(evaluation_result)

        df = evaluation_result.to_pandas()
        if len(df) == len(latencies):
            df["latency_s"] = latencies

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        df.to_csv(RESULTS_PATH, index=False)
        print(f"\nRAGAS results saved -> {RESULTS_PATH}")

    except Exception as e:
        print(f"[ERROR] RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
