"""
src/agent.py — Phase 2 & 3: Agentic RAG Engine
================================================
Uses LangGraph's create_react_agent (the modern replacement for the
deprecated AgentExecutor) to run a tool-calling loop.

The agent:
  1. Reads the user question.
  2. Picks which tool to call (search_text_docs, search_table_docs,
     or get_document_summary).
  3. Observes the result, decides if it needs more info.
  4. Writes a final answer grounded only in retrieved context.
  5. (Phase 3) A grader scores the answer; retries if score < 3.

Entry point for the API:
    result = run_agent(question, vector_store, llm)
    # result = {"answer": str, "sources": list[str],
    #           "steps_taken": int, "grade": int}
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langsmith import traceable
from langgraph.prebuilt import create_react_agent
from src.llm import rewrite_query


# ─────────────────────────────────────────────
# 1. Tool factory
#    Closed over the live vector_store loaded by the API lifespan.
# ─────────────────────────────────────────────

def build_tools(vector_store):
    """
    Returns a list of LangChain tools that the agent can invoke.
    All tools access the vector_store via closure.
    """

    @tool
    def search_text_docs(query: str) -> str:
        """
        Search the X-ray service manual's unstructured text sections
        (installation steps, safety guidelines, system overview, wiring
        descriptions, troubleshooting notes, etc.).
        Use this when the question is about procedures, descriptions,
        or any non-tabular content.

        Args:
            query: A precise search phrase.

        Returns:
            The top matching text passages, with section headers.
        """
        results = vector_store.similarity_search(
            query,
            k=4,
            filter={"source": "unstructured_text"},
        )
        if not results:
            return "No relevant text sections found for this query."

        passages = []
        for doc in results:
            meta = doc.metadata
            headers = [meta.get(h) for h in ["Header 1", "Header 2", "Header 3"] if meta.get(h)]
            header_str = " > ".join(headers) if headers else "General Text"
            page = meta.get("page")
            location = f"{header_str} (Page {page})" if page is not None else header_str
            passages.append(f"[{location}]\n{doc.page_content}")

        return "\n\n---\n\n".join(passages)

    @tool
    def search_table_docs(query: str) -> str:
        """
        Search the X-ray service manual's structured data tables
        (fuse specifications, mAs settings, voltage ratings, exposure
        parameters, component lookup tables, etc.).
        Use this when the question involves numbers, specs, ratings,
        or any data that would live in a table.

        Args:
            query: A precise search phrase related to specifications or data.

        Returns:
            The top matching table chunks in Markdown format.
        """
        results = vector_store.similarity_search(
            query,
            k=3,
            filter={"source": "markdown_table"},
        )
        if not results:
            return "No relevant data tables found for this query."

        passages = []
        for doc in results:
            page = doc.metadata.get("page")
            location = f"Table — Page {page}" if page is not None else "Table"
            passages.append(f"[{location}]\n{doc.page_content}")

        return "\n\n---\n\n".join(passages)

    @tool
    def get_document_summary(_input: str = "") -> str:
        """
        Returns a high-level summary of what the Allengers 100 X-ray
        service manual covers. Use this first if you are unsure which
        section of the manual is relevant, or if the question is about
        the overall system/document.

        Args:
            _input: Not used. Pass an empty string or any value.

        Returns:
            A concise overview of the manual's contents.
        """
        return (
            "The Allengers 100 X-ray Service Manual covers a mobile medical "
            "X-ray unit (100mA capacity). Key sections include:\n"
            "• System Overview & Technical Specifications (kVp, mA, mAs ranges)\n"
            "• Installation & Safety Requirements (power supply, earthing, radiation)\n"
            "• Component Layout (control panel, collimator, X-ray tube, HT transformer)\n"
            "• Fuse & Wiring Specifications (fuse ratings, locations, circuit diagrams)\n"
            "• Exposure Parameter Tables (mA station, kVp, mAs setting tables)\n"
            "• Maintenance & Troubleshooting (common faults, replacement procedures)\n"
            "Use search_text_docs for prose sections and search_table_docs for specs/tables."
        )

    return [search_text_docs, search_table_docs, get_document_summary]


# ─────────────────────────────────────────────
# 2. Phase 3: Self-Critique Grader
# ─────────────────────────────────────────────

def grade_answer(
    generator_llm,
    question: str,
    answer: str,
    context_snippets: list[str],
    judge_llm=None,
) -> int:
    """
    Grades the generated answer on a 1–5 scale.

    Uses judge_llm (cross-provider) if provided; falls back to generator_llm
    with a self-grading warning if judge_llm is None.

    Scoring rubric:
      1 — Wrong, hallucinated, or completely off-topic.
      2 — Partial; key information is missing.
      3 — Acceptable but could be improved.
      4 — Good; addresses the question with evidence from context.
      5 — Complete, precise, fully grounded in retrieved context.

    Returns:
        int score 1–5. Falls back to 3 (neutral pass) on any parse error.
    """
    if judge_llm is None:
        print("⚠️  No judge_llm provided — falling back to self-grading (same model as generator).")
        grader = generator_llm
    else:
        grader = judge_llm

    context_block = "\n\n".join(context_snippets[:4])  # cap to avoid huge prompt

    grade_prompt = (
        "You are a strict, impartial quality-control evaluator for a RAG system "
        "that answers questions about a medical X-ray equipment service manual.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CONTEXT (what the system had access to):\n{context_block}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Score the answer on a scale of 1–5 using this rubric:\n"
        "  1 = Wrong or hallucinated — contradicts context or invents facts\n"
        "  2 = Partially correct — key information missing or imprecise\n"
        "  3 = Acceptable — answers the question but could be more specific\n"
        "  4 = Good — well-grounded in context, addresses all parts\n"
        "  5 = Excellent — precise, complete, fully evidenced by context\n\n"
        "Output ONLY a single integer (1, 2, 3, 4, or 5). No explanation."
    )
    try:
        response = grader.invoke([HumanMessage(content=grade_prompt)])
        score_str = response.content.strip()
        score = int(score_str[0])  # take first char in case of trailing text
        score = max(1, min(5, score))  # clamp to valid range
        grader_label = "judge" if judge_llm else "self"
        print(f"[Grade] Answer graded by {grader_label}: {score}/5")
        return score
    except Exception as e:
        print(f"[WARN] Grader failed ({e}), defaulting to score 3 (pass).")
        return 3


# ─────────────────────────────────────────────
# 3. Main Entry Point
# ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "You are an expert technical assistant for the Allengers 100 X-ray service manual. "
    "You have access to three tools to look up information:\n"
    "  • search_text_docs   — for procedures, descriptions, safety info\n"
    "  • search_table_docs  — for specs, ratings, fuse data, exposure tables\n"
    "  • get_document_summary — for a high-level orientation of the document\n\n"
    "Strategy:\n"
    "1. If the question involves numbers, specs, or ratings → call search_table_docs first.\n"
    "2. If the question is about procedures or descriptions → call search_text_docs first.\n"
    "3. If you are unsure what the question is about → call get_document_summary first.\n"
    "4. After each tool call, decide if you have enough info. If not, call another tool.\n"
    "5. Base your FINAL answer ONLY on the retrieved context. "
    "   If the information is not in the documents, say so clearly. "
    "   Do NOT hallucinate or invent specifications."
)


@traceable(name="Agentic RAG Run")
def run_agent(
    question: str,
    vector_store,
    llm,
    judge_llm=None,
    max_grade_retries: int = 2,
) -> dict[str, Any]:
    """
    Full agentic RAG pipeline with runtime self-correction.

    Args:
        question:          The user's original question.
        vector_store:      The already-loaded ChromaDB instance.
        llm:               Generator LLM (gpt-4o-mini) — answers the question.
        judge_llm:         Judge LLM (Claude) — grades the answer independently.
                           If None, falls back to self-grading via llm.
        max_grade_retries: Max retries when judge score is below threshold.

    Returns:
        {
            "answer":              str        — final synthesized answer,
            "sources":             list[str]  — human-readable source locations,
            "steps_taken":         int        — total tool calls across all attempts,
            "grade":               int        — final judge score (1–5),
            "correction_attempts": int        — number of retries triggered,
            "judge_provider":      str        — 'anthropic' | 'openai' | 'none',
            "confidence":          str        — 'high' | 'medium' | 'low',
        }
    """
    tools = build_tools(vector_store)

    # Determine judge provider label for metadata
    if judge_llm is not None:
        judge_provider = os.getenv("JUDGE_LLM", "anthropic").lower()
    else:
        judge_provider = "openai"  # self-grading fallback

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=AGENT_SYSTEM_PROMPT,
    )

    collected_sources: list[str] = []
    context_snippets: list[str] = []
    steps_taken = 0
    final_answer = ""
    grade = 5
    correction_attempts = 0

    # Retry escalation strategies:
    #   Attempt 0: original question, normal retrieval
    #   Attempt 1: rewritten query (better retrieval), expanded context window
    #   Attempt 2: explicit instruction for precision + both tool types
    def _build_query(attempt: int) -> str:
        if attempt == 0:
            return question
        if attempt == 1:
            rewritten = rewrite_query(llm, question)
            return rewritten
        # Final attempt: ask explicitly for technical precision
        return (
            f"{question} "
            "Please search both text sections and data tables. "
            "Include exact values, specifications, and page references."
        )

    for attempt in range(max_grade_retries + 1):
        query = _build_query(attempt)
        # Expand retrieval window on retries to cast a wider net
        top_k = 4 + (attempt * 2)   # 4 → 6 → 8

        print(f"\n{'='*60}")
        print(f"[Agent] Attempt {attempt + 1}/{max_grade_retries + 1}  (top_k={top_k})")
        print(f"   Query: {query}")
        print(f"{'='*60}")

        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])

        # Extract final answer — last AIMessage with non-empty content
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        # Extract tool observations and source metadata
        collected_sources = []
        context_snippets = []
        attempt_steps = 0

        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "name"):
                observation = str(msg.content)
                context_snippets.append(observation)
                attempt_steps += 1
                matches = re.findall(r"\[([^\]]+)\]", observation)
                for match in matches:
                    if any(kw in match for kw in ["Page", "Table", "General", "Text"]):
                        if match not in collected_sources:
                            collected_sources.append(match)

        steps_taken += attempt_steps

        if not collected_sources:
            collected_sources = ["Retrieved from service manual"]

        print(f"   Tool calls this attempt: {attempt_steps}")
        print(f"   Sources: {collected_sources}")

        # ── Runtime self-correction: grade with independent judge ──
        grade = grade_answer(
            generator_llm=llm,
            question=question,
            answer=final_answer,
            context_snippets=context_snippets,
            judge_llm=judge_llm,
        )

        if grade >= 3:
            print(f"   [PASS] Answer passed judge (score {grade}/5).")
            break
        else:
            correction_attempts += 1
            print(f"[Retry] Score {grade}/5 below threshold - retrying (attempt {correction_attempts})...")

    # Derive human-readable confidence band from final grade
    if grade >= 4:
        confidence = "high"
    elif grade == 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "answer": final_answer,
        "sources": collected_sources,
        "context_snippets": context_snippets,
        "steps_taken": steps_taken,
        "grade": grade,
        "correction_attempts": correction_attempts,
        "judge_provider": judge_provider,
        "confidence": confidence,
    }
