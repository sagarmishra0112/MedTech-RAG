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

def grade_answer(llm, question: str, answer: str, context_snippets: list[str]) -> int:
    """
    Asks the LLM to score its own answer on a 1–5 scale.

    Scoring rubric:
      1 — Answer is wrong, hallucinated, or completely off-topic.
      2 — Answer is partial; key information is missing.
      3 — Answer is acceptable but could be improved.
      4 — Answer is good; addresses the question with evidence.
      5 — Answer is complete, precise, fully grounded in context.

    Returns:
        int score 1–5. Falls back to 3 (pass) on any parsing error.
    """
    context_block = "\n\n".join(context_snippets[:4])  # cap to avoid huge prompt

    grade_prompt = (
        "You are a strict quality-control evaluator for a RAG system.\n\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CONTEXT (excerpts):\n{context_block}\n\n"
        f"ANSWER GENERATED:\n{answer}\n\n"
        "Score the answer on a scale of 1–5 using this rubric:\n"
        "  1 = Wrong or hallucinated\n"
        "  2 = Partially correct, key info missing\n"
        "  3 = Acceptable\n"
        "  4 = Good, well-grounded\n"
        "  5 = Excellent, precise, fully evidenced\n\n"
        "Output ONLY a single integer (1, 2, 3, 4, or 5). No explanation."
    )
    try:
        response = llm.invoke([HumanMessage(content=grade_prompt)])
        score_str = response.content.strip()
        score = int(score_str[0])  # take first char in case of extra text
        score = max(1, min(5, score))  # clamp to valid range
        print(f"📊 Answer graded: {score}/5")
        return score
    except Exception as e:
        print(f"⚠️ Grader failed ({e}), defaulting to score 3 (pass).")
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
    max_grade_retries: int = 2,
) -> dict[str, Any]:
    """
    Full agentic RAG pipeline (Phases 2 + 3).

    Args:
        question:          The user's original question.
        vector_store:      The already-loaded ChromaDB instance.
        llm:               The already-loaded LangChain LLM instance.
        max_grade_retries: How many times to retry if the answer grades poorly.

    Returns:
        {
            "answer":      str        — the final synthesized answer,
            "sources":     list[str]  — human-readable source locations,
            "steps_taken": int        — number of tool calls the agent made,
            "grade":       int        — self-critique score (1–5),
        }
    """
    tools = build_tools(vector_store)

    # LangGraph create_react_agent — the modern replacement for AgentExecutor
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

    for attempt in range(max_grade_retries + 1):
        query = question if attempt == 0 else f"{question} (provide more specific and detailed information)"
        print(f"\n{'='*60}")
        print(f"🤖 Agent attempt {attempt + 1}/{max_grade_retries + 1}")
        print(f"   Question: {query}")
        print(f"{'='*60}")

        # LangGraph agent returns a dict with a "messages" list
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        messages = result.get("messages", [])

        # Extract final answer — it's the last AIMessage content
        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        # Extract tool call observations and sources from message history
        collected_sources = []
        context_snippets = []
        steps_taken = 0

        for msg in messages:
            # ToolMessage holds the tool's return value
            if hasattr(msg, "content") and hasattr(msg, "name"):
                observation = str(msg.content)
                context_snippets.append(observation)
                steps_taken += 1
                # Parse [Source Location] headers embedded in tool output
                matches = re.findall(r"\[([^\]]+)\]", observation)
                for match in matches:
                    if any(kw in match for kw in ["Page", "Table", "General", "Text"]):
                        if match not in collected_sources:
                            collected_sources.append(match)

        if not collected_sources:
            collected_sources = ["Retrieved from service manual"]

        print(f"   Tool calls made: {steps_taken}")
        print(f"   Sources: {collected_sources}")

        # ── Phase 3: Grade the answer ──
        grade = grade_answer(llm, question, final_answer, context_snippets)

        if grade >= 3:
            print(f"✅ Answer passed grading (score {grade}/5). Done.")
            break
        else:
            print(f"🔄 Answer scored {grade}/5 — retrying with refined query...")

    return {
        "answer": final_answer,
        "sources": collected_sources,
        "steps_taken": steps_taken,
        "grade": grade,
    }
