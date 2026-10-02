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
  5. (Phase 3) Claude grades the answer via structured JudgeVerdict.
              If score < 3, the verdict's flaw_type selects the retry
              strategy and Claude's own diagnosis is injected verbatim
              into the next GPT prompt.

Entry point for the API:
    result = run_agent(question, vector_store, llm)
    # result = {"answer": str, "sources": list[str],
    #           "steps_taken": int, "grade": int,
    #           "judge_critique": str}
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
from src.schemas import JudgeVerdict, FlawType


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
# 2. Phase 3: Structured Critique Grader
# ─────────────────────────────────────────────

# Neutral pass-through verdict used when grading fails or is unavailable.
_FALLBACK_VERDICT = JudgeVerdict(
    score=3,
    flaw_type=FlawType.IMPRECISE,
    flaw="Grader unavailable; defaulting to neutral pass.",
    missing="Unknown — grader did not respond.",
)

def grade_answer(
    generator_llm,
    question: str,
    answer: str,
    context_snippets: list[str],
    judge_llm=None,
) -> JudgeVerdict:
    """
    Grades the generated answer and returns a structured JudgeVerdict.

    Uses judge_llm (cross-provider) if provided; falls back to generator_llm
    with a self-grading warning if judge_llm is None.

    The verdict contains:
      - score      : 1–5 quality score
      - flaw_type  : FlawType enum — drives which retry strategy to apply
      - flaw        : ≤15-word diagnosis of why the answer failed
      - missing     : ≤15-word description of what information is absent

    Returns:
        JudgeVerdict. Falls back to _FALLBACK_VERDICT (score=3) on any error.
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
        "Evaluate the answer and return a structured verdict with these fields:\n"
        "  score      : integer 1–5\n"
        "    1 = Wrong or hallucinated — contradicts context or invents facts\n"
        "    2 = Partially correct — key information missing or imprecise\n"
        "    3 = Acceptable — answers the question but could be more specific\n"
        "    4 = Good — well-grounded in context, addresses all parts\n"
        "    5 = Excellent — precise, complete, fully evidenced by context\n"
        "  flaw_type  : one of [missing_info, hallucination, off_scope, imprecise]\n"
        "    missing_info  — answer is on-topic but lacks key facts\n"
        "    hallucination — answer states something not in the retrieved context\n"
        "    off_scope     — answer addresses things the question did not ask\n"
        "    imprecise     — answer is vague or approximate where exactness is needed\n"
        "  flaw        : one short phrase explaining the failure. MAX 15 WORDS.\n"
        "  missing     : one short phrase naming what information is absent. MAX 15 WORDS.\n\n"
        "Be concise. Never exceed 15 words in 'flaw' or 'missing'."
    )
    try:
        structured_grader = grader.with_structured_output(JudgeVerdict)
        verdict: JudgeVerdict = structured_grader.invoke([HumanMessage(content=grade_prompt)])
        verdict.score = max(1, min(5, verdict.score))  # clamp to valid range
        grader_label = "judge" if judge_llm else "self"
        print(f"[Grade] {grader_label}: {verdict.score}/5 | {verdict.flaw_type.value} | {verdict.flaw}")
        return verdict
    except Exception as e:
        print(f"[WARN] Grader failed ({e}), defaulting to neutral pass verdict.")
        return _FALLBACK_VERDICT


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
            "answer":              str  — final synthesized answer,
            "sources":             list[str] — human-readable source locations,
            "steps_taken":         int  — total tool calls across all attempts,
            "grade":               int  — final judge score (1–5),
            "correction_attempts": int  — number of retries triggered,
            "judge_provider":      str  — 'anthropic' | 'openai' | 'none',
            "confidence":          str  — 'high' | 'medium' | 'low',
            "judge_critique":      str  — formatted string of Claude's last rejection
                                         diagnosis, empty if answer passed first time.
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
    last_verdict: JudgeVerdict | None = None  # populated after each grading pass
    judge_critique = ""                       # surfaced in the API response

    # ── Retry strategy selection ───────────────────────────────────────────
    # Each flaw_type maps to a different correction strategy so that Claude's
    # diagnosis directly shapes what GPT does differently on the next attempt.
    def _build_query(attempt: int, verdict: JudgeVerdict | None) -> str:
        if attempt == 0 or verdict is None:
            return question

        flaw_type = verdict.flaw_type
        flaw_text = verdict.flaw
        missing_text = verdict.missing

        if flaw_type == FlawType.MISSING_INFO:
            # Rewrite the query to specifically target the missing information
            rewritten = rewrite_query(llm, f"{question} {missing_text}")
            return (
                f"{rewritten}\n\n"
                f"[JUDGE REJECTION — missing_info]\n"
                f"Previous answer was rejected. Flaw: {flaw_text}\n"
                f"You are specifically missing: {missing_text}\n"
                "Search both search_text_docs and search_table_docs to find this."
            )

        if flaw_type == FlawType.HALLUCINATION:
            # Do not invent — ground strictly in retrieved context only
            return (
                f"{question}\n\n"
                f"[JUDGE REJECTION — hallucination]\n"
                f"Previous answer was rejected. Flaw: {flaw_text}\n"
                "CRITICAL: Your previous answer contained information NOT in the retrieved context.\n"
                "You MUST only state facts that appear verbatim in the tool results.\n"
                "If the information is not found, say so explicitly. Do NOT infer or extrapolate."
            )

        if flaw_type == FlawType.OFF_SCOPE:
            # Narrow scope — answer only what was asked
            return (
                f"{question}\n\n"
                f"[JUDGE REJECTION — off_scope]\n"
                f"Previous answer was rejected. Flaw: {flaw_text}\n"
                "Answer ONLY the specific question above. Do not include related but unasked information."
            )

        # FlawType.IMPRECISE — search for exact values
        return (
            f"{question}\n\n"
            f"[JUDGE REJECTION — imprecise]\n"
            f"Previous answer was rejected. Flaw: {flaw_text}\n"
            f"Missing precision on: {missing_text}\n"
            "Search search_table_docs for exact numeric values, ratings, or specifications.\n"
            "Include exact values and page references in your answer."
        )

    for attempt in range(max_grade_retries + 1):
        query = _build_query(attempt, last_verdict)
        # Expand retrieval window on retries to cast a wider net
        top_k = 4 + (attempt * 2)   # 4 → 6 → 8

        print(f"\n{'='*60}")
        print(f"[Agent] Attempt {attempt + 1}/{max_grade_retries + 1}  (top_k={top_k})")
        print(f"   Query: {query[:120]}{'...' if len(query) > 120 else ''}")
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

        # ── Runtime self-correction: grade with structured judge verdict ──
        last_verdict = grade_answer(
            generator_llm=llm,
            question=question,
            answer=final_answer,
            context_snippets=context_snippets,
            judge_llm=judge_llm,
        )
        grade = last_verdict.score

        if grade >= 3:
            if judge_critique:
                # Print the intermediate critique to terminal so it's visible in server logs,
                # then clear it — the answer ultimately passed so the UI should not show it.
                print(f"   [RESOLVED] Previous rejection was corrected: {judge_critique}")
                judge_critique = ""
            print(f"   [PASS] Answer passed judge (score {grade}/5).")
            break
        else:
            correction_attempts += 1
            # Record Claude's diagnosis — printed to terminal, returned in API response
            # only if the answer never reaches a passing score.
            judge_critique = (
                f"[{last_verdict.flaw_type.value}] "
                f"flaw: {last_verdict.flaw} | "
                f"missing: {last_verdict.missing}"
            )
            print(f"[Retry] Score {grade}/5 — {judge_critique}")
            print(f"        Retrying (attempt {correction_attempts})...")

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
        "judge_critique": judge_critique,
    }
