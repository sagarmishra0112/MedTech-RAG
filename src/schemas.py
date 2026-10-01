from enum import Enum
from pydantic import BaseModel, Field

# ── Existing schemas (unchanged — /query endpoint stays backward-compatible) ──

class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the rules for CRAR?")
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Phase 3: Structured Judge Verdict ──────────────────────────────────────

class FlawType(str, Enum):
    MISSING_INFO  = "missing_info"   # answer exists but key facts are absent → search harder
    HALLUCINATION = "hallucination"  # answer invents facts not in retrieved context → ground strictly
    OFF_SCOPE     = "off_scope"      # answer addresses things not asked → narrow the scope
    IMPRECISE     = "imprecise"      # answer is vague/approximate → search for exact values

class JudgeVerdict(BaseModel):
    """
    Structured output returned by the Claude judge after grading a GPT answer.
    All text fields are capped at 15 words via prompt instruction.
    Used to drive targeted retry behaviour in run_agent().
    """
    score: int = Field(
        description="Answer quality score 1–5. 1=hallucinated/wrong, 5=precise and complete."
    )
    flaw_type: FlawType = Field(
        description="Primary failure category. Drives which retry strategy is applied."
    )
    flaw: str = Field(
        description="Why the answer failed. One phrase, max 15 words."
    )
    missing: str = Field(
        description="What specific information is absent or needed. Max 15 words."
    )


# ── Phase 2: Agentic endpoint schemas ──────────────────────────────────────

class AgentQueryRequest(BaseModel):
    question: str = Field(..., example="What is the filament fuse rating and where is it located?")

class AgentQueryResponse(BaseModel):
    answer: str
    sources: list[str]
    steps_taken: int = Field(
        description="Number of tool calls the agent made before producing this answer."
    )
    grade: int = Field(
        description="Judge score 1–5. 1=hallucinated, 5=excellent."
    )
    correction_attempts: int = Field(
        default=0,
        description="Number of retries the agent made due to a low judge score."
    )
    judge_provider: str = Field(
        default="none",
        description="Which LLM graded the answer: 'anthropic', 'openai' (self-grading), or 'none'.",
    )
    confidence: str = Field(
        default="unknown",
        description="Human-readable confidence band derived from grade: 'high' (4-5), 'medium' (3), 'low' (1-2).",
    )
    judge_critique: str = Field(
        default="",
        description=(
            "Claude's structured diagnosis of the last rejected answer. "
            "Format: '[flaw_type] flaw: <text> | missing: <text>'. "
            "Empty if the answer passed on the first attempt."
        ),
    )