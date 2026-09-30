from pydantic import BaseModel, Field

# ── Existing schemas (unchanged — /query endpoint stays backward-compatible) ──

class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the rules for CRAR?")
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Phase 2: Agentic endpoint schemas ──

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
        description="Which LLM graded the answer: 'anthropic', 'openai' (self-grading), or 'none'."
    )
    confidence: str = Field(
        default="unknown",
        description="Human-readable confidence band derived from grade: 'high' (4-5), 'medium' (3), 'low' (1-2)."
    )