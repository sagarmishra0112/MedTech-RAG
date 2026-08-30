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
        description="Self-critique score 1–5. 1=hallucinated, 5=excellent."
    )