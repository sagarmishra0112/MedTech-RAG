from pydantic import BaseModel, Field

#Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the rules for CRAR?")
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
 