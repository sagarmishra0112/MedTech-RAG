from pydantic import BaseModel, Field

# This is a Pydantic model. It's like a 'Contract' for your API.
class QueryRequest(BaseModel):
    # This says: 'The user MUST send a field called "question" which is a string'
    question: str = Field(..., example="What are the rules for CRAR?")
    
    # This says: 'They can optionally send "top_k", but if they don't, use 3'
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
