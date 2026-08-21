from typing import Literal
from pydantic import BaseModel, Field


class QueryRoute(BaseModel):
    query_type: Literal["single_fact", "multi_part", "summarization"]
    sub_questions: list[str] = Field(min_length=1, max_length=8)


class SourceChunk(BaseModel):
    id: str
    page: int | str
    source_file: str
    snippet: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    # Candidates returned by retrieval/reranking. These are NOT evidence when
    # the model says the answer is absent; the UI labels them accordingly.
    retrieved_sources: list[SourceChunk] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    document: object
    score: float
    query: str
