"""Canonical query routing and decomposition for the single-source RAG pipeline.

The router is intentionally deterministic so routing never depends on an LLM call.
This keeps the retrieval path reliable while still demonstrating routing/decomposition.
"""

from typing import Literal
from pydantic import BaseModel, Field


class QueryRoute(BaseModel):
    query_type: Literal["single_fact", "multi_part", "summarization"]
    sub_questions: list[str] = Field(min_length=1, max_length=8)


def _heuristic_route(question: str) -> QueryRoute:
    q = question.strip()
    lower = q.lower()

    if any(
        term in lower
        for term in (
            "summarize",
            "summarise",
            "summary",
            "overview",
            "main contributions",
            "overall",
        )
    ):
        return QueryRoute(
            query_type="summarization",
            sub_questions=[q],
        )

    # Split common multi-part/comparison questions into independent
    # retrieval-ready questions.
    markers = (
        " and how ",
        " and what ",
        " and why ",
        " and where ",
        " and when ",
    )
    for marker in markers:
        if marker in lower:
            idx = lower.index(marker)
            left = q[:idx].strip().rstrip(" ?")
            right = q[idx + len(marker) :].strip().rstrip(" ?")
            if left and right:
                # Preserve the question intent in each retrieval query.
                if right.lower().startswith(("does ", "is ", "are ", "was ", "were ")):
                    right = right[0].upper() + right[1:]
                else:
                    right = right[0].upper() + right[1:]
                return QueryRoute(
                    query_type="multi_part",
                    sub_questions=[
                        left + "?",
                        right + "?",
                    ],
                )

    if "compare" in lower or "difference between" in lower or "versus" in lower:
        return QueryRoute(
            query_type="multi_part",
            sub_questions=[q],
        )

    return QueryRoute(
        query_type="single_fact",
        sub_questions=[q],
    )


def route_query(question: str) -> QueryRoute:
    """Classify/decompose a query without making the RAG path depend on Gemini."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    return _heuristic_route(question)


__all__ = ["QueryRoute", "route_query"]
