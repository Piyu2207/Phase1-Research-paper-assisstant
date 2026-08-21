"""Backward-compatible wrapper.

There is intentionally one RAG implementation: backend.chains.rag_chain.
"""

from backend.chains.rag_chain import (
    RAG_PROMPT,
    ask_question,
    build_sources,
    create_rag_chain,
    format_context,
    stream_question,
)

__all__ = [
    "RAG_PROMPT",
    "ask_question",
    "build_sources",
    "create_rag_chain",
    "format_context",
    "stream_question",
]
