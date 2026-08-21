import re
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from backend.core.config import get_settings
from backend.core.llm import get_generation_llm
from backend.models.schemas import RAGResponse, SourceChunk
from backend.chains.retrieval import retrieve_for_query

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict single-document RAG assistant.
Answer ONLY from the supplied context.
Every factual claim must be supported by the context.
If the answer is absent, say exactly:
The answer is not in this document.
Do not fabricate citations or source IDs.""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context was retrieved."

    return "\n\n".join(
        f"[SOURCE {d.metadata.get('id', 'unknown')} | "
        f"PAGE {d.metadata.get('page', 'unknown')}]\n"
        f"{d.page_content}"
        for d in documents
    )


def build_sources(documents: list[Document]) -> list[SourceChunk]:
    sources = []

    for d in documents:
        metadata = d.metadata
        snippet = d.page_content.strip()

        if len(snippet) > 600:
            snippet = snippet[:600] + "..."

        sources.append(
            SourceChunk(
                id=str(metadata.get("id", "unknown")),
                page=metadata.get("page", "unknown"),
                source_file=str(metadata.get("source_file", "unknown")),
                snippet=snippet,
            )
        )

    return sources


def _extractive_fallback(
    question: str,
    documents: list[Document],
) -> RAGResponse:
    """Offline fallback used only when Gemini is unavailable."""
    if not documents:
        return RAGResponse(
            answer="The answer is not in this document.",
            sources=[],
        )

    stopwords = {
        "what",
        "which",
        "does",
        "this",
        "that",
        "from",
        "with",
        "about",
        "document",
        "paper",
        "where",
        "when",
        "were",
        "was",
        "how",
        "why",
        "and",
    }

    qwords = {
        w for w in re.findall(r"[a-zA-Z]{4,}", question.lower()) if w not in stopwords
    }

    scored = []

    for document in documents:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            document.page_content.strip(),
        )

        for sentence in sentences:
            words = set(re.findall(r"[a-zA-Z]{4,}", sentence.lower()))
            overlap = len(qwords & words)
            if overlap:
                scored.append((overlap, sentence))

    if not scored:
        return RAGResponse(
            answer="The answer is not in this document.",
            sources=[],
        )

    scored.sort(key=lambda item: item[0], reverse=True)

    return RAGResponse(
        answer=" ".join(sentence for _, sentence in scored[:3]),
        sources=build_sources(documents[:3]),
    )


def _response_text(content) -> str:
    """Normalize Gemini/LangChain text or content-block responses."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))

        return "".join(parts).strip()

    return str(content).strip()


def create_rag_chain(rag_retriever):
    """Canonical LCEL chain: route → retrieve/rerank → prompt → Gemini → sources."""

    def retrieve(question: str):
        documents, diagnostics = retrieve_for_query(
            question,
            rag_retriever,
            return_diagnostics=True,
        )

        return {
            "question": question,
            "documents": documents,
            "context": format_context(documents),
            "diagnostics": diagnostics,
        }

    def generate(value):
        settings = get_settings()

        if not (settings.google_api_key or settings.gemini_api_key):
            result = _extractive_fallback(
                value["question"],
                value["documents"],
            )
            result.retrieved_sources = build_sources(value["documents"])
            return result

        llm = get_generation_llm()

        # IMPORTANT: use ordinary text generation rather than
        # Gemini structured-output/function-calling. Sources are
        # constructed deterministically from retrieved documents.
        messages = RAG_PROMPT.format_messages(
            question=value["question"],
            context=value["context"],
        )

        response = llm.invoke(messages)
        answer = _response_text(response.content)

        unsupported = answer.strip().lower() == "the answer is not in this document."
        sources = [] if unsupported else build_sources(value["documents"])

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_sources=build_sources(value["documents"]),
        )

    return RunnableLambda(retrieve) | RunnableLambda(generate)


def ask_question(question: str, rag_retriever) -> RAGResponse:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    return create_rag_chain(rag_retriever).invoke(question)


def stream_question(question: str, retriever):
    """Stream a grounded answer and then emit deterministic sources."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    documents, _diagnostics = retrieve_for_query(
        question,
        retriever,
        return_diagnostics=True,
    )
    context = format_context(documents)

    settings = get_settings()

    if not settings.google_api_key:
        fallback = _extractive_fallback(question, documents)

        yield {
            "type": "token",
            "content": fallback.answer,
        }
    else:
        llm = get_generation_llm()
        messages = RAG_PROMPT.format_messages(
            question=question,
            context=context,
        )

        for chunk in llm.stream(messages):
            text = _response_text(chunk.content)
            if text:
                yield {
                    "type": "token",
                    "content": text,
                }

    yield {
        "type": "sources",
        "sources": build_sources(documents),
    }
