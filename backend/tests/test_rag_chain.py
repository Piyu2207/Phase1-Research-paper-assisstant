from langchain_core.documents import Document

from backend.chains.rag_chain import ask_question
from backend.retrieval import retriever


class FakeRetriever:
    """Fake retriever that avoids FAISS and CrossEncoder."""

    def __init__(self, documents):
        self.documents = documents

    def invoke_with_scores(self, question: str):
        return [
            {
                "document": document,
                "faiss_score": 1.0,
                "rerank_score": 1.0,
            }
            for document in self.documents
        ]


def fake_route_query(question: str):
    class FakeRoute:
        query_type = "single_fact"
        sub_questions = [question]

    return FakeRoute()


def test_answerable_question(monkeypatch):
    documents = [
        Document(
            page_content=(
                "Attention Is All You Need is a paper that "
                "introduces the Transformer architecture. "
                "The Transformer is based entirely on attention "
                "mechanisms and does not use recurrence."
            ),
            metadata={
                "id": "chunk_1",
                "page": 1,
                "source_file": "attention-is-all-you-need.pdf",
            },
        ),
        Document(
            page_content=(
                "The Transformer architecture allows models "
                "to process sequences efficiently using "
                "self-attention."
            ),
            metadata={
                "id": "chunk_2",
                "page": 2,
                "source_file": "attention-is-all-you-need.pdf",
            },
        ),
    ]

    monkeypatch.setattr(
        retriever,
        "route_query",
        fake_route_query,
    )

    response = ask_question(
        "What architecture is introduced in the document?",
        FakeRetriever(documents),
    )

    assert response.answer
    assert "Transformer" in response.answer
    assert len(response.sources) > 0

    source_ids = [source.id for source in response.sources]

    assert any(source_id in {"chunk_1", "chunk_2"} for source_id in source_ids)


def test_out_of_scope_question(monkeypatch):
    documents = [
        Document(
            page_content=(
                "Attention Is All You Need introduces " "the Transformer architecture."
            ),
            metadata={
                "id": "chunk_1",
                "page": 1,
                "source_file": "attention-is-all-you-need.pdf",
            },
        ),
    ]

    monkeypatch.setattr(
        retriever,
        "route_query",
        fake_route_query,
    )

    response = ask_question(
        "What is the capital of France?",
        FakeRetriever(documents),
    )

    assert response.answer
    assert "not in this document" in response.answer.lower()
    assert response.sources == []
