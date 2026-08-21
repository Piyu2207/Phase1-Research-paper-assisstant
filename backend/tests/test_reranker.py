from langchain_core.documents import Document

from backend.services.reranker import CrossEncoderReranker


def test_reranker_changes_ranking():

    query = "What is the Transformer architecture " "and how does self-attention work?"

    documents = [
        Document(
            page_content=(
                "This paper discusses recurrent neural "
                "networks and sequence modeling."
            ),
            metadata={
                "id": "chunk_rnn",
                "page": 4,
            },
        ),
        Document(
            page_content=(
                "The Transformer architecture is based "
                "entirely on attention mechanisms."
            ),
            metadata={
                "id": "chunk_transformer",
                "page": 3,
            },
        ),
        Document(
            page_content=(
                "The model uses positional encoding to "
                "represent the order of tokens."
            ),
            metadata={
                "id": "chunk_position",
                "page": 4,
            },
        ),
    ]

    reranker = CrossEncoderReranker()

    result = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
    )

    print("\nFINAL RANKING:")

    for rank, item in enumerate(
        result,
        start=1,
    ):
        document = item["document"]

        print(
            rank,
            document.metadata["id"],
            item["score"],
        )

    assert len(result) == 3

    # The Transformer chunk should receive a
    # strong relevance score for this query.
    assert result[0]["document"].metadata["id"] == "chunk_transformer"
