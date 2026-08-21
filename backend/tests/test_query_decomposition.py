from langchain_core.documents import Document

from backend.chains.retrieval import retrieve_for_query


class TrackingRetriever:
    """
    Fake retriever that records every query it receives.
    """

    def __init__(self):
        self.queries = []

    def invoke(self, query: str):

        self.queries.append(query)

        if "Transformer architecture" in query:
            return [
                Document(
                    page_content=(
                        "The Transformer architecture "
                        "uses self-attention mechanisms."
                    ),
                    metadata={
                        "id": "transformer_chunk_1",
                        "page": 3,
                    },
                )
            ]

        if "recurrent neural networks" in query:
            return [
                Document(
                    page_content=(
                        "Recurrent neural networks process "
                        "sequences using recurrent connections."
                    ),
                    metadata={
                        "id": "rnn_chunk_1",
                        "page": 4,
                    },
                )
            ]

        return []


def test_multi_part_query_retrieves_both_parts():

    retriever = TrackingRetriever()

    question = (
        "What is the Transformer architecture "
        "and how does it compare with recurrent neural networks?"
    )

    documents = retrieve_for_query(
        question=question,
        retriever=retriever,
        k=5,
    )

    # -----------------------------------------------------
    # Verify two retrieval queries were generated
    # -----------------------------------------------------

    assert len(retriever.queries) >= 2

    print("\nRETRIEVAL QUERIES:")

    for query in retriever.queries:
        print("-", query)

    # -----------------------------------------------------
    # Verify both concepts were retrieved
    # -----------------------------------------------------

    retrieved_ids = [document.metadata["id"] for document in documents]

    print("\nRETRIEVED CHUNKS:")

    for document in documents:
        print(
            document.metadata["id"],
            "page=",
            document.metadata["page"],
        )

    assert "transformer_chunk_1" in retrieved_ids

    assert "rnn_chunk_1" in retrieved_ids
