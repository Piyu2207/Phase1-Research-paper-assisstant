from backend.services.vector_store_service import VectorStoreService


def test_vector_store_retrieval():
    vector_store = VectorStoreService()
    vector_store.load_index()

    results = vector_store.query_index(
        "What is the Transformer architecture?",
        k=3,
    )

    assert len(results) == 3

    for result in results:
        assert "document" in result
        assert "score" in result

        document = result["document"]

        assert document.metadata["source_file"]
        assert document.metadata["page"] is not None
        assert document.metadata["chunk_index"] is not None
        assert document.page_content.strip()
