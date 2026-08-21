from pathlib import Path

from backend.services.document_service import DocumentService

PDF_PATH = Path("data/papers/attention-is-all-you-need.pdf")


def test_document_service_process_pdf():
    service = DocumentService()

    chunks = service.process_pdf(str(PDF_PATH))

    # Verify chunks were generated
    assert chunks
    assert len(chunks) > 0

    # Verify first chunk contains text
    first_chunk = chunks[0]

    assert first_chunk.page_content.strip()

    # Verify required metadata
    assert first_chunk.metadata["source_file"]
    assert first_chunk.metadata["page"] is not None
    assert first_chunk.metadata["chunk_index"] is not None

    # Verify source file
    assert first_chunk.metadata["source_file"] == PDF_PATH.name

    # Verify chunk index starts from zero
    assert first_chunk.metadata["chunk_index"] == 0


def test_document_service_chunk_metadata():
    service = DocumentService()

    chunks = service.process_pdf(str(PDF_PATH))

    assert chunks

    for index, chunk in enumerate(chunks):
        # Every chunk must contain text
        assert chunk.page_content.strip()

        # Every chunk must have required metadata
        assert chunk.metadata["source_file"]
        assert chunk.metadata["page"] is not None
        assert chunk.metadata["chunk_index"] is not None

        # Chunk index should match its position
        assert chunk.metadata["chunk_index"] == index
