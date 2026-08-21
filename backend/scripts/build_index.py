import argparse
from pathlib import Path
from backend.core.config import get_settings
from backend.ingestion.document_loader import DocumentService
from backend.retrieval.vector_store import VectorStoreService


def main():
    parser = argparse.ArgumentParser(
        description="Build a persistent FAISS index from one PDF."
    )
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    chunks = DocumentService().process_pdf(str(args.pdf))
    store = VectorStoreService()
    store.build(chunks)
    store.save()
    print(f"Indexed {len(chunks)} chunks from {args.pdf.name}.")
    print(f"FAISS index: {store.index_dir}")


if __name__ == "__main__":
    main()
