from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import get_settings


class DocumentService:
    """PDF -> page documents -> citation-safe chunks."""

    def __init__(self):
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, pdf_path: str) -> list[Document]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")
        pages = PyPDFLoader(str(path)).load()
        if not pages:
            raise ValueError("No text could be extracted from the PDF.")
        for page_number, page in enumerate(pages, start=1):
            page.metadata["page"] = page_number
            page.metadata["source_file"] = path.name
        return pages

    def chunk_documents(self, pages: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(pages)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            # Keep IDs unique when multiple PDFs are indexed together.
            # The legacy chunk_index field remains unchanged for compatibility.
            source_stem = Path(
                str(chunk.metadata.get("source_file", "document.pdf"))
            ).stem
            safe_stem = (
                "".join(ch if ch.isalnum() else "_" for ch in source_stem).strip("_")
                or "document"
            )
            chunk.metadata["id"] = f"{safe_stem}::chunk_{index:04d}"
            chunk.metadata.setdefault("source_file", "unknown.pdf")
            chunk.metadata.setdefault("page", -1)
        return chunks

    def process_pdf(self, pdf_path: str) -> list[Document]:
        return self.chunk_documents(self.load_pdf(pdf_path))
