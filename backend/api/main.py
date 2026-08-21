import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl

from backend.core.config import get_settings
from backend.ingestion.document_loader import DocumentService
from backend.retrieval.vector_store import VectorStoreService
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.retriever import RerankedRetriever
from backend.chains.rag_chain import ask_question

app = FastAPI(title="Single-Source RAG Assistant", version="1.1.0")


class QuestionRequest(BaseModel):
    question: str


class UrlIngestRequest(BaseModel):
    url: HttpUrl


def _uploads_dir() -> Path:
    settings = get_settings()
    path = settings.vectorstore_dir / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path() -> Path:
    return _uploads_dir() / "manifest.json"


def _read_manifest() -> list[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_manifest(items: list[dict]) -> None:
    _manifest_path().write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _upsert_manifest(items: list[dict]) -> None:
    current = {item["filename"]: item for item in _read_manifest()}
    for item in items:
        current[item["filename"]] = item
    _write_manifest(list(current.values()))


def _index_all_uploaded_pdfs() -> dict:
    settings = get_settings()
    upload_dir = _uploads_dir()
    pdf_paths = sorted(upload_dir.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(400, "No PDF files found to index.")

    service = DocumentService()
    all_chunks = []
    manifest_items = []

    for pdf_path in pdf_paths:
        chunks = service.process_pdf(str(pdf_path))
        if not chunks:
            continue
        all_chunks.extend(chunks)
        pages = sorted(
            {
                int(c.metadata.get("page", 0))
                for c in chunks
                if str(c.metadata.get("page", "")).isdigit()
            }
        )
        manifest_items.append(
            {
                "filename": pdf_path.name,
                "pages": len(pages) if pages else 0,
                "chunks": len(chunks),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not all_chunks:
        raise HTTPException(400, "No text could be extracted from the uploaded PDFs.")

    store = VectorStoreService(settings.vectorstore_dir / "faiss")
    store.build(all_chunks)
    store.save()
    _write_manifest(manifest_items)

    return {
        "documents": manifest_items,
        "total_documents": len(manifest_items),
        "total_chunks": len(all_chunks),
        "index_dir": str(store.index_dir),
    }


def _safe_pdf_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name or "document.pdf"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    items = _read_manifest()
    existing = {p.name for p in _uploads_dir().glob("*.pdf")}
    items = [item for item in items if item.get("filename") in existing]
    return {"documents": items, "total_documents": len(items)}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Backward-compatible single-file ingestion endpoint.

    The uploaded file is added to the document collection and the combined
    index is rebuilt, so existing documents remain searchable.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    pdf_path = _uploads_dir() / _safe_pdf_filename(file.filename)
    with pdf_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return _index_all_uploaded_pdfs()


@app.post("/ingest-many")
async def ingest_many(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "Upload at least one PDF.")
    saved = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                400, f"Only PDF files are supported: {file.filename or 'unnamed file'}"
            )
        pdf_path = _uploads_dir() / _safe_pdf_filename(file.filename)
        with pdf_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        saved.append(pdf_path.name)
    result = _index_all_uploaded_pdfs()
    result["uploaded"] = saved
    return result


@app.post("/ingest-url")
def ingest_url(request: UrlIngestRequest):
    parsed = urlparse(str(request.url))
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(400, "URL must use http or https.")

    try:
        response = requests.get(
            str(request.url), timeout=45, headers={"User-Agent": "SingleSourceRAG/1.0"}
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(400, f"Could not download PDF: {exc}")

    content_type = response.headers.get("content-type", "").lower()
    path_name = Path(parsed.path).name or "document.pdf"
    filename = _safe_pdf_filename(path_name)
    if "pdf" not in content_type and not str(request.url).lower().split("?", 1)[
        0
    ].endswith(".pdf"):
        raise HTTPException(400, "The URL does not appear to point to a PDF file.")

    pdf_path = _uploads_dir() / filename
    pdf_path.write_bytes(response.content)
    result = _index_all_uploaded_pdfs()
    result["uploaded"] = [filename]
    result["source_url"] = str(request.url)
    return result


@app.post("/query")
def query(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        settings = get_settings()
        store = VectorStoreService(settings.vectorstore_dir / "faiss")
        store.load()
        reranked = RerankedRetriever(store, CrossEncoderReranker())
        response = ask_question(request.question, reranked)
        return response.model_dump()
    except FileNotFoundError:
        raise HTTPException(400, "No indexed PDF found. Upload a PDF first.")
    except Exception as exc:
        raise HTTPException(500, f"Query failed: {exc}")
