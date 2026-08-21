from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.core.config import get_settings


class VectorStoreService:
    """Local sentence-transformer embeddings backed by persistent FAISS."""

    def __init__(self, index_dir: str | Path | None = None):
        settings = get_settings()

        self.index_dir = Path(index_dir or settings.vectorstore_dir / "faiss")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.store: FAISS | None = None

    def build(self, documents: list[Document]) -> FAISS:
        if not documents:
            raise ValueError("Cannot build an index from zero chunks.")

        self.store = FAISS.from_documents(
            documents,
            self.embeddings,
        )
        return self.store

    def save(self) -> None:
        if self.store is None:
            raise ValueError("Build or load the vector store first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.index_dir))

    def load(self) -> FAISS:
        index_file = self.index_dir / "index.faiss"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_dir}")

        self.store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        return self.store

    def as_retriever(self, k: int = 10):
        if self.store is None:
            raise ValueError("Vector store is not loaded.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
    ):
        if self.store is None:
            raise ValueError("Vector store is not loaded.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        return self.store.similarity_search_with_score(
            query,
            k=min(k, self.store.index.ntotal),
        )
