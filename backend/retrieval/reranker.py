from pathlib import Path
import json
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from backend.core.config import get_settings


class CrossEncoderReranker:
    """Second-stage semantic ranking over FAISS candidates."""

    def __init__(
        self, model_name: str | None = None, log_path: str | Path | None = None
    ):
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self.model = CrossEncoder(self.model_name)
        self.log_path = Path(
            log_path or settings.evaluation_dir / "reranking_comparisons.jsonl"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _meta(doc: Document) -> dict:
        return {
            "id": str(doc.metadata.get("id", "unknown")),
            "page": doc.metadata.get("page", "unknown"),
            "source_file": doc.metadata.get("source_file", "unknown"),
        }

    def rerank(
        self, query: str, scored_documents: list[tuple[Document, float]], top_k: int = 5
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not scored_documents:
            return []
        scores = self.model.predict(
            [(query, doc.page_content) for doc, _ in scored_documents]
        )
        ranked = [
            {
                "document": doc,
                "faiss_score": float(faiss_score),
                "rerank_score": float(score),
            }
            for (doc, faiss_score), score in zip(scored_documents, scores)
        ]
        before = [self._meta(doc) for doc, _ in scored_documents]
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        after = [self._meta(item["document"]) for item in ranked]
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"query": query, "before": before, "after": after},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return ranked[:top_k]
