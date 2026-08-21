from pathlib import Path
import numpy as np
import faiss
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from backend.retrieval.vector_store import VectorStoreService as _VectorStoreService


class VectorStoreService(_VectorStoreService):
    """Canonical FAISS service plus compatibility with the original .index format."""

    def __init__(self, index_path: str = "data/vectorstore/faiss.index"):
        self.legacy_path = (
            Path(index_path) if str(index_path).endswith(".index") else None
        )
        if self.legacy_path:
            super().__init__(index_dir=self.legacy_path.parent / self.legacy_path.stem)
            self.index_path = self.legacy_path
            self._legacy = True
            self._legacy_model = None
        else:
            super().__init__(index_dir=index_path)
            self._legacy = False
            self._legacy_model = None

    def _legacy_embeddings(self):
        if self._legacy_model is None:
            self._legacy_model = SentenceTransformer(self.model_name)
        return self._legacy_model

    def build_index(self, documents):
        return self.build(documents)

    def save_index(self):
        if self._legacy:
            # Save in canonical directory; future runs use the cleaner format.
            self.index_dir = self.legacy_path.parent / self.legacy_path.stem
            return self.save()
        return self.save()

    def load_index(self):
        if (
            self._legacy
            and self.legacy_path.exists()
            and self.legacy_path.with_suffix(".documents.npy").exists()
        ):
            self.index = faiss.read_index(str(self.legacy_path))
            data = np.load(
                self.legacy_path.with_suffix(".documents.npy"), allow_pickle=True
            )
            self.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in data
            ]
            return self.index
        return self.load()

    def query_index(self, question: str, k: int = 5):
        if self.index is None:
            self.load_index()
        if self._legacy and self.index_path.exists():
            emb = self._legacy_embeddings().encode(
                [question], normalize_embeddings=True, show_progress_bar=False
            )
            scores, indices = self.index.search(
                np.asarray(emb, dtype="float32"), min(k, self.index.ntotal)
            )
            return [
                {"document": self.documents[int(i)], "score": float(s)}
                for s, i in zip(scores[0], indices[0])
                if int(i) >= 0
            ]
        return [
            {"document": doc, "score": float(score)}
            for doc, score in self.similarity_search_with_score(question, k=k)
        ]

    def get_retriever(self, k: int = 10):
        return self.as_retriever(k=k)
