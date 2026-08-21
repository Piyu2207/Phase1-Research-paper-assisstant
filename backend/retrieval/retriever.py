from langchain_core.documents import Document
from backend.core.config import get_settings
from backend.retrieval.vector_store import VectorStoreService
from backend.retrieval.reranker import CrossEncoderReranker
from backend.chains.query_router import (
    route_query,
)  # compatibility export for tests/integrations


class RerankedRetriever:
    """FAISS top-N retrieval followed by cross-encoder top-K reranking."""

    def __init__(self, vector_store=None, reranker=None):
        settings = get_settings()
        self.vector_store = vector_store or VectorStoreService()
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = settings.retrieval_k
        self.top_k = settings.rerank_top_k

    def invoke_with_scores(self, query: str):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        candidates = self.vector_store.similarity_search_with_score(
            query, k=self.candidate_k
        )
        return self.reranker.rerank(query, candidates, top_k=self.top_k)

    def invoke(self, query: str) -> list[Document]:
        return [item["document"] for item in self.invoke_with_scores(query)]

    def retrieve_many(self, queries: list[str]) -> list[Document]:
        merged, seen = [], set()
        for query in queries:
            for document in self.invoke(query):
                key = str(document.metadata.get("id", document.page_content))
                if key not in seen:
                    seen.add(key)
                    merged.append(document)
        return merged
