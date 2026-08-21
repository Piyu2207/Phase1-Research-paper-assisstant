from functools import lru_cache
from backend.retrieval.vector_store import VectorStoreService
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.retriever import RerankedRetriever
from backend.chains.rag_chain import create_rag_chain


@lru_cache
def get_retriever(index_dir: str | None = None):
    vector_store = VectorStoreService(index_dir=index_dir)
    vector_store.load()
    return RerankedRetriever(vector_store=vector_store, reranker=CrossEncoderReranker())


def get_rag_chain(index_dir: str | None = None):
    return create_rag_chain(get_retriever(index_dir))
