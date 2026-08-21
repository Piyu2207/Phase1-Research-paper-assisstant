from langchain_core.documents import Document
from backend.retrieval import retriever as retriever_module


def retrieve_for_query(
    question: str,
    retriever,
    k: int = 5,
    return_diagnostics: bool = False,
):
    """Route/decompose the query, retrieve and rerank each sub-question."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    route = retriever_module.route_query(question)
    documents: list[Document] = []
    seen: set[str] = set()

    diagnostics = {
        "query_type": route.query_type,
        "sub_questions": route.sub_questions,
        "retrieval": [],
    }

    for sub_question in route.sub_questions:
        if hasattr(retriever, "invoke_with_scores"):
            ranked = retriever.invoke_with_scores(sub_question)
            items = ranked[:k]
            docs = [item["document"] for item in items]

            diagnostics["retrieval"].append(
                {
                    "query": sub_question,
                    "results": [
                        {
                            "id": item["document"].metadata.get("id", "unknown"),
                            "page": item["document"].metadata.get("page", "unknown"),
                            "faiss_score": float(item.get("faiss_score", 0.0)),
                            "rerank_score": float(item.get("rerank_score", 0.0)),
                        }
                        for item in items
                    ],
                }
            )
        else:
            docs = retriever.invoke(sub_question)[:k]
            diagnostics["retrieval"].append(
                {
                    "query": sub_question,
                    "results": [
                        {
                            "id": d.metadata.get("id", "unknown"),
                            "page": d.metadata.get("page", "unknown"),
                            "faiss_score": 0.0,
                            "rerank_score": 0.0,
                        }
                        for d in docs
                    ],
                }
            )

        for doc in docs:
            key = str(doc.metadata.get("id", doc.page_content))
            if key not in seen:
                seen.add(key)
                documents.append(doc)

    return (documents, diagnostics) if return_diagnostics else documents
