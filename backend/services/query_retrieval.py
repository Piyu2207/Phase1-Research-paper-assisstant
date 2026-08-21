from typing import List

from langchain_core.documents import Document

from backend.services.query_router import route_query


def retrieve_for_query(
    question: str,
    retriever,
    k: int = 5,
) -> List[Document]:
    """
    Route the query and retrieve evidence.

    For single-fact and summarization queries:
        Retrieve using the original query.

    For multi-part queries:
        Retrieve separately for every sub-question
        and merge the resulting documents.
    """

    route = route_query(question)

    print("\nQUERY TYPE:")
    print(route.query_type)

    print("\nSUB-QUESTIONS:")

    for sub_question in route.sub_questions:
        print(f"- {sub_question}")

    # -----------------------------------------------------
    # Retrieve for every sub-question
    # -----------------------------------------------------

    retrieved_documents = []

    for sub_question in route.sub_questions:

        documents = retriever.invoke(sub_question)

        retrieved_documents.extend(documents)

    # -----------------------------------------------------
    # Remove duplicate chunks
    # -----------------------------------------------------

    unique_documents = []
    seen_ids = set()

    for document in retrieved_documents:

        chunk_id = document.metadata.get(
            "id",
            document.page_content,
        )

        if chunk_id not in seen_ids:

            seen_ids.add(chunk_id)

            unique_documents.append(document)

    return unique_documents
