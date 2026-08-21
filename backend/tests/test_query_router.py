from backend.services.query_router import route_query


def test_single_fact_query():

    question = (
        "What architecture is introduced in " "the Attention Is All You Need paper?"
    )

    result = route_query(question)

    print("\nQUERY TYPE:")
    print(result.query_type)

    print("\nSUB-QUESTIONS:")
    print(result.sub_questions)

    assert result.query_type == "single_fact"

    assert len(result.sub_questions) >= 1


def test_multi_part_query():

    question = (
        "What is the Transformer architecture "
        "and how does it compare with recurrent neural networks?"
    )

    result = route_query(question)

    print("\nQUERY TYPE:")
    print(result.query_type)

    print("\nSUB-QUESTIONS:")

    for question in result.sub_questions:
        print(question)

    assert result.query_type == "multi_part"

    # We expect at least two retrieval questions.
    assert len(result.sub_questions) >= 2


def test_summarization_query():

    question = "Summarize the main contributions of the paper."

    result = route_query(question)

    print("\nQUERY TYPE:")
    print(result.query_type)

    print("\nSUB-QUESTIONS:")
    print(result.sub_questions)

    assert result.query_type == "summarization"

    assert len(result.sub_questions) >= 1
