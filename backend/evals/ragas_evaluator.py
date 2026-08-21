# backend/evals/ragas_evaluator.py

from typing import List, Dict, Any

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
)

from ragas.llms import LangchainLLMWrapper

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ---------------------------------------------------------
# Evaluation LLM
# ---------------------------------------------------------

evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

ragas_llm = LangchainLLMWrapper(evaluator_llm)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

faithfulness = Faithfulness(llm=ragas_llm)

answer_relevancy = ResponseRelevancy(llm=ragas_llm)

context_precision = ContextPrecision(llm=ragas_llm)


METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
]


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------


def evaluate_rag_results(
    results: List[Dict[str, Any]],
):
    """
    Evaluate RAG responses using RAGAS.

    Each result should contain:

        question
        answer
        contexts
        reference
    """

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": result["reference"],
            }
            for result in results
        ]
    )

    evaluation_result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
    )

    return evaluation_result
