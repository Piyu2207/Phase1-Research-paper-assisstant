import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from backend.evals.eval_dataset import EVAL_DATASET
from backend.evals.ragas_evaluator import evaluate_rag_results
from backend.retrieval.vector_store import VectorStoreService
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.retriever import RerankedRetriever
from backend.chains.rag_chain import create_rag_chain

LOG_DIR = Path("data/evaluation")
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSONL_PATH = LOG_DIR / "ragas_results.jsonl"


def build_retriever():
    store = VectorStoreService()
    store.load()
    return RerankedRetriever(store, CrossEncoderReranker())


def run_rag():
    chain = create_rag_chain(build_retriever())
    results = []
    for item in EVAL_DATASET:
        response = chain.invoke(item["question"])
        results.append(
            {
                "question": item["question"],
                "answer": response.answer,
                "contexts": [s.snippet for s in response.sources],
                "reference": item["expected_answer"],
            }
        )
    return results


def main():
    results = run_rag()
    scores = dict(evaluate_rag_results(results))
    timestamp = datetime.now(timezone.utc).isoformat()
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "question": result["question"],
                        "answer_preview": result["answer"][:250],
                        **{
                            k: scores.get(k, 0)
                            for k in (
                                "faithfulness",
                                "answer_relevancy",
                                "context_precision",
                            )
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    table = pd.DataFrame(
        [
            {"Metric": "Faithfulness", "Score": scores.get("faithfulness", 0)},
            {"Metric": "Answer Relevancy", "Score": scores.get("answer_relevancy", 0)},
            {
                "Metric": "Context Precision",
                "Score": scores.get("context_precision", 0),
            },
        ]
    )
    print(table.to_string(index=False))
    print(f"Saved: {JSONL_PATH}")


if __name__ == "__main__":
    main()
