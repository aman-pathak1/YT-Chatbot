from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama

from ingestion.embeddings import BGEEmbeddings


@dataclass
class RAGASResult:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


class RAGASEvaluator:

    def __init__(self) -> None:

        # Local Qwen3 8B
        ollama_llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
            reasoning=False,
        )

        self.evaluator_llm = LangchainLLMWrapper(ollama_llm)

        # Local BGE-M3
        bge_embeddings = BGEEmbeddings()

        self.evaluator_embeddings = LangchainEmbeddingsWrapper(
            bge_embeddings
        )

        self.metrics = [
            Faithfulness(llm=self.evaluator_llm),

            AnswerRelevancy(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings,
            ),

            ContextPrecision(llm=self.evaluator_llm),

            ContextRecall(llm=self.evaluator_llm),
        ]

        # Qwen3 local inference ko zyada time do
        self.run_config = RunConfig(
            timeout=600,
            max_retries=2,
            max_workers=1,
        )

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> RAGASResult:

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )

        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            run_config=self.run_config,
            show_progress=True,
        )

        scores = result.to_pandas().iloc[0]

        return RAGASResult(
            faithfulness=self._safe_score(
                scores.get("faithfulness")
            ),
            answer_relevancy=self._safe_score(
                scores.get("answer_relevancy")
            ),
            context_precision=self._safe_score(
                scores.get("context_precision")
            ),
            context_recall=self._safe_score(
                scores.get("context_recall")
            ),
        )

    @staticmethod
    def _safe_score(value: Any) -> float | None:

        if value is None:
            return None

        try:
            value = float(value)

            if value != value:  # NaN check
                return None

            return value

        except (TypeError, ValueError):
            return None