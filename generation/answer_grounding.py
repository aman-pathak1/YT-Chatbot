from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


class AnswerGroundingError(Exception):
    """Base exception for answer-grounding errors."""


class AnswerGroundingConfigurationError(AnswerGroundingError):
    """Raised when grounding configuration is invalid."""


class AnswerGroundingInputError(AnswerGroundingError):
    """Raised when grounding input is invalid."""


@dataclass(frozen=True)
class GroundingConfig:
    model_name: str = "qwen3:8b"
    temperature: float = 0.0
    max_tokens: int = 512
    minimum_grounding_score: float = 0.70

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise AnswerGroundingConfigurationError(
                "model_name cannot be empty."
            )

        if self.temperature < 0:
            raise AnswerGroundingConfigurationError(
                "temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise AnswerGroundingConfigurationError(
                "max_tokens must be greater than 0."
            )

        if not 0.0 <= self.minimum_grounding_score <= 1.0:
            raise AnswerGroundingConfigurationError(
                "minimum_grounding_score must be between 0.0 and 1.0."
            )


@dataclass(frozen=True)
class GroundingResult:
    answer: str
    is_grounded: bool
    grounding_score: float
    explanation: str
    cited_timestamps: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.is_grounded


class AnswerGrounding:
    """
    Verifies whether a generated answer is supported by
    the retrieved context.

    The LLM does not generate a new answer.
    It only evaluates grounding.
    """

    def __init__(
        self,
        config: GroundingConfig | None = None,
    ) -> None:
        self.config = config or GroundingConfig()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an answer-grounding evaluator
inside a production RAG system.

Your task is to evaluate whether the generated answer
is supported by the provided retrieved context.

Rules:

1. Evaluate ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT rewrite the answer.
4. Do NOT add missing information.
5. Check whether the factual claims in the answer
   are supported by the context.
6. Check whether the answer contains unsupported claims.
7. Give a grounding score between 0.0 and 1.0.
8. Score 1.0 means the answer is fully supported.
9. Score 0.0 means the answer is completely unsupported.
10. Return ONLY valid JSON.
11. Do not use markdown.
12. Do not include explanations outside the JSON.

The JSON must contain exactly these fields:

is_grounded
grounding_score
explanation

Where:

is_grounded = true or false
grounding_score = number between 0.0 and 1.0
explanation = short explanation of the evaluation.
""".strip(),
                ),
                (
                    "human",
                    """
Retrieved context:

{context}

Generated answer:

{answer}

Evaluate whether the generated answer is fully
supported by the retrieved context.
""".strip(),
                ),
            ]
        )

        self._llm: ChatOllama | None = None
        self._llm_lock = threading.Lock()

        logger.info(
            "Answer grounding component initialized."
        )

    def _load_llm(self) -> ChatOllama:
        """
        Lazily initialize the Ollama LLM.
        """

        if self._llm is not None:
            return self._llm

        with self._llm_lock:
            if self._llm is not None:
                return self._llm

            logger.info(
                "Initializing grounding LLM: %s",
                self.config.model_name,
            )

            try:
                self._llm = ChatOllama(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens,
                    reasoning=False,
                )

            except Exception as exc:
                logger.exception(
                    "Failed to initialize grounding LLM."
                )

                raise AnswerGroundingError(
                    "Unable to initialize grounding LLM."
                ) from exc

        return self._llm

    @staticmethod
    def _validate_answer(answer: str) -> str:
        if not isinstance(answer, str):
            raise AnswerGroundingInputError(
                "Answer must be a string."
            )

        answer = answer.strip()

        if not answer:
            raise AnswerGroundingInputError(
                "Answer cannot be empty."
            )

        return answer

    @staticmethod
    def _validate_documents(
        documents: list[Document],
    ) -> list[Document]:
        if not isinstance(documents, list):
            raise AnswerGroundingInputError(
                "Documents must be provided as a list."
            )

        valid_documents: list[Document] = []

        for document in documents:
            if not isinstance(document, Document):
                continue

            if not document.page_content.strip():
                continue

            valid_documents.append(document)

        if not valid_documents:
            raise AnswerGroundingInputError(
                "At least one valid context document is required."
            )

        return valid_documents

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Convert seconds into HH:MM:SS.
        """

        total_seconds = max(0, int(seconds))

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    def _build_context(
        self,
        documents: list[Document],
    ) -> str:
        """
        Format retrieved documents for grounding evaluation.
        """

        context_parts: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            start_time = metadata.get(
                "chunk_start_time",
                metadata.get("start_time", 0.0),
            )

            try:
                start_time = float(start_time)
            except (TypeError, ValueError):
                start_time = 0.0

            timestamp = self._format_timestamp(
                start_time
            )

            context_parts.append(
                f"[Context {index} | Timestamp: {timestamp}]\n"
                f"{document.page_content.strip()}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _extract_json(response: str) -> str:
        """
        Extract JSON from model output.
        """

        response = response.strip()

        response = re.sub(
            r"```(?:json)?",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = response.replace(
            "```",
            "",
        ).strip()

        start = response.find("{")
        end = response.rfind("}")

        if start == -1:
            raise AnswerGroundingError(
                "No JSON object found in grounding response."
            )

        if end == -1 or end <= start:
            raise AnswerGroundingError(
                "Incomplete JSON object returned by grounding model."
            )

        return response[start : end + 1]

    def _parse_response(
        self,
        response: str,
        answer: str,
        documents: list[Document],
    ) -> GroundingResult:
        """
        Parse and validate grounding evaluation.
        """

        json_text = self._extract_json(response)

        import json

        try:
            data = json.loads(json_text)

        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON returned by grounding model: %s",
                response,
            )

            raise AnswerGroundingError(
                "Grounding model returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise AnswerGroundingError(
                "Grounding response must be a JSON object."
            )

        is_grounded = data.get("is_grounded")
        grounding_score = data.get("grounding_score")
        explanation = data.get("explanation")

        if not isinstance(is_grounded, bool):
            raise AnswerGroundingError(
                "is_grounded must be a boolean."
            )

        try:
            grounding_score = float(grounding_score)

        except (TypeError, ValueError) as exc:
            raise AnswerGroundingError(
                "grounding_score must be a valid number."
            ) from exc

        if not 0.0 <= grounding_score <= 1.0:
            raise AnswerGroundingError(
                "grounding_score must be between 0.0 and 1.0."
            )

        if not isinstance(explanation, str):
            raise AnswerGroundingError(
                "explanation must be a string."
            )

        explanation = explanation.strip()

        if not explanation:
            raise AnswerGroundingError(
                "explanation cannot be empty."
            )

        # Extract timestamps from the answer.
        timestamp_pattern = re.compile(
            r"\[Source:\s*(\d{2}:\d{2}:\d{2})\]"
        )

        cited_timestamps = tuple(
            timestamp_pattern.findall(answer)
        )

        # Final grounding decision also respects the configured
        # minimum score.
        final_grounded = (
            is_grounded
            and grounding_score
            >= self.config.minimum_grounding_score
        )

        return GroundingResult(
            answer=answer,
            is_grounded=final_grounded,
            grounding_score=grounding_score,
            explanation=explanation,
            cited_timestamps=cited_timestamps,
        )

    def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ) -> GroundingResult:
        """
        Evaluate whether an answer is grounded in context.
        """

        answer = self._validate_answer(answer)

        documents = self._validate_documents(
            documents
        )

        context = self._build_context(
            documents
        )

        llm = self._load_llm()

        chain = (
            self.prompt
            | llm
            | StrOutputParser()
        )

        logger.info(
            "Evaluating answer grounding."
        )

        try:
            response = chain.invoke(
                {
                    "context": context,
                    "answer": answer,
                }
            )

        except Exception as exc:
            logger.exception(
                "Answer grounding evaluation failed."
            )

            raise AnswerGroundingError(
                "Failed to evaluate answer grounding."
            ) from exc

        result = self._parse_response(
            response=response,
            answer=answer,
            documents=documents,
        )

        logger.info(
            "Grounding evaluation completed: "
            "grounded=%s score=%.2f",
            result.is_grounded,
            result.grounding_score,
        )

        return result

    def is_grounded(
        self,
        answer: str,
        documents: list[Document],
    ) -> bool:
        """
        Return only the final grounding decision.
        """

        return self.evaluate(
            answer=answer,
            documents=documents,
        ).is_grounded

    @property
    def is_loaded(self) -> bool:
        """
        Return whether the grounding LLM is initialized.
        """

        return self._llm is not None