from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class GuardrailError(Exception):
    """Base exception for answer guardrail errors."""


class GuardrailConfigurationError(GuardrailError):
    """Raised when guardrail configuration is invalid."""


class GuardrailValidationError(GuardrailError):
    """Raised when an answer violates a guardrail."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GuardrailConfig:
    min_answer_length: int = 5
    max_answer_length: int = 8000
    require_context: bool = True
    require_citation_when_timestamp_available: bool = True

    def __post_init__(self) -> None:
        if self.min_answer_length < 0:
            raise GuardrailConfigurationError(
                "min_answer_length cannot be negative."
            )

        if self.max_answer_length <= 0:
            raise GuardrailConfigurationError(
                "max_answer_length must be greater than 0."
            )

        if self.min_answer_length > self.max_answer_length:
            raise GuardrailConfigurationError(
                "min_answer_length cannot exceed max_answer_length."
            )


# ============================================================
# VALIDATION RESULT
# ============================================================

@dataclass(frozen=True)
class GuardrailResult:
    is_valid: bool
    answer: str
    reasons: tuple[str, ...]


# ============================================================
# ANSWER GUARDRAILS
# ============================================================

class AnswerGuardrails:
    """
    Validates generated RAG answers before they are returned
    to the user.

    Checks:

        Answer
          ↓
        Empty?
          ↓
        Length valid?
          ↓
        Context available?
          ↓
        Citation format valid?
          ↓
        Final validated answer
    """

    _CITATION_PATTERN = re.compile(
        r"\[Source:\s*\d{2}:\d{2}:\d{2}(?:\s*,\s*\d{2}:\d{2}:\d{2})*\]"
    )

    def __init__(
        self,
        config: GuardrailConfig | None = None,
    ) -> None:

        self.config = config or GuardrailConfig()

        logger.info(
            "Answer guardrails initialized."
        )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    def _validate_answer_text(
        self,
        answer: str,
    ) -> list[str]:

        reasons: list[str] = []

        if not isinstance(answer, str):
            reasons.append(
                "Answer must be a string."
            )
            return reasons

        answer = answer.strip()

        if not answer:
            reasons.append(
                "Answer is empty."
            )
            return reasons

        if len(answer) < self.config.min_answer_length:
            reasons.append(
                "Answer is shorter than the minimum allowed length."
            )

        if len(answer) > self.config.max_answer_length:
            reasons.append(
                "Answer exceeds the maximum allowed length."
            )

        return reasons

    # ========================================================
    # CONTEXT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_context(
        documents: Sequence[Document],
    ) -> list[str]:

        reasons: list[str] = []

        if not documents:
            reasons.append(
                "No retrieved context was provided."
            )
            return reasons

        for document in documents:

            if not isinstance(document, Document):
                reasons.append(
                    "Context contains an invalid document."
                )
                break

            if not document.page_content.strip():
                reasons.append(
                    "Context contains an empty document."
                )
                break

        return reasons

    # ========================================================
    # CITATION VALIDATION
    # ========================================================

    def _has_timestamp_context(
        self,
        documents: Sequence[Document],
    ) -> bool:

        for document in documents:

            metadata = document.metadata

            if (
                metadata.get("chunk_start_time") is not None
                or metadata.get("start_time") is not None
            ):
                return True

        return False

    def _validate_citations(
        self,
        answer: str,
        documents: Sequence[Document],
    ) -> list[str]:

        reasons: list[str] = []

        if not self.config.require_citation_when_timestamp_available:
            return reasons

        if not self._has_timestamp_context(documents):
            return reasons

        if not self._CITATION_PATTERN.search(answer):
            reasons.append(
                "Answer does not contain a valid timestamp citation "
                "although timestamped context is available."
            )

        return reasons

    # ========================================================
    # FORBIDDEN PATTERN CHECK
    # ========================================================

    @staticmethod
    def _check_forbidden_output(
        answer: str,
    ) -> list[str]:

        reasons: list[str] = []

        forbidden_patterns = [
            "as an ai language model",
            "i cannot access the video",
            "i don't have access to youtube",
        ]

        normalized_answer = answer.lower()

        for pattern in forbidden_patterns:

            if pattern in normalized_answer:
                reasons.append(
                    f"Answer contains forbidden pattern: '{pattern}'."
                )

        return reasons

    # ========================================================
    # VALIDATE
    # ========================================================

    def validate(
        self,
        answer: str,
        documents: Sequence[Document],
    ) -> GuardrailResult:

        reasons: list[str] = []

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        reasons.extend(
            self._validate_answer_text(answer)
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        if self.config.require_context:

            reasons.extend(
                self._validate_context(documents)
            )

        # ----------------------------------------------------
        # Citation
        # ----------------------------------------------------

        if not reasons:

            reasons.extend(
                self._validate_citations(
                    answer,
                    documents,
                )
            )

        # ----------------------------------------------------
        # Forbidden output
        # ----------------------------------------------------

        if not reasons:

            reasons.extend(
                self._check_forbidden_output(answer)
            )

        is_valid = len(reasons) == 0

        if is_valid:

            logger.info(
                "Answer passed all guardrails."
            )

        else:

            logger.warning(
                "Answer failed guardrails: %s",
                reasons,
            )

        return GuardrailResult(
            is_valid=is_valid,
            answer=answer.strip()
            if isinstance(answer, str)
            else str(answer),
            reasons=tuple(reasons),
        )

    # ========================================================
    # ENFORCE
    # ========================================================

    def enforce(
        self,
        answer: str,
        documents: Sequence[Document],
    ) -> str:

        result = self.validate(
            answer=answer,
            documents=documents,
        )

        if not result.is_valid:

            raise GuardrailValidationError(
                "Answer failed guardrails: "
                + " | ".join(result.reasons)
            )

        return result.answer

    # ========================================================
    # SAFE VALIDATION
    # ========================================================

    def validate_or_fallback(
        self,
        answer: str,
        documents: Sequence[Document],
        fallback: str = (
            "I don't have enough information in the "
            "provided video context."
        ),
    ) -> str:

        result = self.validate(
            answer=answer,
            documents=documents,
        )

        if result.is_valid:
            return result.answer

        logger.warning(
            "Returning fallback because answer failed guardrails."
        )

        return fallback