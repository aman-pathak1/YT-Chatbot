from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class ContextOptimizationError(Exception):
    """Base exception for context optimization errors."""


class ContextOptimizationConfigurationError(
    ContextOptimizationError
):
    """Raised when optimizer configuration is invalid."""


class ContextOptimizationInputError(
    ContextOptimizationError
):
    """Raised when optimizer input is invalid."""


@dataclass(frozen=True)
class ContextOptimizerConfig:
    """
    Configuration for context window optimization.

    max_context_characters:
        Maximum total number of characters allowed
        in the final context.

    max_document_characters:
        Maximum characters allowed from one document.

    min_document_characters:
        Minimum size required for a document to be
        considered useful.
    """

    max_context_characters: int = 12000
    max_document_characters: int = 3000
    min_document_characters: int = 20

    def __post_init__(self) -> None:
        if self.max_context_characters <= 0:
            raise ContextOptimizationConfigurationError(
                "max_context_characters must be greater than 0."
            )

        if self.max_document_characters <= 0:
            raise ContextOptimizationConfigurationError(
                "max_document_characters must be greater than 0."
            )

        if self.min_document_characters <= 0:
            raise ContextOptimizationConfigurationError(
                "min_document_characters must be greater than 0."
            )

        if (
            self.min_document_characters
            > self.max_document_characters
        ):
            raise ContextOptimizationConfigurationError(
                "min_document_characters cannot be greater "
                "than max_document_characters."
            )


@dataclass(frozen=True)
class ContextOptimizationResult:
    """
    Result of context window optimization.
    """

    documents: tuple[Document, ...]
    original_document_count: int
    optimized_document_count: int
    original_character_count: int
    optimized_character_count: int

    @property
    def characters_saved(self) -> int:
        return (
            self.original_character_count
            - self.optimized_character_count
        )

    @property
    def compression_ratio(self) -> float:
        if self.original_character_count == 0:
            return 0.0

        return (
            self.optimized_character_count
            / self.original_character_count
        )

    @property
    def optimized_documents(self) -> list[Document]:
        return list(self.documents)


class ContextWindowOptimizer:
    """
    Optimizes retrieved context before sending it
    to the generation model.

    The optimizer is intentionally deterministic.

    It does NOT call an LLM.

    Main responsibilities:

    1. Remove empty/very small documents.
    2. Remove duplicate documents.
    3. Respect per-document character limits.
    4. Respect the overall context character budget.
    5. Preserve document ordering.
    6. Preserve document metadata.
    """

    def __init__(
        self,
        config: ContextOptimizerConfig | None = None,
    ) -> None:
        self.config = config or ContextOptimizerConfig()

        logger.info(
            "Context window optimizer initialized."
        )

    @staticmethod
    def _validate_documents(
        documents: list[Document],
    ) -> list[Document]:
        if not isinstance(documents, list):
            raise ContextOptimizationInputError(
                "Documents must be provided as a list."
            )

        valid_documents: list[Document] = []

        for document in documents:
            if not isinstance(document, Document):
                continue

            if not isinstance(
                document.page_content,
                str,
            ):
                continue

            if not document.page_content.strip():
                continue

            valid_documents.append(document)

        return valid_documents

    def _remove_duplicates(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Remove exact duplicate document content while
        preserving the first occurrence.
        """

        seen: set[str] = set()
        unique_documents: list[Document] = []

        for document in documents:
            normalized_text = " ".join(
                document.page_content.split()
            ).lower()

            if normalized_text in seen:
                continue

            seen.add(normalized_text)
            unique_documents.append(document)

        return unique_documents

    def _truncate_document(
        self,
        document: Document,
    ) -> Document:
        """
        Truncate a document to the configured
        per-document character limit.

        Metadata is preserved.
        """

        text = document.page_content.strip()

        if len(text) <= self.config.max_document_characters:
            return document

        max_chars = self.config.max_document_characters

        truncated_text = text[:max_chars].rstrip()

        # Avoid cutting in the middle of a word when possible.
        last_space = truncated_text.rfind(" ")

        if last_space >= int(max_chars * 0.8):
            truncated_text = truncated_text[:last_space].rstrip()

        metadata = {
            **document.metadata,
            "context_truncated": True,
            "original_content_length": len(text),
            "optimized_content_length": len(
                truncated_text
            ),
        }

        return Document(
            page_content=truncated_text,
            metadata=metadata,
        )

    def _fit_context_budget(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Select documents while respecting the total
        context character budget.

        Documents are assumed to already be ranked
        by retrieval relevance.
        """

        selected: list[Document] = []
        current_characters = 0

        for document in documents:
            text = document.page_content.strip()

            if len(text) < self.config.min_document_characters:
                continue

            separator_cost = 2 if selected else 0

            projected_size = (
                current_characters
                + separator_cost
                + len(text)
            )

            if (
                projected_size
                <= self.config.max_context_characters
            ):
                selected.append(document)

                current_characters = projected_size

                continue

            # If the document doesn't completely fit,
            # use the remaining context budget when useful.
            remaining = (
                self.config.max_context_characters
                - current_characters
                - separator_cost
            )

            if remaining < self.config.min_document_characters:
                break

            partial_text = text[:remaining].rstrip()

            last_space = partial_text.rfind(" ")

            if last_space >= int(remaining * 0.8):
                partial_text = partial_text[:last_space].rstrip()

            if len(partial_text) < self.config.min_document_characters:
                break

            metadata = {
                **document.metadata,
                "context_truncated": True,
                "original_content_length": len(text),
                "optimized_content_length": len(
                    partial_text
                ),
            }

            selected.append(
                Document(
                    page_content=partial_text,
                    metadata=metadata,
                )
            )

            current_characters += (
                separator_cost + len(partial_text)
            )

            break

        return selected

    def optimize(
        self,
        documents: list[Document],
    ) -> ContextOptimizationResult:
        """
        Optimize retrieved documents for the LLM
        context window.
        """

        original_count = len(documents)

        valid_documents = self._validate_documents(
            documents
        )

        original_character_count = sum(
            len(document.page_content)
            for document in valid_documents
        )

        if not valid_documents:
            return ContextOptimizationResult(
                documents=tuple(),
                original_document_count=original_count,
                optimized_document_count=0,
                original_character_count=original_character_count,
                optimized_character_count=0,
            )

        # Step 1: Remove duplicates.
        unique_documents = self._remove_duplicates(
            valid_documents
        )

        # Step 2: Apply per-document limit.
        truncated_documents = [
            self._truncate_document(document)
            for document in unique_documents
        ]

        # Step 3: Remove documents that became
        # too small after truncation.
        usable_documents = [
            document
            for document in truncated_documents
            if len(document.page_content.strip())
            >= self.config.min_document_characters
        ]

        # Step 4: Apply global context budget.
        optimized_documents = self._fit_context_budget(
            usable_documents
        )

        optimized_character_count = sum(
            len(document.page_content)
            for document in optimized_documents
        )

        result = ContextOptimizationResult(
            documents=tuple(optimized_documents),
            original_document_count=original_count,
            optimized_document_count=len(
                optimized_documents
            ),
            original_character_count=original_character_count,
            optimized_character_count=optimized_character_count,
        )

        logger.info(
            "Context optimization completed: "
            "documents=%d -> %d, "
            "characters=%d -> %d",
            result.original_document_count,
            result.optimized_document_count,
            result.original_character_count,
            result.optimized_character_count,
        )

        return result

    def optimize_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Convenience method returning only optimized documents.
        """

        return self.optimize(
            documents
        ).optimized_documents