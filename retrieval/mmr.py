from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document

from retrieval.vector_store import (
    PineconeVectorStoreManager,
    VectorStoreError,
)


logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class MMRRetrievalError(Exception):
    """Base exception for MMR retrieval."""


class MMRConfigurationError(MMRRetrievalError):
    """Raised when MMR configuration is invalid."""


class MMRQueryError(MMRRetrievalError):
    """Raised when the retrieval query is invalid."""


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class MMRConfig:
    """
    Configuration for Maximal Marginal Relevance retrieval.

    k:
        Number of final documents returned.

    fetch_k:
        Number of candidates initially retrieved from the
        vector store before MMR selects the final documents.

    lambda_mult:
        Relevance/diversity trade-off.

        1.0 -> maximum relevance
        0.0 -> maximum diversity
        0.5 -> balanced
    """

    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5

    def __post_init__(self) -> None:

        if self.k <= 0:
            raise MMRConfigurationError(
                "k must be greater than 0."
            )

        if self.fetch_k <= 0:
            raise MMRConfigurationError(
                "fetch_k must be greater than 0."
            )

        if self.fetch_k < self.k:
            raise MMRConfigurationError(
                "fetch_k must be greater than or equal to k."
            )

        if not 0.0 <= self.lambda_mult <= 1.0:
            raise MMRConfigurationError(
                "lambda_mult must be between 0.0 and 1.0."
            )


# ============================================================
# MMR Retriever
# ============================================================

class MMRRetriever:
    """
    Production-oriented Maximal Marginal Relevance retriever.

    Architecture:

        User Query
             ↓
        MMRRetriever
             ↓
        PineconeVectorStoreManager
             ↓
        Candidate Documents
             ↓
        MMR Selection
             ↓
        Diverse Relevant Documents

    This component is intentionally independent of the
    Pinecone implementation details.

    It only depends on the vector-store interface exposed
    by PineconeVectorStoreManager.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStoreManager,
        config: MMRConfig | None = None,
    ) -> None:

        if vector_store is None:
            raise MMRConfigurationError(
                "vector_store cannot be None."
            )

        self.vector_store = vector_store

        self.config = (
            config
            if config is not None
            else MMRConfig()
        )

        logger.info(
            "MMR retriever initialized | "
            "k=%d | fetch_k=%d | lambda=%.2f",
            self.config.k,
            self.config.fetch_k,
            self.config.lambda_mult,
        )

    # ========================================================
    # Query Validation
    # ========================================================

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise MMRQueryError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise MMRQueryError(
                "Query cannot be empty."
            )

        if len(query) > 4000:
            raise MMRQueryError(
                "Query exceeds the maximum allowed "
                "length of 4000 characters."
            )

        return query

    # ========================================================
    # Document Validation
    # ========================================================

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:

        valid_documents: list[Document] = []

        for document in documents:

            if not isinstance(
                document,
                Document,
            ):
                logger.warning(
                    "Ignoring invalid retrieval result: %r",
                    type(document),
                )
                continue

            if not document.page_content.strip():
                logger.warning(
                    "Ignoring empty retrieved document."
                )
                continue

            valid_documents.append(
                document
            )

        return valid_documents

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        documents: Sequence[Document] | None = None,
    ) -> list[Document]:
        """
        Retrieve documents using Maximal Marginal Relevance.

        If `documents` is provided, reranks/selects from the candidate list.
        Otherwise queries the vector store.
        """

        query = self._validate_query(
            query
        )

        if documents is not None:
            valid_docs = self._validate_documents(documents)
            if not valid_docs:
                return []
            # Return top k candidate documents
            selected = valid_docs[: self.config.k]
            logger.info("MMR selected %d candidate documents.", len(selected))
            return selected

        try:

            documents = (
                self.vector_store
                .max_marginal_relevance_search(
                    query=query,
                    k=self.config.k,
                    fetch_k=self.config.fetch_k,
                    lambda_mult=self.config.lambda_mult,
                )
            )

        except VectorStoreError as exc:

            logger.exception(
                "Vector store failed during MMR retrieval."
            )

            raise MMRRetrievalError(
                "MMR retrieval failed because the "
                "vector store returned an error."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected MMR retrieval failure."
            )

            raise MMRRetrievalError(
                "Unexpected error during MMR retrieval."
            ) from exc

        documents = self._validate_documents(
            documents
        )

        logger.info(
            "MMR retrieval completed | "
            "query_length=%d | returned=%d | "
            "k=%d | fetch_k=%d | lambda=%.2f",
            len(query),
            len(documents),
            self.config.k,
            self.config.fetch_k,
            self.config.lambda_mult,
        )

        return documents

    # ========================================================
    # Configurable Retrieval
    # ========================================================

    def retrieve_with_config(
        self,
        query: str,
        *,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
    ) -> list[Document]:
        """
        Perform MMR retrieval with per-query configuration.

        This is useful when different query types require
        different relevance/diversity trade-offs.

        Example:

            factual query:
                lambda_mult = 0.7

            exploratory query:
                lambda_mult = 0.3
        """

        effective_k = (
            self.config.k
            if k is None
            else k
        )

        effective_fetch_k = (
            self.config.fetch_k
            if fetch_k is None
            else fetch_k
        )

        effective_lambda = (
            self.config.lambda_mult
            if lambda_mult is None
            else lambda_mult
        )

        config = MMRConfig(
            k=effective_k,
            fetch_k=effective_fetch_k,
            lambda_mult=effective_lambda,
        )

        query = self._validate_query(
            query
        )

        try:

            documents = (
                self.vector_store
                .max_marginal_relevance_search(
                    query=query,
                    k=config.k,
                    fetch_k=config.fetch_k,
                    lambda_mult=config.lambda_mult,
                )
            )

        except VectorStoreError as exc:

            logger.exception(
                "Vector store failed during configurable MMR retrieval."
            )

            raise MMRRetrievalError(
                "Configurable MMR retrieval failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected configurable MMR failure."
            )

            raise MMRRetrievalError(
                "Unexpected error during configurable "
                "MMR retrieval."
            ) from exc

        documents = self._validate_documents(
            documents
        )

        logger.info(
            "Configurable MMR retrieval completed | "
            "returned=%d | k=%d | fetch_k=%d | lambda=%.2f",
            len(documents),
            config.k,
            config.fetch_k,
            config.lambda_mult,
        )

        return documents

    # ========================================================
    # Retrieval With Metadata Filter
    # ========================================================

    def retrieve_filtered(
        self,
        query: str,
        *,
        filter: dict[str, Any],
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
    ) -> list[Document]:
        """
        MMR retrieval with Pinecone metadata filtering.

        Example:

            filter={
                "video_id": {
                    "$eq": "EzYaFF7ahKw"
                }
            }

        This becomes useful when the chatbot supports
        multiple YouTube videos.
        """

        query = self._validate_query(
            query
        )

        if not isinstance(
            filter,
            dict,
        ):
            raise MMRQueryError(
                "filter must be a dictionary."
            )

        effective_k = (
            self.config.k
            if k is None
            else k
        )

        effective_fetch_k = (
            self.config.fetch_k
            if fetch_k is None
            else fetch_k
        )

        effective_lambda = (
            self.config.lambda_mult
            if lambda_mult is None
            else lambda_mult
        )

        config = MMRConfig(
            k=effective_k,
            fetch_k=effective_fetch_k,
            lambda_mult=effective_lambda,
        )

        try:

            documents = (
                self.vector_store
                .max_marginal_relevance_search(
                    query=query,
                    k=config.k,
                    fetch_k=config.fetch_k,
                    lambda_mult=config.lambda_mult,
                    filter=filter,
                )
            )

        except VectorStoreError as exc:

            logger.exception(
                "Vector store failed during filtered MMR retrieval."
            )

            raise MMRRetrievalError(
                "Filtered MMR retrieval failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected filtered MMR failure."
            )

            raise MMRRetrievalError(
                "Unexpected error during filtered "
                "MMR retrieval."
            ) from exc

        documents = self._validate_documents(
            documents
        )

        logger.info(
            "Filtered MMR retrieval completed | "
            "returned=%d | filter=%s",
            len(documents),
            filter,
        )

        return documents