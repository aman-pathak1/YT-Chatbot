from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from retrieval.vector_store import (
    PineconeVectorStoreManager,
    VectorStoreError,
)


logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class HybridRetrievalError(Exception):
    """Base exception for hybrid retrieval."""


class HybridConfigurationError(HybridRetrievalError):
    """Raised when hybrid retrieval configuration is invalid."""


class HybridQueryError(HybridRetrievalError):
    """Raised when the retrieval query is invalid."""


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class HybridRetrievalConfig:
    """
    Configuration for dense + sparse hybrid retrieval.
    """

    dense_k: int = 10
    sparse_k: int = 10
    final_k: int = 10

    rrf_k: int = 60

    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def __post_init__(self) -> None:

        if self.dense_k <= 0:
            raise HybridConfigurationError(
                "dense_k must be greater than 0."
            )

        if self.sparse_k <= 0:
            raise HybridConfigurationError(
                "sparse_k must be greater than 0."
            )

        if self.final_k <= 0:
            raise HybridConfigurationError(
                "final_k must be greater than 0."
            )

        if self.rrf_k <= 0:
            raise HybridConfigurationError(
                "rrf_k must be greater than 0."
            )

        if self.dense_weight < 0:
            raise HybridConfigurationError(
                "dense_weight cannot be negative."
            )

        if self.sparse_weight < 0:
            raise HybridConfigurationError(
                "sparse_weight cannot be negative."
            )

        if (
            self.dense_weight == 0
            and self.sparse_weight == 0
        ):
            raise HybridConfigurationError(
                "At least one retrieval weight "
                "must be greater than 0."
            )


# ============================================================
# Internal Candidate
# ============================================================

@dataclass
class _Candidate:
    """
    Internal representation of one hybrid candidate.
    """

    document: Document

    dense_rank: int | None = None
    sparse_rank: int | None = None

    dense_score: float = 0.0
    sparse_score: float = 0.0

    fused_score: float = 0.0


# ============================================================
# Hybrid Retriever
# ============================================================

class HybridRetriever:
    """
    Hybrid dense + sparse retriever.

    Dense:
        Pinecone vector search.

    Sparse:
        Local BM25 search.

    Fusion:
        Weighted Reciprocal Rank Fusion (RRF).

    Constructor:

        HybridRetriever(
            vector_store=...,
            documents=...,
            config=...,
        )
    """

    def __init__(
        self,
        vector_store: PineconeVectorStoreManager,
        documents: Sequence[Document],
        config: HybridRetrievalConfig | None = None,
    ) -> None:

        if vector_store is None:
            raise HybridConfigurationError(
                "vector_store cannot be None."
            )

        if not documents:
            raise HybridConfigurationError(
                "documents cannot be empty."
            )

        self.vector_store = vector_store

        self.config = (
            config
            if config is not None
            else HybridRetrievalConfig()
        )

        self.documents = self._validate_documents(
            documents
        )

        if not self.documents:
            raise HybridConfigurationError(
                "No valid documents were provided."
            )

        # ----------------------------------------------------
        # Build BM25 index once.
        # ----------------------------------------------------

        tokenized_documents = [
            self._tokenize(
                document.page_content
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        logger.info(
            "Hybrid retriever initialized | "
            "documents=%d | dense_k=%d | sparse_k=%d | "
            "final_k=%d | dense_weight=%.2f | "
            "sparse_weight=%.2f",
            len(self.documents),
            self.config.dense_k,
            self.config.sparse_k,
            self.config.final_k,
            self.config.dense_weight,
            self.config.sparse_weight,
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise HybridQueryError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise HybridQueryError(
                "Query cannot be empty."
            )

        if len(query) > 4000:
            raise HybridQueryError(
                "Query exceeds maximum length of "
                "4000 characters."
            )

        return query

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
                    "Ignoring invalid document: %r",
                    type(document),
                )
                continue

            if not isinstance(
                document.page_content,
                str,
            ):
                continue

            if not document.page_content.strip():
                continue

            valid_documents.append(
                document
            )

        return valid_documents

    # ========================================================
    # Tokenization
    # ========================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Simple Unicode-friendly tokenization.

        Case is normalized using casefold().
        """

        return [
            token.casefold()
            for token in text.split()
            if token.strip()
        ]

    # ========================================================
    # Document Identity
    # ========================================================

    @staticmethod
    def _document_key(
        document: Document,
    ) -> str:
        """
        Generate stable identity for a document.

        This allows dense and sparse results referring
        to the same transcript chunk to be merged.
        """

        metadata = document.metadata

        video_id = metadata.get(
            "video_id"
        )

        chunk_index = metadata.get(
            "chunk_index"
        )

        segment_index = metadata.get(
            "segment_index"
        )

        if video_id is not None:

            if chunk_index is not None:
                return (
                    f"{video_id}:chunk:{chunk_index}"
                )

            if segment_index is not None:
                return (
                    f"{video_id}:segment:{segment_index}"
                )

        source = metadata.get(
            "source"
        )

        start_time = metadata.get(
            "start_time",
            metadata.get(
                "chunk_start_time"
            ),
        )

        return (
            f"{source}|"
            f"{start_time}|"
            f"{document.page_content}"
        )

    # ========================================================
    # Dense Retrieval
    # ========================================================

    def _dense_retrieve(
        self,
        query: str,
    ) -> list[Document]:

        try:

            documents = (
                self.vector_store.similarity_search(
                    query=query,
                    k=self.config.dense_k,
                )
            )

        except VectorStoreError as exc:

            logger.exception(
                "Dense retrieval failed."
            )

            raise HybridRetrievalError(
                "Dense retrieval failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected dense retrieval error."
            )

            raise HybridRetrievalError(
                "Unexpected dense retrieval error."
            ) from exc

        return self._validate_documents(
            documents
        )

    # ========================================================
    # Sparse Retrieval
    # ========================================================

    def _sparse_retrieve(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        top_indices = ranked_indices[
            : self.config.sparse_k
        ]

        return [
            (
                self.documents[index],
                float(scores[index]),
            )
            for index in top_indices
        ]

    # ========================================================
    # RRF Fusion
    # ========================================================

    def _fuse_results(
        self,
        dense_documents: Sequence[Document],
        sparse_documents: Sequence[
            tuple[Document, float]
        ],
    ) -> list[Document]:

        candidates: dict[str, _Candidate] = {}

        # ----------------------------------------------------
        # Dense results
        # ----------------------------------------------------

        for rank, document in enumerate(
            dense_documents,
            start=1,
        ):

            key = self._document_key(
                document
            )

            candidate = candidates.get(
                key
            )

            if candidate is None:

                candidate = _Candidate(
                    document=document
                )

                candidates[key] = candidate

            candidate.dense_rank = rank

            candidate.dense_score = 1.0 / (
                self.config.rrf_k + rank
            )

            candidate.fused_score += (
                self.config.dense_weight
                * candidate.dense_score
            )

        # ----------------------------------------------------
        # Sparse results
        # ----------------------------------------------------

        for rank, (
            document,
            sparse_score,
        ) in enumerate(
            sparse_documents,
            start=1,
        ):

            key = self._document_key(
                document
            )

            candidate = candidates.get(
                key
            )

            if candidate is None:

                candidate = _Candidate(
                    document=document
                )

                candidates[key] = candidate

            candidate.sparse_rank = rank
            candidate.sparse_score = sparse_score

            sparse_rrf_score = 1.0 / (
                self.config.rrf_k + rank
            )

            candidate.fused_score += (
                self.config.sparse_weight
                * sparse_rrf_score
            )

        # ----------------------------------------------------
        # Sort by fused RRF score.
        # ----------------------------------------------------

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.fused_score,
                candidate.dense_rank
                if candidate.dense_rank is not None
                else float("inf"),
                candidate.sparse_rank
                if candidate.sparse_rank is not None
                else float("inf"),
            ),
            reverse=True,
        )

        return [
            candidate.document
            for candidate in ranked_candidates[
                : self.config.final_k
            ]
        ]

    # ========================================================
    # Public Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        """
        Perform hybrid retrieval.
        """

        query = self._validate_query(
            query
        )

        dense_documents = (
            self._dense_retrieve(
                query
            )
        )

        sparse_documents = (
            self._sparse_retrieve(
                query
            )
        )

        results = self._fuse_results(
            dense_documents=dense_documents,
            sparse_documents=sparse_documents,
        )

        logger.info(
            "Hybrid retrieval completed | "
            "dense=%d | sparse=%d | final=%d",
            len(dense_documents),
            len(sparse_documents),
            len(results),
        )

        return results

    # ========================================================
    # Retrieval With Diagnostics
    # ========================================================

    def retrieve_with_scores(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Return retrieval results with dense rank,
        sparse rank, sparse score and fused score.

        Useful for debugging and evaluation.
        """

        query = self._validate_query(
            query
        )

        dense_documents = (
            self._dense_retrieve(
                query
            )
        )

        sparse_documents = (
            self._sparse_retrieve(
                query
            )
        )

        candidates: dict[str, _Candidate] = {}

        # ----------------------------------------------------
        # Dense candidates
        # ----------------------------------------------------

        for rank, document in enumerate(
            dense_documents,
            start=1,
        ):

            key = self._document_key(
                document
            )

            candidate = candidates.get(
                key
            )

            if candidate is None:

                candidate = _Candidate(
                    document=document
                )

                candidates[key] = candidate

            candidate.dense_rank = rank
            candidate.fused_score += (
                self.config.dense_weight
                / (
                    self.config.rrf_k + rank
                )
            )

        # ----------------------------------------------------
        # Sparse candidates
        # ----------------------------------------------------

        for rank, (
            document,
            sparse_score,
        ) in enumerate(
            sparse_documents,
            start=1,
        ):

            key = self._document_key(
                document
            )

            candidate = candidates.get(
                key
            )

            if candidate is None:

                candidate = _Candidate(
                    document=document
                )

                candidates[key] = candidate

            candidate.sparse_rank = rank
            candidate.sparse_score = sparse_score

            candidate.fused_score += (
                self.config.sparse_weight
                / (
                    self.config.rrf_k + rank
                )
            )

        # ----------------------------------------------------
        # Final ranking
        # ----------------------------------------------------

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: candidate.fused_score,
            reverse=True,
        )

        return [
            {
                "document": candidate.document,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "fused_score": candidate.fused_score,
            }
            for candidate in ranked_candidates[
                : self.config.final_k
            ]
        ]

    # ========================================================
    # Rebuild BM25
    # ========================================================

    def rebuild(
        self,
        documents: Sequence[Document],
    ) -> None:
        """
        Rebuild the local BM25 index.

        This should be called when the local transcript
        corpus changes.
        """

        valid_documents = (
            self._validate_documents(
                documents
            )
        )

        if not valid_documents:
            raise HybridConfigurationError(
                "Cannot rebuild BM25 index with "
                "empty documents."
            )

        self.documents = valid_documents

        tokenized_documents = [
            self._tokenize(
                document.page_content
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        logger.info(
            "BM25 index rebuilt | documents=%d",
            len(self.documents),
        )