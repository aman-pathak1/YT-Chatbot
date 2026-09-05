# retrieval/reranker.py

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Sequence

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RerankerError(Exception):
    """Base exception for reranking errors."""


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""


class RerankerQueryError(RerankerError):
    """Raised when the query is invalid."""


class RerankerDocumentError(RerankerError):
    """Raised when retrieved documents are invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    batch_size: int = 8
    device: str | None = None
    max_length: int = 512

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise RerankerConfigurationError(
                "model_name cannot be empty."
            )

        if self.top_n <= 0:
            raise RerankerConfigurationError(
                "top_n must be greater than 0."
            )

        if self.batch_size <= 0:
            raise RerankerConfigurationError(
                "batch_size must be greater than 0."
            )

        if self.max_length <= 0:
            raise RerankerConfigurationError(
                "max_length must be greater than 0."
            )


# ============================================================
# RERANKED RESULT
# ============================================================

@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    score: float
    original_rank: int


# ============================================================
# RERANKER
# ============================================================

class BGEReranker:
    """
    Production-oriented BGE cross-encoder reranker.

    Flow:

        Query
          +
        Retrieved Documents
          ↓
        Cross Encoder
          ↓
        Relevance Scores
          ↓
        Sorted Documents
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
    ) -> None:

        self.config = config or RerankerConfig()

        self._model: CrossEncoder | None = None
        self._model_lock = threading.Lock()

        self.device = self._resolve_device(
            self.config.device
        )

        logger.info(
            "BGE Reranker initialized with device=%s",
            self.device,
        )

    # ========================================================
    # DEVICE
    # ========================================================

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> str:

        if requested_device:
            return requested_device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self) -> CrossEncoder:

        if self._model is not None:
            return self._model

        with self._model_lock:

            if self._model is not None:
                return self._model

            logger.info(
                "Loading reranker model: %s",
                self.config.model_name,
            )

            try:
                self._model = CrossEncoder(
                    self.config.model_name,
                    max_length=self.config.max_length,
                    device=self.device,
                )

            except Exception as exc:

                logger.exception(
                    "Failed to load reranker model."
                )

                raise RerankerError(
                    "Unable to load reranker model."
                ) from exc

        logger.info(
            "Reranker model loaded successfully."
        )

        return self._model

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise RerankerQueryError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise RerankerQueryError(
                "Query cannot be empty."
            )

        return query

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:

        if not documents:
            raise RerankerDocumentError(
                "No documents were provided for reranking."
            )

        validated_documents: list[Document] = []

        for document in documents:

            if not isinstance(document, Document):
                raise RerankerDocumentError(
                    "All items must be LangChain Document objects."
                )

            if not document.page_content.strip():
                continue

            validated_documents.append(document)

        if not validated_documents:
            raise RerankerDocumentError(
                "No documents contain usable content."
            )

        return validated_documents

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[Document]:

        query = self._validate_query(query)

        documents = self._validate_documents(documents)

        model = self._load_model()

        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        logger.info(
            "Reranking %d documents.",
            len(documents),
        )

        try:

            scores = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

        except RuntimeError as exc:

            if "out of memory" in str(exc).lower():

                logger.exception(
                    "Reranker ran out of memory."
                )

                raise RerankerError(
                    "Reranker ran out of memory. "
                    "Reduce batch_size or max_length."
                ) from exc

            raise RerankerError(
                "Reranking failed."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected reranking error."
            )

            raise RerankerError(
                "Reranking failed."
            ) from exc

        scored_documents = list(
            zip(documents, scores)
        )

        scored_documents.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        top_n = min(
            self.config.top_n,
            len(scored_documents),
        )

        results = [
            document
            for document, _score
            in scored_documents[:top_n]
        ]

        logger.info(
            "Reranking completed. Returning top %d documents.",
            len(results),
        )

        return results

    # ========================================================
    # RERANK WITH SCORES
    # ========================================================

    def rerank_with_scores(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[RerankedDocument]:

        query = self._validate_query(query)

        documents = self._validate_documents(documents)

        model = self._load_model()

        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        try:

            scores = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

        except RuntimeError as exc:

            if "out of memory" in str(exc).lower():

                raise RerankerError(
                    "Reranker ran out of memory. "
                    "Reduce batch_size or max_length."
                ) from exc

            raise RerankerError(
                "Reranking failed."
            ) from exc

        except Exception as exc:

            raise RerankerError(
                "Reranking failed."
            ) from exc

        ranked = [
            RerankedDocument(
                document=document,
                score=float(score),
                original_rank=index + 1,
            )
            for index, (document, score)
            in enumerate(zip(documents, scores))
        ]

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return ranked[: self.config.top_n]

    # ========================================================
    # RETRIEVE + RERANK
    # ========================================================

    def rerank_retrieval_results(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[Document]:

        """
        Convenience method for pipeline integration.

        Hybrid/MMR retrieval
                ↓
        Reranker
                ↓
        Top-N relevant documents
        """

        return self.rerank(
            query=query,
            documents=documents,
        )

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return self._model is not None