from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Sequence
from typing import Final

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================
# Constants
# ============================================================

DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_METRIC: Final[str] = "cosine"
DEFAULT_CLOUD: Final[str] = "aws"
DEFAULT_REGION: Final[str] = "us-east-1"

DEFAULT_NAMESPACE: Final[str] = "youtube-transcripts"

DEFAULT_BATCH_SIZE: Final[int] = 100


# ============================================================
# Exceptions
# ============================================================


class VectorStoreError(Exception):
    """Base exception for vector-store operations."""


class PineconeConfigurationError(VectorStoreError):
    """Raised when Pinecone configuration is invalid."""


class PineconeIndexError(VectorStoreError):
    """Raised when Pinecone index operations fail."""


class InvalidDocumentError(VectorStoreError):
    """Raised when documents are invalid."""


class VectorUpsertError(VectorStoreError):
    """Raised when document upsert fails."""


# ============================================================
# Pinecone Vector Store
# ============================================================


class PineconeVectorStoreManager:
    """
    Production-oriented Pinecone vector-store manager.

    Responsibilities:

        LangChain Documents
                ↓
        Validation
                ↓
        BGE-M3 Embeddings
                ↓
        Pinecone Vector Store
                ↓
        Namespace
                ↓
        Upsert / Retrieval

    Responsibilities intentionally excluded:

        - Embedding model implementation
        - Text splitting
        - Translation
        - LLM generation

    Those components remain independent and are composed later
    by the application / LangChain pipeline.

    Expected embedding dimension:

        BGE-M3 → 1024

    Expected similarity metric:

        cosine
    """

    def __init__(
        self,
        embedding_model,
        index_name: str,
        namespace: str = DEFAULT_NAMESPACE,
        dimension: int = DEFAULT_DIMENSION,
        metric: str = DEFAULT_METRIC,
        cloud: str = DEFAULT_CLOUD,
        region: str = DEFAULT_REGION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        api_key: str | None = None,
    ) -> None:

        # --------------------------------------------------------
        # Configuration validation
        # --------------------------------------------------------

        if not index_name or not index_name.strip():
            raise ValueError(
                "index_name cannot be empty."
            )

        if not namespace or not namespace.strip():
            raise ValueError(
                "namespace cannot be empty."
            )

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than 0."
            )

        if not metric.strip():
            raise ValueError(
                "metric cannot be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if embedding_model is None:
            raise ValueError(
                "embedding_model cannot be None."
            )

        # --------------------------------------------------------
        # Store configuration
        # --------------------------------------------------------

        self.embedding_model = embedding_model
        self.index_name = index_name.strip()
        self.namespace = namespace.strip()
        self.dimension = dimension
        self.metric = metric.strip().lower()
        self.cloud = cloud.strip().lower()
        self.region = region.strip()
        self.batch_size = batch_size

        # --------------------------------------------------------
        # API key
        # --------------------------------------------------------

        self.api_key = (
            api_key
            or os.getenv("PINECONE_API_KEY")
        )

        if not self.api_key:
            raise PineconeConfigurationError(
                "PINECONE_API_KEY was not found. "
                "Add it to the environment or .env file."
            )

        # --------------------------------------------------------
        # Pinecone client
        # --------------------------------------------------------

        try:

            self.client = Pinecone(
                api_key=self.api_key
            )

        except Exception as exc:

            logger.exception(
                "Failed to initialize Pinecone client."
            )

            raise PineconeConfigurationError(
                "Unable to initialize Pinecone client."
            ) from exc

        logger.info(
            "Pinecone manager initialized: "
            "index=%s namespace=%s dimension=%d metric=%s",
            self.index_name,
            self.namespace,
            self.dimension,
            self.metric,
        )

        # --------------------------------------------------------
        # Ensure index
        # --------------------------------------------------------

        self._ensure_index()

        # --------------------------------------------------------
        # LangChain VectorStore
        # --------------------------------------------------------

        try:

            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_model,
                namespace=self.namespace,
                pinecone_api_key=self.api_key,
            )

        except Exception as exc:

            logger.exception(
                "Failed to initialize LangChain PineconeVectorStore."
            )

            raise PineconeIndexError(
                "Unable to initialize LangChain Pinecone vector store."
            ) from exc

    # ============================================================
    # Index Management
    # ============================================================

    def _ensure_index(self) -> None:
        """
        Create the Pinecone index if it does not exist.

        If it already exists, validate its dimension and metric.
        """

        try:

            existing_indexes = self.client.list_indexes()

            existing_names = {
                item["name"]
                for item in existing_indexes
            }

        except Exception as exc:

            logger.exception(
                "Unable to list Pinecone indexes."
            )

            raise PineconeIndexError(
                "Failed to inspect Pinecone indexes."
            ) from exc

        if self.index_name not in existing_names:

            logger.info(
                "Pinecone index '%s' does not exist. "
                "Creating it.",
                self.index_name,
            )

            try:

                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region,
                    ),
                )

            except Exception as exc:

                logger.exception(
                    "Failed to create Pinecone index '%s'.",
                    self.index_name,
                )

                raise PineconeIndexError(
                    f"Unable to create Pinecone index "
                    f"'{self.index_name}'."
                ) from exc

            logger.info(
                "Pinecone index '%s' creation requested.",
                self.index_name,
            )

        else:

            logger.info(
                "Pinecone index '%s' already exists.",
                self.index_name,
            )

        self._validate_index()

    def _validate_index(self) -> None:
        """
        Validate the actual Pinecone index configuration.

        Dimension mismatch is treated as a hard failure because
        vectors from BGE-M3 cannot be inserted into an index with
        a different dimensionality.
        """

        try:

            description = self.client.describe_index(
                self.index_name
            )

        except Exception as exc:

            logger.exception(
                "Unable to describe Pinecone index '%s'.",
                self.index_name,
            )

            raise PineconeIndexError(
                f"Unable to validate Pinecone index "
                f"'{self.index_name}'."
            ) from exc

        actual_dimension = getattr(
            description,
            "dimension",
            None,
        )

        actual_metric = getattr(
            description,
            "metric",
            None,
        )

        if (
            actual_dimension is not None
            and actual_dimension != self.dimension
        ):

            raise PineconeIndexError(
                "Pinecone dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {actual_dimension}."
            )

        if (
            actual_metric is not None
            and actual_metric.lower()
            != self.metric
        ):

            raise PineconeIndexError(
                "Pinecone metric mismatch. "
                f"Expected '{self.metric}', "
                f"received '{actual_metric}'."
            )

        logger.info(
            "Pinecone index validation successful: "
            "dimension=%s metric=%s",
            actual_dimension,
            actual_metric,
        )

    # ============================================================
    # Document Validation
    # ============================================================

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Validate LangChain Documents before upsert.
        """

        if not documents:

            raise InvalidDocumentError(
                "No documents were provided."
            )

        validated: list[Document] = []

        for index, document in enumerate(documents):

            if not isinstance(document, Document):

                raise InvalidDocumentError(
                    f"Expected Document at index {index}, "
                    f"received {type(document).__name__}."
                )

            text = document.page_content.strip()

            if not text:

                raise InvalidDocumentError(
                    f"Document at index {index} contains empty text."
                )

            validated.append(
                document
            )

        return validated

    # ============================================================
    # Deterministic IDs
    # ============================================================

    def _generate_document_id(
        self,
        document: Document,
        position: int,
    ) -> str:
        """
        Generate a deterministic Pinecone vector ID.

        The hash is based on stable document information instead
        of Python's built-in hash(), whose value is not stable
        across processes.
        """

        metadata = document.metadata

        video_id = str(
            metadata.get(
                "video_id",
                "unknown-video",
            )
        )

        chunk_index = str(
            metadata.get(
                "chunk_index",
                position,
            )
        )

        start_time = str(
            metadata.get(
                "chunk_start_time",
                metadata.get(
                    "start_time",
                    0.0,
                ),
            )
        )

        raw_key = (
            f"{video_id}|"
            f"{chunk_index}|"
            f"{start_time}|"
            f"{document.page_content}"
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return (
            f"{video_id}-{chunk_index}-{digest[:16]}"
        )

    # ============================================================
    # Metadata Preparation
    # ============================================================

    @staticmethod
    def _prepare_metadata(
        document: Document,
    ) -> dict:
        """
        Prepare Pinecone-compatible metadata.

        Pinecone metadata should contain scalar values or lists
        of scalar values. Complex Python objects are removed.
        """

        metadata: dict = {}

        for key, value in document.metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):

                metadata[key] = value

            elif isinstance(value, list):

                if all(
                    isinstance(
                        item,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    )
                    for item in value
                ):

                    metadata[key] = value

        return metadata

    # ============================================================
    # Upsert
    # ============================================================

    def add_documents(
        self,
        documents: Sequence[Document],
    ) -> list[str]:
        """
        Add LangChain Documents to Pinecone.

        The LangChain Pinecone vector store handles:

            Document → embedding → Pinecone upsert

        Returns:
            Deterministic vector IDs.
        """

        validated_documents = (
            self._validate_documents(
                documents
            )
        )

        document_ids = [
            self._generate_document_id(
                document=document,
                position=index,
            )
            for index, document in enumerate(
                validated_documents
            )
        ]

        logger.info(
            "Upserting %d documents into Pinecone "
            "namespace '%s'.",
            len(validated_documents),
            self.namespace,
        )

        try:

            for start in range(
                0,
                len(validated_documents),
                self.batch_size,
            ):

                end = min(
                    start + self.batch_size,
                    len(validated_documents),
                )

                batch_documents = (
                    validated_documents[start:end]
                )

                batch_ids = (
                    document_ids[start:end]
                )

                logger.debug(
                    "Upserting documents %d-%d.",
                    start,
                    end - 1,
                )

                self.vector_store.add_documents(
                    documents=batch_documents,
                    ids=batch_ids,
                )

        except Exception as exc:

            logger.exception(
                "Pinecone document upsert failed."
            )

            raise VectorUpsertError(
                "Failed to upsert documents into Pinecone."
            ) from exc

        logger.info(
            "Successfully upserted %d documents.",
            len(document_ids),
        )

        return document_ids

    # ============================================================
    # Similarity Search
    # ============================================================

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """
        Perform basic similarity retrieval.

        This is intentionally kept as a low-level retrieval
        method. Advanced retrieval strategies such as:

            - MMR
            - Hybrid retrieval
            - reranking
            - contextual compression
            - query rewriting

        will be implemented at the retrieval layer.
        """

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        try:

            return self.vector_store.similarity_search(
                query=query,
                k=k,
            )

        except Exception as exc:

            logger.exception(
                "Pinecone similarity search failed."
            )

            raise VectorStoreError(
                "Failed to perform similarity search."
            ) from exc

    # ============================================================
    # MMR Search
    # ============================================================

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """
        Perform Maximal Marginal Relevance retrieval.

        MMR improves diversity among retrieved chunks and will
        later form part of the retrieval pipeline.
        """

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        if fetch_k < k:
            raise ValueError(
                "fetch_k must be greater than or equal to k."
            )

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )

        try:

            return (
                self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
            )

        except Exception as exc:

            logger.exception(
                "Pinecone MMR search failed."
            )

            raise VectorStoreError(
                "Failed to perform MMR retrieval."
            ) from exc

    # ============================================================
    # Namespace Information
    # ============================================================

    def describe(
        self,
    ) -> dict:
        """
        Return basic vector-store configuration.
        """

        return {
            "index_name": self.index_name,
            "namespace": self.namespace,
            "dimension": self.dimension,
            "metric": self.metric,
            "batch_size": self.batch_size,
        }