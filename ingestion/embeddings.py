from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from typing import Final

import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL_NAME: Final[str] = "BAAI/bge-m3"

# BGE-M3 dense embedding dimension.
EXPECTED_EMBEDDING_DIMENSION: Final[int] = 1024

DEFAULT_BATCH_SIZE: Final[int] = 32

DEFAULT_QUERY_INSTRUCTION: Final[str] = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# Exceptions
# ============================================================


class EmbeddingError(Exception):
    """Base exception for embedding pipeline failures."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when invalid text or document input is provided."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the embedding model cannot be initialized."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when vector generation fails."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when generated vectors fail validation."""


# ============================================================
# BGE-M3 Embedding Component
# ============================================================


class BGEEmbeddings(Embeddings):
    """
    Production-oriented BGE-M3 dense embedding component.

    Architecture:

        Documents / Query
                ↓
        Input Validation
                ↓
        Query Instruction (query only)
                ↓
        SentenceTransformer
                ↓
        Batched BGE-M3 Inference
                ↓
        L2-Normalized Dense Vectors
                ↓
        Vector Validation
                ↓
        Pinecone / Retriever

    Design characteristics:

        - Implements LangChain's Embeddings interface.
        - Lazy model initialization.
        - Model is loaded only once per instance.
        - Thread-safe model initialization.
        - Automatic CUDA/CPU device resolution.
        - Batched document inference.
        - Query/document embedding separation.
        - Optional query instruction.
        - Normalized dense vectors.
        - Embedding dimension validation.
        - NaN/Infinity validation.
        - Empty input protection.
        - Explicit inference-only execution.
        - Production-friendly exception hierarchy.

    Important:
        This class generates BGE-M3 dense embeddings.

        Pinecone persistence and retrieval belong to separate
        infrastructure components and should not be implemented
        inside this class.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        expected_dimension: int | None = EXPECTED_EMBEDDING_DIMENSION,
        trust_remote_code: bool = False,
    ) -> None:

        # --------------------------------------------------------
        # Configuration validation
        # --------------------------------------------------------

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if expected_dimension is not None:
            if expected_dimension <= 0:
                raise ValueError(
                    "expected_dimension must be greater than 0."
                )

        if not isinstance(query_instruction, str):
            raise ValueError(
                "query_instruction must be a string."
            )

        self.model_name = model_name.strip()
        self.device = self._resolve_device(device)

        self.batch_size = batch_size

        self.normalize_embeddings = (
            normalize_embeddings
        )

        self.query_instruction = (
            query_instruction.strip()
        )

        self.expected_dimension = (
            expected_dimension
        )

        self.trust_remote_code = (
            trust_remote_code
        )

        # --------------------------------------------------------
        # Lazy-loaded model
        # --------------------------------------------------------

        self._model: SentenceTransformer | None = None

        # Multiple requests may attempt first initialization
        # simultaneously in a deployed application.
        self._model_lock = threading.Lock()

        logger.info(
            "Configured BGE-M3 embeddings: "
            "model=%s device=%s batch_size=%d normalize=%s",
            self.model_name,
            self.device,
            self.batch_size,
            self.normalize_embeddings,
        )

    # ============================================================
    # Device Resolution
    # ============================================================

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> str:
        """
        Resolve inference device.

        If no device is explicitly provided:

            CUDA available → cuda
            otherwise      → cpu
        """

        if requested_device is None:

            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if not isinstance(requested_device, str):

            raise ValueError(
                "device must be a string or None."
            )

        device = requested_device.strip().lower()

        if device not in {"cpu", "cuda"}:

            raise ValueError(
                "device must be either 'cpu', 'cuda', or None."
            )

        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):

            raise ValueError(
                "CUDA was requested but is not available."
            )

        return device

    # ============================================================
    # Lazy Model Loading
    # ============================================================

    def _get_model(
        self,
    ) -> SentenceTransformer:
        """
        Return the loaded SentenceTransformer model.

        Model initialization is lazy so constructing the pipeline
        does not immediately allocate model memory.

        Initialization is protected by a lock to prevent duplicate
        model loading under concurrent first requests.
        """

        if self._model is not None:
            return self._model

        with self._model_lock:

            if self._model is not None:
                return self._model

            logger.info(
                "Loading embedding model '%s' on %s.",
                self.model_name,
                self.device,
            )

            try:

                model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=(
                        self.trust_remote_code
                    ),
                )

                model.eval()

            except Exception as exc:

                logger.exception(
                    "Failed to load embedding model '%s'.",
                    self.model_name,
                )

                raise EmbeddingModelError(
                    f"Unable to load embedding model "
                    f"'{self.model_name}'."
                ) from exc

            # ----------------------------------------------------
            # Verify actual model dimension once.
            # ----------------------------------------------------

            try:

                actual_dimension = (
                    model.get_sentence_embedding_dimension()
                )

            except Exception as exc:

                raise EmbeddingModelError(
                    "Unable to determine embedding dimension."
                ) from exc

            if actual_dimension is None:

                raise EmbeddingModelError(
                    "Embedding model returned no vector dimension."
                )

            if (
                self.expected_dimension is not None
                and actual_dimension
                != self.expected_dimension
            ):

                raise EmbeddingModelError(
                    "Unexpected embedding dimension. "
                    f"Expected {self.expected_dimension}, "
                    f"model reports {actual_dimension}."
                )

            self._model = model

            logger.info(
                "Embedding model loaded successfully. "
                "dimension=%d",
                actual_dimension,
            )

            return self._model

    # ============================================================
    # Input Validation
    # ============================================================

    @staticmethod
    def _validate_text(
        text: str,
        *,
        field_name: str,
    ) -> str:
        """
        Validate one text input.
        """

        if not isinstance(text, str):

            raise InvalidEmbeddingInputError(
                f"{field_name} must be a string."
            )

        cleaned_text = text.strip()

        if not cleaned_text:

            raise InvalidEmbeddingInputError(
                f"{field_name} cannot be empty."
            )

        return cleaned_text

    @classmethod
    def _validate_texts(
        cls,
        texts: Sequence[str],
    ) -> list[str]:
        """
        Validate a collection of document texts.
        """

        if not texts:

            raise InvalidEmbeddingInputError(
                "No texts were provided for embedding."
            )

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):

            cleaned_texts.append(
                cls._validate_text(
                    text,
                    field_name=f"text[{index}]",
                )
            )

        return cleaned_texts

    # ============================================================
    # Vector Validation
    # ============================================================

    def _validate_vector(
        self,
        vector: Sequence[float],
        *,
        vector_index: int | None = None,
    ) -> list[float]:
        """
        Validate one generated dense vector.

        Checks:

            - vector exists
            - expected dimensionality
            - numeric values
            - no NaN
            - no Infinity
        """

        location = (
            f" at index {vector_index}"
            if vector_index is not None
            else ""
        )

        if not vector:

            raise EmbeddingValidationError(
                f"Empty embedding vector generated{location}."
            )

        if (
            self.expected_dimension is not None
            and len(vector)
            != self.expected_dimension
        ):

            raise EmbeddingValidationError(
                f"Invalid embedding dimension{location}. "
                f"Expected {self.expected_dimension}, "
                f"received {len(vector)}."
            )

        validated_vector: list[float] = []

        for value_index, value in enumerate(vector):

            try:
                numeric_value = float(value)

            except (TypeError, ValueError) as exc:

                raise EmbeddingValidationError(
                    "Non-numeric embedding value "
                    f"at vector{location}, "
                    f"position {value_index}."
                ) from exc

            if not math.isfinite(numeric_value):

                raise EmbeddingValidationError(
                    "Non-finite embedding value "
                    f"at vector{location}, "
                    f"position {value_index}."
                )

            validated_vector.append(
                numeric_value
            )

        return validated_vector

    def _validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        """
        Validate a complete document embedding batch.
        """

        if len(vectors) != expected_count:

            raise EmbeddingValidationError(
                "Embedding count mismatch. "
                f"Expected {expected_count}, "
                f"received {len(vectors)}."
            )

        return [
            self._validate_vector(
                vector,
                vector_index=index,
            )
            for index, vector in enumerate(vectors)
        ]

    # ============================================================
    # Core Encoding
    # ============================================================

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Run BGE-M3 dense inference.

        SentenceTransformer handles internal batching according
        to `batch_size`.
        """

        cleaned_texts = self._validate_texts(
            texts
        )

        model = self._get_model()

        try:

            with torch.inference_mode():

                embeddings = model.encode(
                    cleaned_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=(
                        self.normalize_embeddings
                    ),
                )

        except torch.cuda.OutOfMemoryError as exc:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.exception(
                "CUDA out of memory while generating embeddings."
            )

            raise EmbeddingGenerationError(
                "GPU memory was exhausted while generating "
                "embeddings. Reduce batch_size or use CPU."
            ) from exc

        except Exception as exc:

            logger.exception(
                "BGE-M3 embedding generation failed."
            )

            raise EmbeddingGenerationError(
                "Failed to generate BGE-M3 embeddings."
            ) from exc

        vectors = embeddings.tolist()

        return self._validate_vectors(
            vectors,
            expected_count=len(cleaned_texts),
        )

    # ============================================================
    # LangChain Embeddings Interface
    # ============================================================

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Embed document/passages.

        This signature intentionally follows LangChain's
        `Embeddings` interface.

        Pinecone and LangChain vector stores can therefore use
        this class directly.
        """

        logger.debug(
            "Embedding %d document texts.",
            len(texts),
        )

        return self._encode(
            texts
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed a retrieval query.

        Query instructions are applied only to queries,
        never to stored document passages.
        """

        cleaned_query = self._validate_text(
            text,
            field_name="query",
        )

        if self.query_instruction:

            query_text = (
                f"{self.query_instruction}"
                f"{cleaned_query}"
            )

        else:
            query_text = cleaned_query

        vector = self._encode(
            [query_text]
        )[0]

        return vector

    # ============================================================
    # Project Convenience API
    # ============================================================

    def embed_langchain_documents(
        self,
        documents: Sequence[Document],
    ) -> list[list[float]]:
        """
        Convenience method for embedding LangChain Documents.

        LangChain's standard `embed_documents()` accepts strings.
        Our ingestion pipeline works with Document objects, so this
        method safely bridges the two representations.

        Metadata is intentionally not embedded.
        """

        if not documents:

            raise InvalidEmbeddingInputError(
                "No LangChain Documents were provided."
            )

        texts: list[str] = []

        for index, document in enumerate(documents):

            if not isinstance(document, Document):

                raise InvalidEmbeddingInputError(
                    "Expected LangChain Document "
                    f"at index {index}, received "
                    f"{type(document).__name__}."
                )

            text = self._validate_text(
                document.page_content,
                field_name=(
                    f"documents[{index}].page_content"
                ),
            )

            texts.append(text)

        vectors = self.embed_documents(
            texts
        )

        logger.info(
            "Generated %d document embeddings.",
            len(vectors),
        )

        return vectors

    # ============================================================
    # Runtime Information
    # ============================================================

    @property
    def embedding_dimension(
        self,
    ) -> int:
        """
        Return the actual embedding dimension.

        Calling this property initializes the model if it has
        not already been loaded.
        """

        model = self._get_model()

        dimension = (
            model.get_sentence_embedding_dimension()
        )

        if dimension is None:

            raise EmbeddingModelError(
                "Unable to determine embedding dimension."
            )

        return int(dimension)

    @property
    def is_loaded(
        self,
    ) -> bool:
        """
        Return whether the model is currently loaded.
        """

        return self._model is not None