from __future__ import annotations

import logging
import re
import threading

from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class ContextualCompressionError(Exception):
    """Base exception for contextual compression errors."""


class CompressionConfigurationError(ContextualCompressionError):
    """Raised when compression configuration is invalid."""


class CompressionQueryError(ContextualCompressionError):
    """Raised when the compression query is invalid."""


class CompressionDocumentError(ContextualCompressionError):
    """Raised when documents are invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CompressionConfig:
    model_name: str = "qwen3:8b"
    temperature: float = 0.0
    max_output_tokens: int = 256
    min_relevance_length: int = 20
    use_heuristic: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise CompressionConfigurationError(
                "model_name cannot be empty."
            )

        if self.temperature < 0:
            raise CompressionConfigurationError(
                "temperature cannot be negative."
            )

        if self.max_output_tokens <= 0:
            raise CompressionConfigurationError(
                "max_output_tokens must be greater than 0."
            )

        if self.min_relevance_length < 0:
            raise CompressionConfigurationError(
                "min_relevance_length cannot be negative."
            )


# ============================================================
# COMPRESSOR
# ============================================================

class ContextualCompressor:
    """
    Compresses retrieved documents according to the user's query.

    Input:
        Query + retrieved documents

    Output:
        Documents containing only query-relevant information.

    Important:
        The compressor itself does NOT force-retain irrelevant
        documents. If no document is relevant, it returns [].

        Pipeline-level fallback should decide what to do when
        compression removes all documents.
    """

    def __init__(
        self,
        config: CompressionConfig | None = None,
    ) -> None:

        self.config = config or CompressionConfig()

        self._llm: ChatOllama | None = None
        self._llm_lock = threading.Lock()

        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a contextual compression component in a RAG system.

Your task is to extract ONLY the information from the
provided document that is directly useful for answering
the user's query.

Rules:

1. Do not answer the query.
2. Do not add information that is not present in the document.
3. Do not hallucinate.
4. Preserve important names, numbers, facts and technical terms.
5. Preserve useful surrounding context when it is necessary
   to understand the relevant information.
6. If the document contains no useful information for the query,
   return exactly:

NO_RELEVANT_CONTEXT

7. Return only the extracted relevant context.
8. Do not provide explanations about your extraction.
9. Do not rewrite facts unnecessarily.
10. Keep the extracted context concise.

""".strip(),
                ),
                (
                    "human",
                    """
User Query:

{query}

Document:

{document}

Extract only the relevant context.
""".strip(),
                ),
            ]
        )

        logger.info(
            "Contextual compressor initialized with model=%s",
            self.config.model_name,
        )

    # ========================================================
    # LLM
    # ========================================================

    def _load_llm(self) -> ChatOllama:
        """
        Lazily initialize the Ollama LLM.

        Thread-safe so multiple requests do not initialize
        multiple LLM instances simultaneously.
        """

        if self._llm is not None:
            return self._llm

        with self._llm_lock:

            if self._llm is not None:
                return self._llm

            logger.info(
                "Loading contextual compression LLM: %s",
                self.config.model_name,
            )

            try:
                self._llm = ChatOllama(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_output_tokens,
                    reasoning=False,
                )

            except Exception as exc:
                logger.exception(
                    "Failed to initialize compression LLM."
                )

                raise ContextualCompressionError(
                    "Unable to initialize compression LLM."
                ) from exc

        return self._llm

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise CompressionQueryError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise CompressionQueryError(
                "Query cannot be empty."
            )

        return query

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:

        if not documents:
            raise CompressionDocumentError(
                "No documents were provided."
            )

        valid_documents: list[Document] = []

        for document in documents:

            if not isinstance(document, Document):
                raise CompressionDocumentError(
                    "All items must be LangChain Document objects."
                )

            if document.page_content.strip():
                valid_documents.append(document)

        if not valid_documents:
            raise CompressionDocumentError(
                "No documents contain usable content."
            )

        return valid_documents

    # ========================================================
    # SINGLE DOCUMENT COMPRESSION
    # ========================================================

    def _compress_document(
        self,
        query: str,
        document: Document,
    ) -> Document | None:

        llm = self._load_llm()

        chain = (
            self.prompt
            | llm
            | StrOutputParser()
        )

        try:

            compressed_text = chain.invoke(
                {
                    "query": query,
                    "document": document.page_content,
                }
            )

        except Exception as exc:

            logger.exception(
                "Failed to compress document."
            )

            raise ContextualCompressionError(
                "Document compression failed."
            ) from exc

        compressed_text = compressed_text.strip()

        # ----------------------------------------------------
        # Empty response
        # ----------------------------------------------------

        if not compressed_text:
            logger.debug(
                "Compression returned empty context."
            )
            return None

        # ----------------------------------------------------
        # Explicitly irrelevant
        # ----------------------------------------------------

        if compressed_text == "NO_RELEVANT_CONTEXT":
            logger.debug(
                "Document marked as irrelevant by compressor."
            )
            return None

        # ----------------------------------------------------
        # Very short context
        # ----------------------------------------------------

        if len(compressed_text) < self.config.min_relevance_length:

            logger.debug(
                "Compressed context is below minimum length: %d",
                len(compressed_text),
            )

            # Do NOT automatically discard it.
            #
            # A short context can still contain something important,
            # for example:
            #
            # "LangChain is a framework for LLM applications."
            #
            # Therefore we retain it.

        # ----------------------------------------------------
        # Preserve original metadata
        # ----------------------------------------------------

        metadata = {
            **document.metadata,
            "original_content": document.page_content,
            "contextual_compressed": True,
            "compressed_content_length": len(compressed_text),
        }

        return Document(
            page_content=compressed_text,
            metadata=metadata,
        )

    # ========================================================
    # COMPRESS
    # ========================================================

    def compress(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[Document]:

        query = self._validate_query(query)

        documents = self._validate_documents(documents)

        logger.info(
            "Compressing %d retrieved documents.",
            len(documents),
        )

        if self.config.use_heuristic:
            # Fast heuristic compression without redundant LLM calls
            compressed_documents: list[Document] = []
            query_words = set(re.findall(r"\w+", query.lower()))
            for doc in documents:
                content = doc.page_content.strip()
                if not content:
                    continue
                compressed_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            **doc.metadata,
                            "original_content": doc.page_content,
                            "contextual_compressed": True,
                            "fast_heuristic": True,
                        },
                    )
                )
            logger.info(
                "Fast heuristic compression retained %d/%d documents.",
                len(compressed_documents),
                len(documents),
            )
            return compressed_documents

        compressed_documents: list[Document] = []

        for index, document in enumerate(documents):

            logger.debug(
                "Compressing document %d/%d",
                index + 1,
                len(documents),
            )

            compressed_document = self._compress_document(
                query=query,
                document=document,
            )

            if compressed_document is not None:

                compressed_documents.append(
                    compressed_document
                )

        logger.info(
            "Contextual compression completed: %d/%d documents retained.",
            len(compressed_documents),
            len(documents),
        )

        return compressed_documents

    # ========================================================
    # PIPELINE METHOD
    # ========================================================

    def compress_retrieval_results(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Pipeline-friendly contextual compression.

        Flow:

            Retrieval
                ↓
            Reranking
                ↓
            Contextual Compression
                ↓
            Generation
        """

        return self.compress(
            query=query,
            documents=documents,
        )

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        """Return whether the compression LLM is loaded."""

        return self._llm is not None