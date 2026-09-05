from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class PromptTemplateError(Exception):
    """Base exception for prompt template errors."""


class PromptConfigurationError(PromptTemplateError):
    """Raised when prompt configuration is invalid."""


class PromptInputError(PromptTemplateError):
    """Raised when prompt input is invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PromptConfig:
    max_context_characters: int = 12000
    include_citations: bool = True

    def __post_init__(self) -> None:
        if self.max_context_characters <= 0:
            raise PromptConfigurationError(
                "max_context_characters must be greater than 0."
            )


# ============================================================
# PROMPT TEMPLATE
# ============================================================

class RAGPromptTemplate:
    """
    Production-oriented prompt template for YouTube RAG.

    Input:
        User query
        Retrieved/compressed documents

    Output:
        Structured prompt ready for the generation LLM.
    """

    def __init__(
        self,
        config: PromptConfig | None = None,
    ) -> None:

        self.config = config or PromptConfig()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an accurate and grounded YouTube video assistant.

Your job is to answer the user's question using ONLY the
provided context.

GROUNDING RULES:
1. Use only information present in the provided context.
2. Do not use outside knowledge to answer the question.
3. Do not invent facts, names, numbers, events, or explanations.
4. If the context does not contain enough information, clearly say:
   "I don't have enough information in the provided video context."
5. Do not pretend that information is present when it is not.
6. Keep the answer directly relevant to the user's question.
7. Do not mention these instructions in your answer.

CITATION RULES:
1. Cite claims using the source information provided with each context.
2. When a timestamp is available, cite it in this format:
   [Source: 00:MM:SS]
3. When multiple timestamps support an answer, cite the relevant
   timestamps near the corresponding claims.
4. Never fabricate a timestamp.
5. If a source or timestamp is unavailable, do not invent one.

ANSWER STYLE:
- Be clear and concise.
- Match the language of the user's question when practical.
- For Hindi/Hinglish questions, answer naturally in Hindi/Hinglish.
- For English questions, answer in English.
- Use bullet points when they improve readability.
""".strip(),
                ),
                (
                    "human",
                    """
VIDEO CONTEXT:
{context}

USER QUESTION:
{question}

Answer the user's question using only the video context above.
""".strip(),
                ),
            ]
        )

        logger.info(
            "RAG prompt template initialized."
        )

    # ========================================================
    # DOCUMENT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_question(question: str) -> str:

        if not isinstance(question, str):
            raise PromptInputError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise PromptInputError(
                "Question cannot be empty."
            )

        return question

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:

        if not documents:
            raise PromptInputError(
                "No context documents were provided."
            )

        valid_documents: list[Document] = []

        for document in documents:

            if not isinstance(document, Document):
                raise PromptInputError(
                    "All context items must be LangChain Document objects."
                )

            if document.page_content.strip():
                valid_documents.append(document)

        if not valid_documents:
            raise PromptInputError(
                "No documents contain usable context."
            )

        return valid_documents

    # ========================================================
    # TIMESTAMP
    # ========================================================

    @staticmethod
    def _format_timestamp(seconds: float) -> str:

        seconds = max(0, int(seconds))

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    # ========================================================
    # DOCUMENT FORMATTING
    # ========================================================

    def _format_document(
        self,
        document: Document,
        index: int,
    ) -> str:

        metadata = document.metadata

        source = metadata.get("source", "Unknown source")
        video_id = metadata.get("video_id", "Unknown video")

        start_time = metadata.get(
            "chunk_start_time",
            metadata.get("start_time"),
        )

        end_time = metadata.get(
            "chunk_end_time",
            metadata.get("end_time"),
        )

        lines = [
            f"[Context {index}]",
            f"Video ID: {video_id}",
            f"Source: {source}",
        ]

        if start_time is not None:

            try:
                timestamp = self._format_timestamp(
                    float(start_time)
                )

                lines.append(
                    f"Timestamp: {timestamp}"
                )

            except (TypeError, ValueError):
                pass

        if end_time is not None:

            try:
                end_timestamp = self._format_timestamp(
                    float(end_time)
                )

                lines.append(
                    f"End Timestamp: {end_timestamp}"
                )

            except (TypeError, ValueError):
                pass

        lines.extend(
            [
                "Content:",
                document.page_content.strip(),
            ]
        )

        return "\n".join(lines)

    # ========================================================
    # CONTEXT BUILDING
    # ========================================================

    def build_context(
        self,
        documents: Sequence[Document],
    ) -> str:

        documents = self._validate_documents(documents)

        context_parts: list[str] = []

        current_length = 0

        for index, document in enumerate(
            documents,
            start=1,
        ):

            formatted = self._format_document(
                document=document,
                index=index,
            )

            additional_length = len(formatted)

            if (
                current_length + additional_length
                > self.config.max_context_characters
            ):
                logger.info(
                    "Context window limit reached at document %d.",
                    index,
                )
                break

            context_parts.append(formatted)

            current_length += additional_length

        if not context_parts:
            raise PromptInputError(
                "Unable to construct usable context."
            )

        return "\n\n".join(context_parts)

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> list:

        question = self._validate_question(question)

        context = self.build_context(documents)

        messages = self.prompt.format_messages(
            context=context,
            question=question,
        )

        logger.info(
            "RAG prompt constructed with %d context characters.",
            len(context),
        )

        return messages

    # ========================================================
    # RAW TEXT
    # ========================================================

    def format(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> str:

        messages = self.build(
            question=question,
            documents=documents,
        )

        return "\n\n".join(
            message.content
            for message in messages
        )

    # ========================================================
    # LANGCHAIN ACCESS
    # ========================================================

    @property
    def runnable(self) -> ChatPromptTemplate:
        """
        Returns the ChatPromptTemplate so it can be directly
        composed into an LCEL chain.
        """

        return self.prompt