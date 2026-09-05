from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from generation.llm import LLMConfig, QwenGenerator
from generation.prompt_template import (
    PromptConfig,
    RAGPromptTemplate,
)

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class AnswerGenerationError(Exception):
    """Base exception for answer generation errors."""


class AnswerInputError(AnswerGenerationError):
    """Raised when answer generation input is invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AnswerGeneratorConfig:
    llm: LLMConfig = LLMConfig()
    prompt: PromptConfig = PromptConfig()


# ============================================================
# ANSWER GENERATOR
# ============================================================

class AnswerGenerator:
    """
    Final RAG answer generation component.

    Pipeline:

        Question + Context
                ↓
          RAG Prompt
                ↓
             Qwen3
                ↓
        StrOutputParser
                ↓
           Final Answer
    """

    def __init__(
        self,
        config: AnswerGeneratorConfig | None = None,
    ) -> None:

        self.config = config or AnswerGeneratorConfig()

        self.prompt_template = RAGPromptTemplate(
            config=self.config.prompt,
        )

        self.llm = QwenGenerator(
            config=self.config.llm,
        )

        # ----------------------------------------------------
        # Actual LCEL chain
        # ----------------------------------------------------

        self.chain: Runnable = (
            self.prompt_template.runnable
            | self.llm.runnable
            | StrOutputParser()
        )

        logger.info(
            "Answer generation LCEL chain initialized."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_question(question: str) -> str:

        if not isinstance(question, str):
            raise AnswerInputError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise AnswerInputError(
                "Question cannot be empty."
            )

        return question

    @staticmethod
    def _validate_documents(
        documents: Sequence[Document],
    ) -> list[Document]:

        if not documents:
            raise AnswerInputError(
                "No context documents were provided."
            )

        valid_documents: list[Document] = []

        for document in documents:

            if not isinstance(document, Document):
                raise AnswerInputError(
                    "All context items must be LangChain Document objects."
                )

            if document.page_content.strip():
                valid_documents.append(document)

        if not valid_documents:
            raise AnswerInputError(
                "No documents contain usable context."
            )

        return valid_documents

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        question: str,
        documents: Sequence[Document],
    ) -> str:

        question = self._validate_question(question)

        documents = self._validate_documents(documents)

        logger.info(
            "Generating answer using %d context documents.",
            len(documents),
        )

        try:

            answer = self.chain.invoke(
                {
                    "question": question,
                    "context": self.prompt_template.build_context(
                        documents
                    ),
                }
            )

        except Exception as exc:

            logger.exception(
                "Answer generation failed."
            )

            raise AnswerGenerationError(
                "Failed to generate answer."
            ) from exc

        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()

        if not answer:
            raise AnswerGenerationError(
                "LLM returned an empty answer."
            )

        return answer

    def generate_stream(
        self,
        question: str,
        documents: Sequence[Document],
    ):
        """
        Stream answer tokens from the LLM chain.
        """
        question = self._validate_question(question)
        documents = self._validate_documents(documents)

        logger.info(
            "Streaming answer using %d context documents.",
            len(documents),
        )

        try:
            for chunk in self.chain.stream(
                {
                    "question": question,
                    "context": self.prompt_template.build_context(documents),
                }
            ):
                yield chunk
        except Exception as exc:
            logger.exception("Answer streaming failed.")
            raise AnswerGenerationError("Failed to stream answer.") from exc

    # ========================================================
    # LCEL ACCESS
    # ========================================================

    @property
    def runnable(self) -> Runnable:
        """
        Exposes the complete RAG generation chain.

        Input:
            {
                "question": str,
                "context": str
            }

        Output:
            str
        """

        return self.chain

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_ready(self) -> bool:

        return (
            self.llm.is_initialized
            and self.prompt_template is not None
        )