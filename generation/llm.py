from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class LLMGenerationError(Exception):
    """Base exception for LLM generation errors."""


class LLMConfigurationError(LLMGenerationError):
    """Raised when LLM configuration is invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "qwen3:8b"
    temperature: float = 0.0
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise LLMConfigurationError(
                "model_name cannot be empty."
            )

        if self.temperature < 0:
            raise LLMConfigurationError(
                "temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise LLMConfigurationError(
                "max_tokens must be greater than 0."
            )


# ============================================================
# QWEN GENERATOR
# ============================================================

class QwenGenerator:
    """
    LangChain wrapper around the locally running Qwen3 model.

    The same LLM instance is reused throughout the application
    instead of creating a new model object for every request.

    LCEL usage:

        prompt.runnable | generator.runnable
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
    ) -> None:

        self.config = config or LLMConfig()

        self._llm: ChatOllama | None = None
        self._lock = threading.Lock()

        logger.info(
            "Qwen generator initialized: model=%s",
            self.config.model_name,
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_llm(self) -> ChatOllama:

        if self._llm is not None:
            return self._llm

        with self._lock:

            if self._llm is not None:
                return self._llm

            logger.info(
                "Initializing Ollama model: %s",
                self.config.model_name,
            )

            try:
                self._llm = ChatOllama(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens,
                    reasoning=False,
                )

            except Exception as exc:

                logger.exception(
                    "Failed to initialize Qwen3."
                )

                raise LLMGenerationError(
                    "Unable to initialize Qwen3 LLM."
                ) from exc

        return self._llm

    # ========================================================
    # RUNNABLE
    # ========================================================

    @property
    def runnable(self) -> BaseChatModel:
        """
        Returns the LangChain ChatOllama Runnable.

        This allows direct LCEL composition.
        """

        return self._load_llm()

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt: str,
    ) -> str:

        if not isinstance(prompt, str):
            raise LLMGenerationError(
                "Prompt must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise LLMGenerationError(
                "Prompt cannot be empty."
            )

        llm = self._load_llm()

        try:

            response = llm.invoke(prompt)

        except Exception as exc:

            logger.exception(
                "Qwen3 generation failed."
            )

            raise LLMGenerationError(
                "LLM generation failed."
            ) from exc

        content = response.content

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        if not content:
            raise LLMGenerationError(
                "LLM returned an empty response."
            )

        return content

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_initialized(self) -> bool:
        return self._llm is not None