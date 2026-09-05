from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class MultiQueryError(Exception):
    """Base exception for multi-query generation errors."""


class MultiQueryConfigurationError(MultiQueryError):
    """Raised when configuration is invalid."""


class MultiQueryInputError(MultiQueryError):
    """Raised when query input is invalid."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MultiQueryConfig:
    model_name: str = "qwen3:8b"
    num_queries: int = 3
    temperature: float = 0.0
    max_tokens: int = 512

    def __post_init__(self) -> None:

        if not self.model_name.strip():
            raise MultiQueryConfigurationError(
                "model_name cannot be empty."
            )

        if self.num_queries < 2:
            raise MultiQueryConfigurationError(
                "num_queries must be at least 2."
            )

        if self.num_queries > 10:
            raise MultiQueryConfigurationError(
                "num_queries cannot exceed 10."
            )

        if self.temperature < 0:
            raise MultiQueryConfigurationError(
                "temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise MultiQueryConfigurationError(
                "max_tokens must be greater than 0."
            )


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class MultiQueryResult:
    original_query: str
    queries: tuple[str, ...]

    @property
    def all_queries(self) -> list[str]:
        return list(self.queries)


# ============================================================
# MULTI QUERY GENERATOR
# ============================================================

class MultiQueryGenerator:
    """
    Generates multiple semantically different retrieval queries
    from one user's question.

    Example:

        User Query:
        "What does the video say about YouTube?"

        Generated Queries:
        1. What does the video mention about YouTube?
        2. What information is provided about the YouTube channel?
        3. What does the video discuss regarding YouTube playlists?
    """

    def __init__(
        self,
        config: MultiQueryConfig | None = None,
    ) -> None:

        self.config = config or MultiQueryConfig()

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a search-query expansion component for a RAG system.

Generate multiple alternative search queries for the user's
original question.

Rules:
1. Generate exactly the requested number of queries.
2. Each query must represent the same information need.
3. Use different wording or perspectives.
4. Keep queries suitable for semantic and keyword retrieval.
5. Do not answer the question.
6. Do not add facts that are not present in the user's question.
7. Do not generate duplicate queries.
8. Return ONLY valid JSON.
9. The JSON must have exactly this structure:

{{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}}

Do not include markdown fences.
Do not include explanations before or after the JSON.
""".strip(),
                ),
                (
                    "human",
                    """
Original user query:
{query}

Generate exactly {num_queries} alternative retrieval queries.
""".strip(),
                ),
            ]
        )

        self._llm: ChatOllama | None = None
        self._llm_lock = threading.Lock()

        logger.info(
            "Multi-query generator initialized."
        )

    # ========================================================
    # LLM
    # ========================================================

    def _load_llm(self) -> ChatOllama:

        if self._llm is not None:
            return self._llm

        with self._llm_lock:

            if self._llm is not None:
                return self._llm

            logger.info(
                "Initializing multi-query LLM: %s",
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
                    "Failed to initialize multi-query LLM."
                )

                raise MultiQueryError(
                    "Unable to initialize multi-query LLM."
                ) from exc

        return self._llm

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise MultiQueryInputError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise MultiQueryInputError(
                "Query cannot be empty."
            )

        return query

    # ========================================================
    # JSON EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_json(response: str) -> str:

        response = response.strip()

        # Remove accidental markdown code fences.
        response = re.sub(
            r"^```(?:json)?\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"\s*```$",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = response.strip()

        # If the model added text around the JSON,
        # try to extract the JSON object.
        if not response.startswith("{"):

            start = response.find("{")

            if start != -1:
                response = response[start:]

        if not response.endswith("}"):

            end = response.rfind("}")

            if end != -1:
                response = response[: end + 1]

        return response

    # ========================================================
    # JSON PARSING
    # ========================================================

    def _parse_response(
        self,
        response: str,
    ) -> list[str]:

        response = self._extract_json(response)

        try:

            data = json.loads(response)

        except json.JSONDecodeError as exc:

            logger.error(
                "Invalid JSON returned by multi-query model: %s",
                response,
            )

            raise MultiQueryError(
                "Multi-query model returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):

            raise MultiQueryError(
                "Multi-query response must be a JSON object."
            )

        queries = data.get("queries")

        if not isinstance(queries, list):

            raise MultiQueryError(
                "Multi-query response must contain a "
                "'queries' list."
            )

        cleaned_queries: list[str] = []

        for generated_query in queries:

            if not isinstance(generated_query, str):
                continue

            generated_query = generated_query.strip()

            if generated_query:
                cleaned_queries.append(
                    generated_query
                )

        # ----------------------------------------------------
        # Remove duplicates while preserving order.
        # ----------------------------------------------------

        unique_queries = list(
            dict.fromkeys(cleaned_queries)
        )

        if len(unique_queries) != self.config.num_queries:

            raise MultiQueryError(
                f"Expected exactly "
                f"{self.config.num_queries} unique queries, "
                f"but received {len(unique_queries)}."
            )

        return unique_queries

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        query: str,
    ) -> MultiQueryResult:

        query = self._validate_query(query)

        llm = self._load_llm()

        # ----------------------------------------------------
        # LCEL chain
        # ----------------------------------------------------

        chain = (
            self.prompt
            | llm
            | StrOutputParser()
        )

        logger.info(
            "Generating %d alternative queries.",
            self.config.num_queries,
        )

        try:

            response = chain.invoke(
                {
                    "query": query,
                    "num_queries": self.config.num_queries,
                }
            )

        except Exception as exc:

            logger.exception(
                "Multi-query generation failed."
            )

            raise MultiQueryError(
                "Failed to generate alternative queries."
            ) from exc

        queries = self._parse_response(
            response
        )

        logger.info(
            "Generated %d unique retrieval queries.",
            len(queries),
        )

        return MultiQueryResult(
            original_query=query,
            queries=tuple(queries),
        )

    # ========================================================
    # STRING-ONLY API
    # ========================================================

    def generate_queries(
        self,
        query: str,
    ) -> list[str]:

        result = self.generate(query)

        return result.all_queries

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None