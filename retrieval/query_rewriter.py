from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================

class QueryRewriterError(Exception):
    """Base exception for query rewriting errors."""


class QueryValidationError(QueryRewriterError):
    """Raised when the user query is invalid."""


class QueryParsingError(QueryRewriterError):
    """Raised when the LLM response cannot be parsed."""


# ============================================================
# Structured Query Result
# ============================================================

@dataclass(frozen=True)
class RewrittenQuery:
    """
    Structured result produced by the query rewriting stage.

    Pipeline:

        Conversation Memory
              +
        Current User Query
              ↓
        Query Rewriter
              ↓
        RewrittenQuery
              ↓
        Multi-Query Generator
    """

    original_query: str
    rewritten_query: str
    language: str
    intent: str
    keywords: tuple[str, ...]

    def __post_init__(self) -> None:

        if not self.original_query.strip():
            raise ValueError(
                "original_query cannot be empty."
            )

        if not self.rewritten_query.strip():
            raise ValueError(
                "rewritten_query cannot be empty."
            )

    @property
    def primary_query(self) -> str:
        return self.rewritten_query


# ============================================================
# Query Rewriter
# ============================================================

class QueryRewriter:
    """
    Production-oriented context-aware query rewriting component.

    Responsibilities:

        1. Validate current user query
        2. Understand conversation context
        3. Resolve obvious references from previous turns
        4. Understand query language
        5. Identify user intent
        6. Extract important keywords
        7. Rewrite query for retrieval

    Does NOT:

        - Generate multiple queries
        - Retrieve documents
        - Perform reranking
        - Answer the question
        - Store conversation memory

    Conversation memory is supplied externally.
    """

    DEFAULT_MODEL = "qwen3:8b"

    SUPPORTED_LANGUAGES = {
        "english",
        "hindi",
        "hinglish",
        "mixed",
        "other",
    }

    SUPPORTED_INTENTS = {
        "factual",
        "explanation",
        "comparison",
        "summary",
        "procedural",
        "definition",
        "causal",
        "other",
    }

    MAX_QUERY_LENGTH = 4000
    MAX_HISTORY_LENGTH = 12000

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        num_predict: int = 256,
    ) -> None:

        if not isinstance(model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        if temperature < 0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if num_predict <= 0:
            raise ValueError(
                "num_predict must be greater than zero."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
            reasoning=False,
        )

        # ----------------------------------------------------
        # Context-aware Prompt
        # ----------------------------------------------------

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the query rewriting component of a
production-grade YouTube RAG system.

The system retrieves relevant passages from
YouTube video transcripts.

Your task is ONLY to rewrite the CURRENT USER
QUESTION into ONE precise, standalone,
retrieval-optimized query.

You must NOT answer the question.

You are given:

1. Previous conversation history.
2. The current user question.

Use previous conversation ONLY when it is
necessary to understand the current question.

IMPORTANT CONVERSATION RULES:

1. Resolve obvious references such as:
   - this
   - that
   - it
   - this topic
   - this video
   - previous one
   - isme
   - usme
   - ye
   - woh
   - above
   - previous answer

2. If the current question is already
   self-contained, do NOT unnecessarily
   inject information from conversation history.

3. Never invent information that is not present
   in the current question or conversation history.

4. Do not treat assistant statements as verified
   external knowledge.

5. Preserve the user's actual information need.

6. The rewritten query must be understandable
   WITHOUT needing the conversation history.

7. Do not answer the question.

Before rewriting, internally understand:

1. The language of the current query.
2. The user's information intent.
3. Relevant entities/concepts from the conversation.
4. Whether previous context is actually required.

Language must be one of:

- english
- hindi
- hinglish
- mixed
- other

Intent must be one of:

- factual
- explanation
- comparison
- summary
- procedural
- definition
- causal
- other

Rewriting rules:

1. Preserve the exact information need.
2. Remove conversational filler.
3. Preserve important technical terms.
4. Preserve important names and entities.
5. Resolve obvious conversational ambiguity.
6. Do not invent information.
7. Do not add unsupported assumptions.
8. Keep the rewritten query concise.
9. Make the query suitable for semantic retrieval.
10. Generate exactly ONE query.
11. Do not generate alternative queries.
12. Do not generate multiple queries.
13. Do not answer the user's question.
14. Make the rewritten query standalone.

For Hindi/Hinglish queries:

- Understand the intended meaning first.
- Do not blindly translate every word.
- Preserve technical terms such as RAG, LangChain,
  embeddings, vector database, Pinecone, etc.
- The rewritten query can be in English when that
  makes semantic retrieval clearer.

Return ONLY valid JSON.

Required format:

{{
    "language": "english | hindi | hinglish | mixed | other",
    "intent": "factual | explanation | comparison | summary | procedural | definition | causal | other",
    "rewritten_query": "single standalone optimized retrieval query",
    "keywords": [
        "important keyword 1",
        "important keyword 2"
    ]
}}

IMPORTANT:

- `rewritten_query` must contain exactly ONE query.
- The rewritten query must be standalone.
- Do not return `search_queries`.
- Do not return explanations.
- Do not use markdown.
""",
                ),
                (
                    "human",
                    """
Previous conversation:
{conversation_history}

Current user query:
{query}
""",
                ),
            ]
        )

        # ----------------------------------------------------
        # LCEL Chain
        # ----------------------------------------------------

        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ========================================================
    # Query Validation
    # ========================================================

    @classmethod
    def _validate_query(
        cls,
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise QueryValidationError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise QueryValidationError(
                "Query cannot be empty."
            )

        if len(query) > cls.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                "Query exceeds maximum allowed length "
                f"of {cls.MAX_QUERY_LENGTH} characters."
            )

        return query

    # ========================================================
    # Conversation History Validation
    # ========================================================

    @classmethod
    def _validate_history(
        cls,
        conversation_history: str | None,
    ) -> str:

        if conversation_history is None:
            return ""

        if not isinstance(
            conversation_history,
            str,
        ):
            raise QueryValidationError(
                "conversation_history must be a string."
            )

        history = conversation_history.strip()

        if not history:
            return ""

        # Safety/context-window boundary.
        if len(history) > cls.MAX_HISTORY_LENGTH:
            history = history[
                -cls.MAX_HISTORY_LENGTH:
            ]

        return history

    # ========================================================
    # JSON Extraction
    # ========================================================

    @staticmethod
    def _extract_json(
        response: str,
    ) -> dict[str, Any]:

        response = response.strip()

        if not response:
            raise QueryParsingError(
                "LLM returned an empty response."
            )

        # ----------------------------------------------------
        # Direct JSON
        # ----------------------------------------------------

        try:

            parsed = json.loads(response)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------
        # JSON surrounded by text
        # ----------------------------------------------------

        match = re.search(
            r"\{.*\}",
            response,
            flags=re.DOTALL,
        )

        if not match:
            raise QueryParsingError(
                "No valid JSON object found "
                "in LLM response."
            )

        try:

            parsed = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError as exc:

            raise QueryParsingError(
                "LLM returned malformed JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise QueryParsingError(
                "Expected JSON object."
            )

        return parsed

    # ========================================================
    # Output Validation
    # ========================================================

    def _validate_output(
        self,
        data: dict[str, Any],
        original_query: str,
    ) -> RewrittenQuery:

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        language = str(
            data.get(
                "language",
                "other",
            )
        ).strip().lower()

        if language not in self.SUPPORTED_LANGUAGES:

            logger.warning(
                "Unknown language returned by LLM: %s",
                language,
            )

            language = "other"

        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        intent = str(
            data.get(
                "intent",
                "other",
            )
        ).strip().lower()

        if intent not in self.SUPPORTED_INTENTS:

            logger.warning(
                "Unknown intent returned by LLM: %s",
                intent,
            )

            intent = "other"

        # ----------------------------------------------------
        # Rewritten Query
        # ----------------------------------------------------

        rewritten_query = str(
            data.get(
                "rewritten_query",
                "",
            )
        ).strip()

        if not rewritten_query:

            raise QueryParsingError(
                "LLM did not return rewritten_query."
            )

        # ----------------------------------------------------
        # Keywords
        # ----------------------------------------------------

        raw_keywords = data.get(
            "keywords",
            [],
        )

        keywords: list[str] = []

        if isinstance(
            raw_keywords,
            list,
        ):

            seen: set[str] = set()

            for keyword in raw_keywords:

                if not isinstance(
                    keyword,
                    str,
                ):
                    continue

                keyword = keyword.strip()

                normalized = keyword.casefold()

                if (
                    keyword
                    and normalized not in seen
                ):

                    keywords.append(keyword)

                    seen.add(normalized)

        return RewrittenQuery(
            original_query=original_query,
            rewritten_query=rewritten_query,
            language=language,
            intent=intent,
            keywords=tuple(keywords),
        )

    # ========================================================
    # Rewrite
    # ========================================================

    def rewrite(
        self,
        query: str,
        conversation_history: str = "",
    ) -> RewrittenQuery:
        """
        Rewrite the current user query using optional
        conversation history.

        Backward compatible:

            rewrite(query)

        Memory-aware:

            rewrite(
                query,
                conversation_history=history,
            )
        """

        original_query = self._validate_query(
            query
        )

        history = self._validate_history(
            conversation_history
        )

        try:

            response = self.chain.invoke(
                {
                    "query": original_query,
                    "conversation_history": history,
                }
            )

            data = self._extract_json(
                response
            )

            result = self._validate_output(
                data=data,
                original_query=original_query,
            )

        except QueryRewriterError:
            raise

        except Exception as exc:

            logger.exception(
                "Query rewriting failed."
            )

            raise QueryRewriterError(
                "Failed to rewrite user query."
            ) from exc

        logger.info(
            "Query rewritten successfully | "
            "history=%s | language=%s | intent=%s | "
            "original='%s' | rewritten='%s'",
            bool(history),
            result.language,
            result.intent,
            result.original_query,
            result.rewritten_query,
        )

        return result

    # ========================================================
    # Simple API
    # ========================================================

    def rewrite_query(
        self,
        query: str,
        conversation_history: str = "",
    ) -> str:
        """
        Return only the rewritten query.
        """

        result = self.rewrite(
            query=query,
            conversation_history=conversation_history,
        )

        return result.rewritten_query