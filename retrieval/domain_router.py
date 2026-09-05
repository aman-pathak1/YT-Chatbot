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


class DomainRoutingError(Exception):
    """Base exception for domain routing errors."""


class DomainRouterConfigurationError(DomainRoutingError):
    """Raised when domain router configuration is invalid."""


class DomainRouterInputError(DomainRoutingError):
    """Raised when the input query is invalid."""


@dataclass(frozen=True)
class DomainRouterConfig:
    model_name: str = "qwen3:8b"
    temperature: float = 0.0
    max_tokens: int = 256
    use_fast_routing: bool = True

    allowed_domains: tuple[str, ...] = (
        "technical",
        "programming",
        "machine_learning",
        "data_science",
        "business",
        "education",
        "general",
        "other",
    )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise DomainRouterConfigurationError(
                "model_name cannot be empty."
            )

        if self.temperature < 0:
            raise DomainRouterConfigurationError(
                "temperature cannot be negative."
            )

        if self.max_tokens <= 0:
            raise DomainRouterConfigurationError(
                "max_tokens must be greater than 0."
            )

        if not self.allowed_domains:
            raise DomainRouterConfigurationError(
                "allowed_domains cannot be empty."
            )


@dataclass(frozen=True)
class DomainRoutingResult:
    original_query: str
    domain: str
    confidence: float
    reasoning: str

    @property
    def route(self) -> str:
        return self.domain


class DomainRouter:
    """
    Classifies a user query into exactly one domain.

    The router only performs domain classification.
    It does not answer or rewrite the user's query.
    """

    def __init__(
        self,
        config: DomainRouterConfig | None = None,
    ) -> None:

        self.config = config or DomainRouterConfig()

        allowed_domains = ", ".join(
            self.config.allowed_domains
        )

        # IMPORTANT:
        # There is NO literal JSON object in this prompt.
        # Therefore ChatPromptTemplate cannot confuse JSON
        # braces with template variables.

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
You are a domain classification component
inside a production RAG system.

Your ONLY task is to classify the user's query
into exactly ONE domain.

Allowed domains:
{allowed_domains}

Domain meanings:

technical:
Questions about technical concepts, systems,
databases, APIs, architectures, infrastructure,
networking, cloud, or software technology.

programming:
Questions specifically about programming,
coding, algorithms, data structures, debugging,
or implementation.

machine_learning:
Questions about machine learning, deep learning,
neural networks, NLP, computer vision, LLMs,
RAG, embeddings, transformers, or AI models.

data_science:
Questions about statistics, data analysis,
data processing, visualization, or analytics.

business:
Questions about companies, markets, finance,
management, strategy, entrepreneurship, or business.

education:
Questions about learning, teaching, courses,
exams, academic concepts, or study guidance.

general:
General knowledge questions that do not clearly
belong to another domain.

other:
Queries that do not reasonably fit any of the
available domains.

Rules:

1. Select exactly ONE domain.
2. The selected domain must be one of the allowed domains.
3. Choose the domain representing the primary intent.
4. Do NOT answer the user's question.
5. Do NOT rewrite the user's question.
6. Do NOT generate multiple domains.
7. Do NOT invent facts.
8. Confidence must be between 0.0 and 1.0.
9. Reasoning must be short.
10. Return ONLY valid JSON.
11. Do not use markdown.
12. Do not include explanations outside the JSON.

The JSON must contain exactly these three fields:

domain
confidence
reasoning

Example values:
domain = technical
confidence = 0.95
reasoning = The query concerns a technical concept.

Return the result as a JSON object using those fields.
""".strip(),
                ),
                (
                    "human",
                    """
User query:

{query}

Classify this query into exactly ONE allowed domain.
""".strip(),
                ),
            ]
        )

        self._llm: ChatOllama | None = None
        self._llm_lock = threading.Lock()

        logger.info(
            "Domain router initialized."
        )

    def _load_llm(self) -> ChatOllama:
        """
        Lazily initialize the Ollama LLM.
        """

        if self._llm is not None:
            return self._llm

        with self._llm_lock:

            if self._llm is not None:
                return self._llm

            logger.info(
                "Initializing domain router LLM: %s",
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
                    "Failed to initialize domain router LLM."
                )

                raise DomainRoutingError(
                    "Unable to initialize domain router LLM."
                ) from exc

        return self._llm

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate the input query.
        """

        if not isinstance(query, str):
            raise DomainRouterInputError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise DomainRouterInputError(
                "Query cannot be empty."
            )

        return query

    @staticmethod
    def _extract_json(response: str) -> str:
        """
        Extract the JSON object from the model response.

        Handles accidental markdown fences and
        surrounding text.
        """

        response = response.strip()

        # Remove markdown code fences.
        response = re.sub(
            r"```(?:json)?",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = response.replace(
            "```",
            "",
        ).strip()

        # Find JSON object.
        start = response.find("{")
        end = response.rfind("}")

        if start == -1:
            raise DomainRoutingError(
                "No JSON object found in model response."
            )

        if end == -1 or end <= start:
            raise DomainRoutingError(
                "Incomplete JSON object returned by model."
            )

        return response[start : end + 1]

    def _parse_response(
        self,
        response: str,
        original_query: str,
    ) -> DomainRoutingResult:
        """
        Parse and validate the LLM response.
        """

        json_text = self._extract_json(response)

        try:
            data = json.loads(json_text)

        except json.JSONDecodeError as exc:

            logger.error(
                "Invalid JSON returned by domain router: %s",
                response,
            )

            raise DomainRoutingError(
                "Domain router returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise DomainRoutingError(
                "Domain router response must be a JSON object."
            )

        domain = data.get("domain")
        confidence = data.get("confidence")
        reasoning = data.get("reasoning")

        # -------------------------
        # Domain validation
        # -------------------------

        if not isinstance(domain, str):
            raise DomainRoutingError(
                "Domain must be a string."
            )

        domain = domain.strip()

        if domain not in self.config.allowed_domains:
            raise DomainRoutingError(
                f"Invalid domain '{domain}'. "
                f"Allowed domains: "
                f"{self.config.allowed_domains}"
            )

        # -------------------------
        # Confidence validation
        # -------------------------

        if isinstance(confidence, bool):
            raise DomainRoutingError(
                "Confidence must be a number."
            )

        try:
            confidence = float(confidence)

        except (TypeError, ValueError) as exc:

            raise DomainRoutingError(
                "Confidence must be a valid number."
            ) from exc

        if not 0.0 <= confidence <= 1.0:
            raise DomainRoutingError(
                "Confidence must be between 0.0 and 1.0."
            )

        # -------------------------
        # Reasoning validation
        # -------------------------

        if not isinstance(reasoning, str):
            raise DomainRoutingError(
                "Reasoning must be a string."
            )

        reasoning = reasoning.strip()

        if not reasoning:
            raise DomainRoutingError(
                "Reasoning cannot be empty."
            )

        return DomainRoutingResult(
            original_query=original_query,
            domain=domain,
            confidence=confidence,
            reasoning=reasoning,
        )

    def route(
        self,
        query: str,
    ) -> DomainRoutingResult:
        """
        Classify a query into exactly one domain.
        """

        query = self._validate_query(query)

        if self.config.use_fast_routing:
            q_lower = query.lower()
            domain = "general"
            if any(k in q_lower for k in ["code", "python", "function", "bug", "error", "script", "program"]):
                domain = "programming"
            elif any(k in q_lower for k in ["model", "ai", "rag", "embedding", "llm", "neural", "train", "machine learning"]):
                domain = "machine_learning"
            elif any(k in q_lower for k in ["data", "sql", "pandas", "analysis"]):
                domain = "data_science"
            elif any(k in q_lower for k in ["market", "business", "money", "price", "company"]):
                domain = "business"
            elif any(k in q_lower for k in ["learn", "exam", "course", "study"]):
                domain = "education"

            logger.info("Fast domain routing used: domain='%s'", domain)
            return DomainRoutingResult(
                original_query=query,
                domain=domain,
                confidence=1.0,
                reasoning="Fast rule-based domain routing.",
            )

        llm = self._load_llm()

        chain = (
            self.prompt
            | llm
            | StrOutputParser()
        )

        logger.info(
            "Routing query to domain."
        )

        try:

            response = chain.invoke(
                {
                    "query": query,
                }
            )

        except Exception as exc:

            logger.exception(
                "Domain routing failed."
            )

            raise DomainRoutingError(
                "Failed to classify query domain."
            ) from exc

        result = self._parse_response(
            response=response,
            original_query=query,
        )

        logger.info(
            "Query routed to domain='%s' confidence=%.2f",
            result.domain,
            result.confidence,
        )

        return result

    def route_domain(
        self,
        query: str,
    ) -> str:
        """
        Return only the selected domain.
        """

        result = self.route(query)

        return result.domain

    @property
    def is_loaded(self) -> bool:
        """
        Return whether the LLM has been initialized.
        """

        return self._llm is not None