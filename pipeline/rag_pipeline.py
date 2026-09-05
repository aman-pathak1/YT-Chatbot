from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from memory.conversation_memory import ConversationMemory

from retrieval.domain_router import DomainRouter
from retrieval.query_rewriter import QueryRewriter
from retrieval.multi_query import MultiQueryGenerator
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.mmr import MMRRetriever
from retrieval.reranker import BGEReranker
from retrieval.contextual_compressor import ContextualCompressor

from generation.context_optimizer import ContextWindowOptimizer
from generation.answer_generator import AnswerGenerator
from generation.answer_grounding import AnswerGrounding
from generation.guardrails import AnswerGuardrails


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""


class RAGPipelineInputError(RAGPipelineError):
    """Raised when pipeline input is invalid."""


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class RAGPipelineResult:
    """
    Final result returned by the RAG pipeline.
    """

    question: str
    answer: str
    domain: str
    rewritten_query: str
    generated_queries: tuple[str, ...]
    retrieved_documents: tuple[Document, ...]
    reranked_documents: tuple[Document, ...]
    compressed_documents: tuple[Document, ...]
    optimized_documents: tuple[Document, ...]
    grounding_score: float
    is_grounded: bool
    guardrails_passed: bool

    @property
    def documents(self) -> list[Document]:
        return list(self.optimized_documents)


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Production-ready end-to-end RAG orchestration layer.

    Flow:

        Conversation Memory
            ↓
        Question
            ↓
        Domain Router
            ↓
        Context-Aware Query Rewriter
            ↓
        Multi Query
            ↓
        Hybrid Retrieval
            ↓
        MMR
            ↓
        Reranker
            ↓
        Contextual Compression
            ↓
        Context Window Optimization
            ↓
        Answer Generation
            ↓
        Answer Grounding
            ↓
        Guardrails
            ↓
        Memory Update
            ↓
        Final Answer
    """

    def __init__(
        self,
        domain_router: DomainRouter,
        query_rewriter: QueryRewriter,
        multi_query_generator: MultiQueryGenerator,
        hybrid_retriever: HybridRetriever,
        mmr_retriever: MMRRetriever,
        reranker: BGEReranker,
        contextual_compressor: ContextualCompressor,
        context_optimizer: ContextWindowOptimizer,
        answer_generator: AnswerGenerator,
        answer_grounding: AnswerGrounding,
        guardrails: AnswerGuardrails,
        memory: ConversationMemory | None = None,
    ) -> None:

        self.domain_router = domain_router
        self.query_rewriter = query_rewriter
        self.multi_query_generator = multi_query_generator
        self.hybrid_retriever = hybrid_retriever
        self.mmr_retriever = mmr_retriever
        self.reranker = reranker
        self.contextual_compressor = contextual_compressor
        self.context_optimizer = context_optimizer
        self.answer_generator = answer_generator
        self.answer_grounding = answer_grounding
        self.guardrails = guardrails

        # Optional conversation memory.
        self.memory = memory

        logger.info(
            "RAG pipeline initialized | memory=%s",
            memory is not None,
        )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:

        if not isinstance(question, str):
            raise RAGPipelineInputError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise RAGPipelineInputError(
                "Question cannot be empty."
            )

        return question

    @staticmethod
    def _validate_documents(
        documents: list[Document],
    ) -> list[Document]:

        if not isinstance(documents, list):
            raise RAGPipelineInputError(
                "Documents must be provided as a list."
            )

        valid_documents = [
            document
            for document in documents
            if isinstance(document, Document)
            and document.page_content.strip()
        ]

        if not valid_documents:
            raise RAGPipelineInputError(
                "At least one valid document is required."
            )

        return valid_documents

    # ========================================================
    # MEMORY
    # ========================================================

    def _get_conversation_history(self) -> str:
        """
        Get bounded conversation history for query rewriting.

        If memory is disabled, return empty history.
        """

        if self.memory is None:
            return ""

        try:

            history = self.memory.get_rewriter_context()

            logger.debug(
                "Conversation history loaded | "
                "session=%s | characters=%d",
                self.memory.session_id,
                len(history),
            )

            return history

        except Exception as exc:

            logger.exception(
                "Failed to read conversation memory."
            )

            raise RAGPipelineError(
                "Failed to read conversation memory."
            ) from exc

    def _save_conversation_turn(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Save a successful conversation turn.

        Memory failure must not fail an otherwise successful
        RAG response.
        """

        if self.memory is None:
            return

        try:

            self.memory.add_turn(
                user_message=question,
                assistant_message=answer,
            )

            logger.debug(
                "Conversation turn saved | "
                "session=%s | turns=%d",
                self.memory.session_id,
                self.memory.turn_count(),
            )

        except Exception:

            logger.exception(
                "Failed to update conversation memory."
            )

    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def _retrieve_for_queries(
        self,
        queries: list[str],
        documents: list[Document],
    ) -> list[Document]:

        retrieved_documents: list[Document] = []

        for query in queries:

            results = self.hybrid_retriever.retrieve(
                query=query,
            )

            retrieved_documents.extend(results)

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

        unique_documents: list[Document] = []
        seen: set[str] = set()

        for document in retrieved_documents:

            key = (
                document.metadata.get("video_id"),
                document.metadata.get("chunk_index"),
                document.page_content.strip(),
            )

            key_string = repr(key)

            if key_string in seen:
                continue

            seen.add(key_string)
            unique_documents.append(document)

        logger.info(
            "Retrieved %d unique documents from %d queries.",
            len(unique_documents),
            len(queries),
        )

        return unique_documents

    # ========================================================
    # CONTEXTUAL COMPRESSION
    # ========================================================

    def _compress_context(
        self,
        query: str,
        reranked_documents: list[Document],
    ) -> list[Document]:
        """
        Perform contextual compression.

        Production fallback:

        If the compressor removes every document, use the
        reranked documents instead of crashing the pipeline.

        Why?

        Contextual compression is an optimization stage.
        It should not become a single point of failure for
        an otherwise valid retrieval result.
        """

        compressed_documents = (
            self.contextual_compressor.compress(
                query=query,
                documents=reranked_documents,
            )
        )

        # ----------------------------------------------------
        # Normal case
        # ----------------------------------------------------

        if compressed_documents:

            logger.info(
                "Contextual compression retained %d/%d documents.",
                len(compressed_documents),
                len(reranked_documents),
            )

            return compressed_documents

        # ----------------------------------------------------
        # Fallback case
        # ----------------------------------------------------

        logger.warning(
            "Contextual compression removed all %d documents. "
            "Falling back to reranked documents.",
            len(reranked_documents),
        )

        return reranked_documents

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(
        self,
        question: str,
        documents: list[Document],
    ) -> RAGPipelineResult:

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        question = self._validate_question(
            question
        )

        documents = self._validate_documents(
            documents
        )

        logger.info(
            "Starting RAG pipeline."
        )

        # ====================================================
        # 0. LOAD CONVERSATION MEMORY
        # ====================================================

        conversation_history = (
            self._get_conversation_history()
        )

        if conversation_history:

            logger.info(
                "Using conversation memory for "
                "context-aware query rewriting."
            )

        else:

            logger.info(
                "No previous conversation context."
            )

        # ====================================================
        # 1. DOMAIN ROUTING
        # ====================================================

        domain_result = self.domain_router.route(
            question
        )

        logger.info(
            "Domain: %s",
            domain_result.domain,
        )

        # ====================================================
        # 2. CONTEXT-AWARE QUERY REWRITING
        # ====================================================

        rewritten_result = (
            self.query_rewriter.rewrite(
                query=question,
                conversation_history=conversation_history,
            )
        )

        rewritten_query = (
            rewritten_result.rewritten_query
        )

        logger.info(
            "Rewritten query: %s",
            rewritten_query,
        )

        # ====================================================
        # 3. MULTI QUERY GENERATION
        # ====================================================

        multi_query_result = (
            self.multi_query_generator.generate(
                rewritten_query
            )
        )

        generated_queries = list(
            multi_query_result.queries
        )

        # Always keep rewritten query.
        if rewritten_query not in generated_queries:

            generated_queries.insert(
                0,
                rewritten_query,
            )

        logger.info(
            "Generated %d retrieval queries.",
            len(generated_queries),
        )

        # ====================================================
        # 4. HYBRID RETRIEVAL
        # ====================================================

        retrieved_documents = (
            self._retrieve_for_queries(
                queries=generated_queries,
                documents=documents,
            )
        )

        if not retrieved_documents:

            raise RAGPipelineError(
                "No documents were retrieved."
            )

        # ====================================================
        # 5. MMR
        # ====================================================

        mmr_documents = (
            self.mmr_retriever.retrieve(
                query=rewritten_query,
                documents=retrieved_documents,
            )
        )

        if not mmr_documents:

            raise RAGPipelineError(
                "MMR retrieval returned no documents."
            )

        # ====================================================
        # 6. RERANKING
        # ====================================================

        reranked_documents = (
            self.reranker.rerank(
                query=rewritten_query,
                documents=mmr_documents,
            )
        )

        if not reranked_documents:

            raise RAGPipelineError(
                "Reranker returned no documents."
            )

        # ====================================================
        # 7. CONTEXTUAL COMPRESSION
        # ====================================================

        compressed_documents = (
            self._compress_context(
                query=rewritten_query,
                reranked_documents=reranked_documents,
            )
        )

        if not compressed_documents:

            # This should practically never happen because
            # _compress_context() falls back to reranked docs.
            raise RAGPipelineError(
                "No context available after compression fallback."
            )

        # ====================================================
        # 8. CONTEXT WINDOW OPTIMIZATION
        # ====================================================

        optimization_result = (
            self.context_optimizer.optimize(
                compressed_documents
            )
        )

        optimized_documents = (
            optimization_result.optimized_documents
        )

        if not optimized_documents:

            raise RAGPipelineError(
                "Context optimization produced no context."
            )

        # ====================================================
        # 9. ANSWER GENERATION
        # ====================================================

        answer = self.answer_generator.generate(
            question=question,
            documents=optimized_documents,
        )

        if not answer or not answer.strip():

            raise RAGPipelineError(
                "Answer generator returned an empty answer."
            )

        # ====================================================
        # 10. ANSWER GROUNDING
        # ====================================================

        grounding_result = (
            self.answer_grounding.evaluate(
                answer=answer,
                documents=optimized_documents,
            )
        )

        # ====================================================
        # 11. GUARDRAILS
        # ====================================================

        guardrail_result = (
            self.guardrails.validate(
                answer=answer,
                documents=optimized_documents,
            )
        )

        final_answer = answer

        if not guardrail_result.is_valid:

            logger.warning(
                "Answer failed guardrails: %s",
                guardrail_result.reasons,
            )

            final_answer = (
                self.guardrails.validate_or_fallback(
                    answer=answer,
                    documents=optimized_documents,
                )
            )

        # ====================================================
        # 12. SAVE SUCCESSFUL TURN TO MEMORY
        # ====================================================

        self._save_conversation_turn(
            question=question,
            answer=final_answer,
        )

        # ====================================================
        # PIPELINE COMPLETE
        # ====================================================

        logger.info(
            "RAG pipeline completed. "
            "grounded=%s score=%.2f",
            grounding_result.is_grounded,
            grounding_result.grounding_score,
        )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return RAGPipelineResult(

            question=question,

            answer=final_answer,

            domain=domain_result.domain,

            rewritten_query=rewritten_query,

            generated_queries=tuple(
                generated_queries
            ),

            retrieved_documents=tuple(
                retrieved_documents
            ),

            reranked_documents=tuple(
                reranked_documents
            ),

            compressed_documents=tuple(
                compressed_documents
            ),

            optimized_documents=tuple(
                optimized_documents
            ),

            grounding_score=(
                grounding_result.grounding_score
            ),

            is_grounded=(
                grounding_result.is_grounded
            ),

            guardrails_passed=(
                guardrail_result.is_valid
            ),
        )

    def run_stream(
        self,
        question: str,
        documents: list[Document],
    ):
        """
        Stream answer tokens directly from LLM generation.
        Returns a tuple of (token_generator, result_fetcher_callable).
        """
        question = self._validate_question(question)
        documents = self._validate_documents(documents)

        conversation_history = self._get_conversation_history()
        domain_result = self.domain_router.route(question)
        rewritten_result = self.query_rewriter.rewrite(
            query=question,
            conversation_history=conversation_history,
        )
        rewritten_query = rewritten_result.rewritten_query

        multi_query_result = self.multi_query_generator.generate(rewritten_query)
        generated_queries = list(multi_query_result.queries)
        if rewritten_query not in generated_queries:
            generated_queries.insert(0, rewritten_query)

        retrieved_documents = self._retrieve_for_queries(
            queries=generated_queries,
            documents=documents,
        )
        if not retrieved_documents:
            raise RAGPipelineError("No documents were retrieved.")

        mmr_documents = self.mmr_retriever.retrieve(
            query=rewritten_query,
            documents=retrieved_documents,
        )
        if not mmr_documents:
            raise RAGPipelineError("MMR retrieval returned no documents.")

        reranked_documents = self.reranker.rerank(
            query=rewritten_query,
            documents=mmr_documents,
        )
        if not reranked_documents:
            raise RAGPipelineError("Reranker returned no documents.")

        compressed_documents = self._compress_context(
            query=rewritten_query,
            reranked_documents=reranked_documents,
        )

        optimization_result = self.context_optimizer.optimize(compressed_documents)
        optimized_documents = optimization_result.optimized_documents

        # Generator for tokens
        full_answer_chunks = []
        for chunk in self.answer_generator.generate_stream(
            question=question,
            documents=optimized_documents,
        ):
            full_answer_chunks.append(chunk)
            yield chunk

        full_answer = "".join(full_answer_chunks).strip()
        self._save_conversation_turn(question=question, answer=full_answer)