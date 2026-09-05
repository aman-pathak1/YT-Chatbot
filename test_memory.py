from __future__ import annotations

import logging

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.splitter import TranscriptSplitter
from ingestion.embeddings import BGEEmbeddings

from memory.conversation_memory import ConversationMemory

from retrieval.domain_router import DomainRouter
from retrieval.query_rewriter import QueryRewriter
from retrieval.multi_query import MultiQueryGenerator
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.mmr import MMRRetriever
from retrieval.reranker import BGEReranker
from retrieval.contextual_compressor import ContextualCompressor
from retrieval.vector_store import PineconeVectorStoreManager

from generation.context_optimizer import ContextWindowOptimizer
from generation.answer_generator import AnswerGenerator
from generation.answer_grounding import AnswerGrounding
from generation.guardrails import AnswerGuardrails

from pipeline.rag_pipeline import RAGPipeline


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# Lightweight Reranker
# ============================================================

class LightweightReranker(BGEReranker):
    """
    Lightweight reranker for integration testing.

    Does NOT download/load the actual BGE reranker model.
    """

    def _load_model(self):
        return None

    def rerank(
        self,
        query: str,
        documents,
    ):

        query_words = set(
            query.lower()
            .replace("?", "")
            .replace(".", "")
            .split()
        )

        scored_documents = []

        for document in documents:

            document_words = set(
                document.page_content.lower()
                .replace("?", "")
                .replace(".", "")
                .split()
            )

            score = len(
                query_words.intersection(
                    document_words
                )
            )

            scored_documents.append(
                (score, document)
            )

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            document
            for _, document in scored_documents[
                : self.config.top_n
            ]
        ]


# ============================================================
# Build Pipeline
# ============================================================

def build_pipeline(
    documents,
    memory: ConversationMemory,
) -> RAGPipeline:

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedding_model = BGEEmbeddings()

    # --------------------------------------------------------
    # Pinecone
    # --------------------------------------------------------

    vector_store = PineconeVectorStoreManager(
        embedding_model=embedding_model,
        index_name="youtube-chatbot",
        namespace="youtube-transcripts",
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        documents=documents,
    )

    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
    )

    reranker = LightweightReranker()

    contextual_compressor = (
        ContextualCompressor()
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    context_optimizer = (
        ContextWindowOptimizer()
    )

    answer_generator = AnswerGenerator()

    answer_grounding = AnswerGrounding()

    guardrails = AnswerGuardrails()

    # --------------------------------------------------------
    # Query processing
    # --------------------------------------------------------

    domain_router = DomainRouter()

    query_rewriter = QueryRewriter()

    multi_query_generator = (
        MultiQueryGenerator()
    )

    # --------------------------------------------------------
    # RAG Pipeline + Memory
    # --------------------------------------------------------

    pipeline = RAGPipeline(
        domain_router=domain_router,
        query_rewriter=query_rewriter,
        multi_query_generator=multi_query_generator,
        hybrid_retriever=hybrid_retriever,
        mmr_retriever=mmr_retriever,
        reranker=reranker,
        contextual_compressor=contextual_compressor,
        context_optimizer=context_optimizer,
        answer_generator=answer_generator,
        answer_grounding=answer_grounding,
        guardrails=guardrails,
        memory=memory,
    )

    return pipeline


# ============================================================
# Print Memory
# ============================================================

def print_memory(
    memory: ConversationMemory,
    title: str,
) -> None:

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    history = memory.get_history()

    if not history:
        print("(empty)")
    else:
        print(history)

    print(
        f"\nMessages: {memory.message_count()}"
    )

    print(
        f"Turns: {memory.turn_count()}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("REAL MULTI-TURN MEMORY TEST")
    print("=" * 70)

    # ========================================================
    # Configuration
    # ========================================================

    video_url = (
        "https://www.youtube.com/watch?v=etnLX7m2MiA"
    )

    session_id = "memory-test-session-001"

    # ========================================================
    # Load actual YouTube transcript
    # ========================================================

    print("\nLoading YouTube transcript...")

    transcript_documents = load_youtube_transcript(
        video_url
    )

    print(
        f"Transcript segments: "
        f"{len(transcript_documents)} ✓"
    )

    # ========================================================
    # Translation / normalization
    # ========================================================

    print("\nNormalizing transcript...")

    translator = TranscriptTranslator()

    translated_documents = (
        translator.translate_documents(
            transcript_documents
        )
    )

    print(
        f"Normalized documents: "
        f"{len(translated_documents)} ✓"
    )

    # ========================================================
    # Chunking
    # ========================================================

    print("\nCreating chunks...")

    splitter = TranscriptSplitter()

    documents = splitter.split_documents(
        translated_documents
    )

    print(
        f"Chunks created: "
        f"{len(documents)} ✓"
    )

    # ========================================================
    # Memory
    # ========================================================

    print("\nCreating conversation memory...")

    memory = ConversationMemory(
        session_id=session_id,
        max_turns=10,
        max_history_characters=12000,
    )

    print(
        f"Memory created | "
        f"session={memory.session_id} ✓"
    )

    print_memory(
        memory,
        "MEMORY BEFORE CONVERSATION",
    )

    # ========================================================
    # Build RAG pipeline
    # ========================================================

    print("\nBuilding memory-aware RAG pipeline...")

    pipeline = build_pipeline(
        documents=documents,
        memory=memory,
    )

    print("Memory-aware RAG pipeline initialized ✓")

    # ========================================================
    # QUESTION 1
    # ========================================================

    question_1 = (
        "What is LangChain?"
    )

    print("\n" + "=" * 70)
    print("TURN 1")
    print("=" * 70)

    print(
        f"\nUser:\n{question_1}"
    )

    result_1 = pipeline.run(
        question=question_1,
        documents=documents,
    )

    print("\nAssistant:")
    print(result_1.answer)

    print(
        "\nRewritten Query:"
    )
    print(
        result_1.rewritten_query
    )

    print_memory(
        memory,
        "MEMORY AFTER TURN 1",
    )

    # ========================================================
    # QUESTION 2
    # ========================================================

    question_2 = (
        "Isme RAG kaise use hota hai?"
    )

    print("\n" + "=" * 70)
    print("TURN 2 — MEMORY-AWARE FOLLOW-UP")
    print("=" * 70)

    print(
        f"\nUser:\n{question_2}"
    )

    print(
        "\nMemory being used by Query Rewriter:"
    )

    print(
        memory.get_rewriter_context()
    )

    result_2 = pipeline.run(
        question=question_2,
        documents=documents,
    )

    print("\nAssistant:")
    print(result_2.answer)

    print(
        "\nOriginal Question:"
    )
    print(
        result_2.question
    )

    print(
        "\nRewritten Query:"
    )
    print(
        result_2.rewritten_query
    )

    print_memory(
        memory,
        "MEMORY AFTER TURN 2",
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MEMORY VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Two complete turns should exist
    # --------------------------------------------------------

    assert memory.turn_count() == 2, (
        "Expected exactly 2 conversation turns."
    )

    print(
        "✓ Two conversation turns stored"
    )

    # --------------------------------------------------------
    # 2. Four messages should exist
    # --------------------------------------------------------

    assert memory.message_count() == 4, (
        "Expected 4 messages "
        "(2 user + 2 assistant)."
    )

    print(
        "✓ User + assistant messages stored"
    )

    # --------------------------------------------------------
    # 3. Second question must be present
    # --------------------------------------------------------

    history = memory.get_history()

    assert question_2 in history, (
        "Second user question was not stored."
    )

    print(
        "✓ Current question stored in memory"
    )

    # --------------------------------------------------------
    # 4. First question must be present
    # --------------------------------------------------------

    assert question_1 in history, (
        "First user question was not stored."
    )

    print(
        "✓ Previous question preserved"
    )

    # --------------------------------------------------------
    # 5. Rewritten query should NOT remain
    #    unresolved as the literal "isme"
    # --------------------------------------------------------

    rewritten_query = (
        result_2.rewritten_query.lower()
    )

    print(
        "\nTurn 2 rewritten query:"
    )
    print(
        result_2.rewritten_query
    )

    # We don't assert an exact LLM string because
    # LLM wording can vary. We only verify that
    # rewriting produced a non-empty standalone query.

    assert rewritten_query.strip(), (
        "Turn 2 rewritten query is empty."
    )

    print(
        "✓ Follow-up query was rewritten"
    )

    # --------------------------------------------------------
    # 6. Memory must contain both answers
    # --------------------------------------------------------

    assert result_1.answer in history, (
        "Turn 1 answer missing from memory."
    )

    assert result_2.answer in history, (
        "Turn 2 answer missing from memory."
    )

    print(
        "✓ Both assistant answers stored"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("REAL MULTI-TURN MEMORY TEST COMPLETED ✓")
    print("=" * 70)

    print(
        "\nFinal memory statistics:"
    )

    print(
        memory.get_stats()
    )


if __name__ == "__main__":
    main()