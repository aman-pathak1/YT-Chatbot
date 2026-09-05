from __future__ import annotations

import logging

from langchain_core.documents import Document

from ingestion.embeddings import BGEEmbeddings

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


# ============================================================
# Lightweight Reranker
# ============================================================

class LightweightReranker(BGEReranker):
    """
    Lightweight reranker used only for integration testing.

    The actual BGE reranker model is NOT loaded.
    Therefore, no large Hugging Face reranker model
    is downloaded during this test.
    """

    def _load_model(self):
        return None

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

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
# Test Documents
# ============================================================

def create_test_documents() -> list[Document]:
    """
    Small local corpus used by BM25 and Pinecone.

    These documents represent transcript chunks
    used in the previous component tests.
    """

    return [
        Document(
            page_content=(
                "Hi Guys, My name is Nitesh and "
                "you are welcome to my YouTube channel."
            ),
            metadata={
                "source": "https://youtu.be/etnLX7m2MiA",
                "video_id": "etnLX7m2MiA",
                "chunk_index": 0,
                "chunk_start_time": 0.0,
                "chunk_end_time": 5.0,
            },
        ),

        Document(
            page_content=(
                "To my YouTube channel. In this video "
                "we continue the playlist."
            ),
            metadata={
                "source": "https://youtu.be/etnLX7m2MiA",
                "video_id": "etnLX7m2MiA",
                "chunk_index": 1,
                "chunk_start_time": 5.0,
                "chunk_end_time": 10.0,
            },
        ),

        Document(
            page_content=(
                "People continue their long chain "
                "playlists and learn programming."
            ),
            metadata={
                "source": "https://youtu.be/etnLX7m2MiA",
                "video_id": "etnLX7m2MiA",
                "chunk_index": 2,
                "chunk_start_time": 10.0,
                "chunk_end_time": 15.0,
            },
        ),
    ]


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("REAL RAG PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # ========================================================
    # Test Corpus
    # ========================================================

    documents = create_test_documents()

    question = (
        "What does the video say about YouTube?"
    )

    print("\nQuestion:")
    print(question)

    print(
        f"\nTest corpus documents: "
        f"{len(documents)}"
    )

    # ========================================================
    # REAL COMPONENTS
    # ========================================================

    print(
        "\nInitializing real components..."
    )

    # --------------------------------------------------------
    # 1. Domain Router
    # --------------------------------------------------------

    domain_router = DomainRouter()

    # --------------------------------------------------------
    # 2. Query Rewriter
    # --------------------------------------------------------

    query_rewriter = QueryRewriter()

    # --------------------------------------------------------
    # 3. Multi Query Generator
    # --------------------------------------------------------

    multi_query_generator = (
        MultiQueryGenerator()
    )

    # --------------------------------------------------------
    # 4. BGE-M3 Embedding Model
    # --------------------------------------------------------

    print(
        "\nInitializing BGE-M3 embeddings..."
    )

    embedding_model = BGEEmbeddings()

    print(
        "BGE-M3 embedding model initialized ✓"
    )

    # --------------------------------------------------------
    # 5. Pinecone Vector Store
    # --------------------------------------------------------

    vector_store = PineconeVectorStoreManager(
        embedding_model=embedding_model,
        index_name="youtube-chatbot",
    )

    print(
        "Pinecone vector store initialized ✓"
    )

    # --------------------------------------------------------
    # 6. UPSERT TEST DOCUMENTS INTO PINECONE
    # --------------------------------------------------------

    print(
        "\nUpserting test documents into Pinecone..."
    )

    upserted_ids = vector_store.add_documents(
        documents
    )

    print(
        f"✓ Test documents upserted: "
        f"{len(upserted_ids)}"
    )

    # --------------------------------------------------------
    # 7. Hybrid Retriever
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        documents=documents,
    )

    print(
        "Hybrid retriever initialized ✓"
    )

    # --------------------------------------------------------
    # 8. MMR Retriever
    # --------------------------------------------------------

    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
    )

    print(
        "MMR retriever initialized ✓"
    )

    # --------------------------------------------------------
    # 9. Lightweight Reranker
    # --------------------------------------------------------

    reranker = LightweightReranker()

    print(
        "Lightweight reranker initialized ✓"
    )

    # --------------------------------------------------------
    # 10. Contextual Compression
    # --------------------------------------------------------

    contextual_compressor = (
        ContextualCompressor()
    )

    print(
        "Contextual compressor initialized ✓"
    )

    # --------------------------------------------------------
    # 11. Context Window Optimizer
    # --------------------------------------------------------

    context_optimizer = (
        ContextWindowOptimizer()
    )

    print(
        "Context window optimizer initialized ✓"
    )

    # --------------------------------------------------------
    # 12. Answer Generator
    # --------------------------------------------------------

    answer_generator = (
        AnswerGenerator()
    )

    print(
        "Answer generator initialized ✓"
    )

    # --------------------------------------------------------
    # 13. Answer Grounding
    # --------------------------------------------------------

    answer_grounding = (
        AnswerGrounding()
    )

    print(
        "Answer grounding initialized ✓"
    )

    # --------------------------------------------------------
    # 14. Guardrails
    # --------------------------------------------------------

    guardrails = AnswerGuardrails()

    print(
        "Guardrails initialized ✓"
    )

    # ========================================================
    # BUILD RAG PIPELINE
    # ========================================================

    print(
        "\nBuilding RAG pipeline..."
    )

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
    )

    print(
        "RAG pipeline initialized ✓"
    )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    print(
        "\nRunning pipeline..."
    )

    result = pipeline.run(
        question=question,
        documents=documents,
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "-" * 70)
    print("PIPELINE RESULT")
    print("-" * 70)

    print(
        f"\nDomain: "
        f"{result.domain}"
    )

    print(
        f"Rewritten query: "
        f"{result.rewritten_query}"
    )

    print("\nGenerated queries:")

    for index, query in enumerate(
        result.generated_queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )

    print(
        f"\nRetrieved documents: "
        f"{len(result.retrieved_documents)}"
    )

    print(
        f"Reranked documents: "
        f"{len(result.reranked_documents)}"
    )

    print(
        f"Compressed documents: "
        f"{len(result.compressed_documents)}"
    )

    print(
        f"Optimized documents: "
        f"{len(result.optimized_documents)}"
    )

    print(
        f"\nGrounding score: "
        f"{result.grounding_score:.2f}"
    )

    print(
        f"Grounded: "
        f"{result.is_grounded}"
    )

    print(
        f"Guardrails passed: "
        f"{result.guardrails_passed}"
    )

    print(
        "\nFinal Answer:"
    )

    print(
        result.answer
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert result.question == question

    assert result.domain in (
        "technical",
        "programming",
        "machine_learning",
        "data_science",
        "business",
        "education",
        "general",
        "other",
    )

    assert result.rewritten_query.strip()

    assert len(
        result.generated_queries
    ) >= 3

    assert len(
        result.retrieved_documents
    ) > 0

    assert len(
        result.reranked_documents
    ) > 0

    assert len(
        result.compressed_documents
    ) > 0

    assert len(
        result.optimized_documents
    ) > 0

    assert result.answer.strip()

    assert 0.0 <= (
        result.grounding_score
    ) <= 1.0

    assert result.is_grounded is True

    assert result.guardrails_passed is True

    assert "[Source:" in result.answer

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "REAL DOMAIN ROUTING: PASS ✓"
    )

    print(
        "REAL QUERY REWRITING: PASS ✓"
    )

    print(
        "REAL MULTI-QUERY: PASS ✓"
    )

    print(
        "REAL BGE-M3 EMBEDDINGS: PASS ✓"
    )

    print(
        "REAL PINECONE UPSERT: PASS ✓"
    )

    print(
        "REAL PINECONE VECTOR STORE: PASS ✓"
    )

    print(
        "REAL HYBRID RETRIEVAL: PASS ✓"
    )

    print(
        "REAL MMR: PASS ✓"
    )

    print(
        "LIGHTWEIGHT RERANKER: PASS ✓"
    )

    print(
        "REAL CONTEXTUAL COMPRESSION: PASS ✓"
    )

    print(
        "REAL CONTEXT OPTIMIZATION: PASS ✓"
    )

    print(
        "REAL ANSWER GENERATION: PASS ✓"
    )

    print(
        "REAL ANSWER GROUNDING: PASS ✓"
    )

    print(
        "REAL GUARDRAILS: PASS ✓"
    )

    print(
        "BGE RERANKER DOWNLOAD: SKIPPED ✓"
    )

    print("=" * 70)

    print(
        "\nREAL RAG PIPELINE INTEGRATION "
        "TEST SUCCESSFUL ✓"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()