from __future__ import annotations

import logging

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.splitter import TranscriptSplitter
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

from evaluation.ragas_evaluator import RAGASEvaluator


logging.basicConfig(
    level=logging.INFO,
)


# ============================================================
# Lightweight Reranker
# ============================================================

class LightweightReranker(BGEReranker):
    """
    Lightweight reranker used for integration testing.

    The actual BGE reranker model is NOT loaded.
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
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FINAL REAL YOUTUBE RAGAS EVALUATION")
    print("=" * 70)

    video_url = (
        "https://www.youtube.com/watch?v=etnLX7m2MiA"
    )

    question = (
        "What does the video say about YouTube?"
    )

    ground_truth = (
        "The speaker welcomes viewers to his YouTube "
        "channel and says that the video continues "
        "the playlist."
    )

    print("\nQuestion:")
    print(question)

    # ========================================================
    # 1. Load actual YouTube transcript
    # ========================================================

    print("\nLoading actual YouTube transcript...")

    transcript_documents = load_youtube_transcript(
        video_url
    )

    print(
        f"Transcript segments loaded: "
        f"{len(transcript_documents)} ✓"
    )

    # ========================================================
    # 2. Translation / normalization
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
    # 3. Actual timestamp-aware chunking
    # ========================================================

    print("\nCreating actual retrieval chunks...")

    splitter = TranscriptSplitter()

    documents = splitter.split_documents(
        translated_documents
    )

    print(
        f"Retrieval chunks: "
        f"{len(documents)} ✓"
    )

    # ========================================================
    # 4. BGE-M3
    # ========================================================

    print("\nInitializing BGE-M3...")

    embedding_model = BGEEmbeddings()

    print("BGE-M3 initialized ✓")

    # ========================================================
    # 5. Existing Pinecone index
    # ========================================================

    print("\nConnecting to existing Pinecone index...")

    vector_store = PineconeVectorStoreManager(
        embedding_model=embedding_model,
        index_name="youtube-chatbot",
        namespace="youtube-transcripts",
    )

    print("Pinecone connected ✓")

    # ========================================================
    # 6. Hybrid Retriever
    # ========================================================

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        documents=documents,
    )

    # ========================================================
    # 7. MMR
    # ========================================================

    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
    )

    # ========================================================
    # 8. Lightweight Reranker
    # ========================================================

    reranker = LightweightReranker()

    # ========================================================
    # 9. Contextual Compression
    # ========================================================

    contextual_compressor = (
        ContextualCompressor()
    )

    # ========================================================
    # 10. Context Optimizer
    # ========================================================

    context_optimizer = (
        ContextWindowOptimizer()
    )

    # ========================================================
    # 11. Answer Generator
    # ========================================================

    answer_generator = AnswerGenerator()

    # ========================================================
    # 12. Grounding
    # ========================================================

    answer_grounding = AnswerGrounding()

    # ========================================================
    # 13. Guardrails
    # ========================================================

    guardrails = AnswerGuardrails()

    # ========================================================
    # 14. Query components
    # ========================================================

    domain_router = DomainRouter()

    query_rewriter = QueryRewriter()

    multi_query_generator = (
        MultiQueryGenerator()
    )

    # ========================================================
    # 15. Build RAG Pipeline
    # ========================================================

    print("\nBuilding RAG pipeline...")

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

    print("RAG pipeline initialized ✓")

    # ========================================================
    # 16. Run RAG
    # ========================================================

    print("\nRunning RAG pipeline...")

    result = pipeline.run(
        question=question,
        documents=documents,
    )

    print("RAG pipeline completed ✓")

    # ========================================================
    # 17. Generated Answer
    # ========================================================

    print("\n" + "-" * 70)
    print("GENERATED ANSWER")
    print("-" * 70)

    print(result.answer)

    # ========================================================
    # 18. Contexts for RAGAS
    # ========================================================

    ragas_contexts = [
        document.page_content
        for document in result.optimized_documents
    ]

    print("\n" + "-" * 70)
    print("CONTEXTS USED FOR RAGAS")
    print("-" * 70)

    for index, context in enumerate(
        ragas_contexts,
        start=1,
    ):
        print(f"\nContext {index}:")
        print(context)

    # ========================================================
    # 19. RAGAS
    # ========================================================

    print("\n" + "-" * 70)
    print("RUNNING FINAL RAGAS EVALUATION")
    print("-" * 70)

    evaluator = RAGASEvaluator()

    ragas_result = evaluator.evaluate_sample(
        question=question,
        answer=result.answer,
        contexts=ragas_contexts,
        ground_truth=ground_truth,
    )

    # ========================================================
    # 20. Final Scores
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RAGAS SCORES")
    print("=" * 70)

    print(
        f"faithfulness       : "
        f"{ragas_result.faithfulness}"
    )

    print(
        f"answer_relevancy   : "
        f"{ragas_result.answer_relevancy}"
    )

    print(
        f"context_precision  : "
        f"{ragas_result.context_precision}"
    )

    print(
        f"context_recall     : "
        f"{ragas_result.context_recall}"
    )

    print("\n" + "=" * 70)
    print("FINAL REAL RAGAS EVALUATION COMPLETED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()