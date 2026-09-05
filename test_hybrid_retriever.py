from __future__ import annotations

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.embeddings import BGEEmbeddings

from retrieval.vector_store import PineconeVectorStoreManager
from retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalConfig,
    HybridRetrievalError,
)


def main() -> None:

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    # ============================================================
    # 1. Load the same documents used in Pinecone test
    # ============================================================

    url = "https://youtu.be/EzYaFF7ahKw"

    print("\n[1/4] Loading transcript...")

    documents = load_youtube_transcript(url)

    test_documents = documents[:3]

    print(
        f"✓ Loaded {len(test_documents)} transcript segments"
    )

    # ============================================================
    # 2. Translate
    # ============================================================

    print("\n[2/4] Translating documents...")

    translator = TranscriptTranslator(
        model_name="ai4bharat/indictrans2-indic-en-dist-200M",
        batch_size=3,
        max_length=256,
        num_beams=1,
    )

    translated_documents = (
        translator.translate_documents(
            test_documents
        )
    )

    print(
        f"✓ Translated {len(translated_documents)} documents"
    )

    # ============================================================
    # 3. Connect to Pinecone
    # ============================================================

    print("\n[3/4] Connecting to Pinecone...")

    embeddings = BGEEmbeddings(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=3,
        normalize_embeddings=True,
    )

    vector_store = PineconeVectorStoreManager(
        embedding_model=embeddings,
        index_name="youtube-chatbot",
        namespace="test",
        dimension=1024,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        batch_size=3,
    )

    print("✓ Pinecone connected")

    # ============================================================
    # 4. Hybrid Retrieval
    # ============================================================

    print("\n[4/4] Initializing Hybrid Retriever...")

    config = HybridRetrievalConfig(
        dense_k=3,
        sparse_k=3,
        final_k=3,
        rrf_k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        documents=translated_documents,
        config=config,
    )

    query = "What does the video say about YouTube?"

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL CONFIGURATION")
    print("=" * 70)

    print(f"\nDense K       : {config.dense_k}")
    print(f"Sparse K      : {config.sparse_k}")
    print(f"Final K       : {config.final_k}")
    print(f"RRF K         : {config.rrf_k}")
    print(f"Dense Weight  : {config.dense_weight}")
    print(f"Sparse Weight : {config.sparse_weight}")

    print("\nQuery:")
    print(query)

    try:

        results = hybrid_retriever.retrieve(
            query
        )

    except HybridRetrievalError as exc:

        print(
            f"\n✗ Hybrid retrieval failed:\n{exc}"
        )

        raise

    # ============================================================
    # Results
    # ============================================================

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\n✓ Retrieved documents: "
        f"{len(results)}"
    )

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print("\nContent:")
        print(
            document.page_content
        )

        print("\nMetadata:")
        print(
            document.metadata
        )

    # ============================================================
    # Detailed Score Diagnostics
    # ============================================================

    print("\n" + "=" * 70)
    print("RRF SCORE DIAGNOSTICS")
    print("=" * 70)

    diagnostic_results = (
        hybrid_retriever.retrieve_with_scores(
            query
        )
    )

    for index, item in enumerate(
        diagnostic_results,
        start=1,
    ):

        print(
            f"\n--- Candidate {index} ---"
        )

        print(
            "Dense Rank   :",
            item["dense_rank"],
        )

        print(
            "Sparse Rank  :",
            item["sparse_rank"],
        )

        print(
            "Sparse Score :",
            item["sparse_score"],
        )

        print(
            "Fused Score  :",
            item["fused_score"],
        )

        print(
            "Content      :",
            item["document"].page_content,
        )

    # ============================================================
    # Validation
    # ============================================================

    assert results

    assert len(results) <= config.final_k

    assert all(
        document.page_content.strip()
        for document in results
    )

    assert diagnostic_results

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()