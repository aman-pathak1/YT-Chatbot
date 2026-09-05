from __future__ import annotations

from ingestion.embeddings import BGEEmbeddings
from retrieval.vector_store import PineconeVectorStoreManager
from retrieval.mmr import (
    MMRConfig,
    MMRRetriever,
    MMRRetrievalError,
)


def main() -> None:

    print("=" * 70)
    print("MMR RETRIEVAL TEST")
    print("=" * 70)

    # ============================================================
    # 1. Initialize BGE-M3
    # ============================================================

    print("\n[1/3] Initializing BGE-M3...")

    embeddings = BGEEmbeddings(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=8,
        normalize_embeddings=True,
    )

    print(
        f"✓ Embedding dimension: "
        f"{embeddings.embedding_dimension}"
    )

    # ============================================================
    # 2. Connect to Existing Pinecone Namespace
    # ============================================================

    print("\n[2/3] Connecting to Pinecone...")

    vector_store = PineconeVectorStoreManager(
        embedding_model=embeddings,
        index_name="youtube-chatbot",
        namespace="test",
        dimension=1024,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        batch_size=8,
    )

    print("✓ Connected to existing Pinecone namespace")

    # ============================================================
    # 3. MMR Retrieval
    # ============================================================

    print("\n[3/3] Running MMR retrieval...")

    config = MMRConfig(
        k=3,
        fetch_k=3,
        lambda_mult=0.5,
    )

    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
        config=config,
    )

    query = "What is this video about?"

    print("\n" + "=" * 70)
    print("MMR CONFIGURATION")
    print("=" * 70)

    print(f"\nk              : {config.k}")
    print(f"fetch_k        : {config.fetch_k}")
    print(f"lambda_mult    : {config.lambda_mult}")

    print("\nQuery:")
    print(query)

    try:

        results = mmr_retriever.retrieve(
            query
        )

    except MMRRetrievalError as exc:

        print(
            f"\n✗ MMR retrieval failed:\n{exc}"
        )

        raise

    # ============================================================
    # Results
    # ============================================================

    print("\n" + "=" * 70)
    print("MMR RESULTS")
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
    # Validation
    # ============================================================

    assert len(results) <= config.k

    assert all(
        document.page_content.strip()
        for document in results
    )

    # Ensure no duplicate chunks
    chunk_indices = [
        document.metadata.get("segment_index")
        for document in results
        if document.metadata.get("segment_index")
        is not None
    ]

    assert len(chunk_indices) == len(
        set(chunk_indices)
    ), (
        "Duplicate transcript segments "
        "returned by MMR."
    )

    # ============================================================
    # Final
    # ============================================================

    print("\n" + "=" * 70)
    print("MMR RETRIEVAL TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()