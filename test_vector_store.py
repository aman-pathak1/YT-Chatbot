from __future__ import annotations

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.embeddings import BGEEmbeddings
from retrieval.vector_store import PineconeVectorStoreManager


def main() -> None:

    # ============================================================
    # 1. Fetch YouTube Transcript
    # ============================================================

    url = input("Enter YouTube URL: ").strip()

    print("\n[1/5] Fetching transcript...")

    documents = load_youtube_transcript(url)

    print(
        f"✓ Total transcript segments: "
        f"{len(documents)}"
    )

    test_documents = documents[:3]

    print(
        f"✓ Using first {len(test_documents)} "
        "segments for testing."
    )

    if not test_documents:
        print("✗ No transcript segments available.")
        return

    # ============================================================
    # 2. Translate
    # ============================================================

    print("\n[2/5] Translating transcript...")

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
        f"✓ Translated documents: "
        f"{len(translated_documents)}"
    )

    # ============================================================
    # 3. Initialize BGE-M3
    # ============================================================

    print("\n[3/5] Initializing BGE-M3...")

    embeddings = BGEEmbeddings(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=3,
        normalize_embeddings=True,
    )

    print(
        f"✓ Embedding dimension: "
        f"{embeddings.embedding_dimension}"
    )

    # ============================================================
    # 4. Initialize Pinecone
    # ============================================================

    print("\n[4/5] Initializing Pinecone...")

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

    print("✓ Pinecone initialized")

    # ============================================================
    # 5. Upsert + Retrieval Test
    # ============================================================

    print("\n[5/5] Testing upsert and retrieval...")

    document_ids = vector_store.add_documents(
        translated_documents
    )

    print(
        f"✓ Upserted vectors: "
        f"{len(document_ids)}"
    )

    print("\nVector IDs:")

    for document_id in document_ids:
        print(f"  {document_id}")

    # ------------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------------

    query = "What is this video about?"

    print(
        f"\nQuery: {query}"
    )

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    print(
        f"✓ Retrieved documents: "
        f"{len(results)}"
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

    # ============================================================
    # Final Verification
    # ============================================================

    assert len(document_ids) == len(
        translated_documents
    )

    assert len(results) <= 3

    print("\n" + "=" * 70)
    print("PINECONE VECTOR STORE TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()