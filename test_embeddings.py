from __future__ import annotations

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.embeddings import BGEEmbeddings


def main() -> None:

    # ============================================================
    # 1. Fetch YouTube Transcript
    # ============================================================

    url = input("Enter YouTube URL: ").strip()

    print("\n[1/4] Fetching transcript...")

    documents = load_youtube_transcript(url)

    print(
        f"✓ Total transcript segments: "
        f"{len(documents)}"
    )

    # ============================================================
    # 2. Translate First 3 Segments
    # ============================================================

    test_documents = documents[:3]

    print(
        f"✓ Using first {len(test_documents)} "
        "segments for testing."
    )

    if not test_documents:
        print("✗ No transcript segments available.")
        return

    print("\n[2/4] Translating transcript...")

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
        f"✓ Translated segments: "
        f"{len(translated_documents)}"
    )

    # ============================================================
    # 3. Generate BGE-M3 Embeddings
    # ============================================================

    print("\n[3/4] Generating BGE-M3 embeddings...")

    embeddings = BGEEmbeddings(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=3,
        normalize_embeddings=True,
    )

    vectors = embeddings.embed_langchain_documents(
        translated_documents
    )

    print(
        f"✓ Generated vectors: "
        f"{len(vectors)}"
    )

    print(
        f"✓ Vector dimension: "
        f"{len(vectors[0])}"
    )

    print(
        f"✓ Model dimension: "
        f"{embeddings.embedding_dimension}"
    )

    print(
        f"✓ Model loaded: "
        f"{embeddings.is_loaded}"
    )

    # ============================================================
    # 4. Test Query Embedding
    # ============================================================

    print("\n[4/4] Testing query embedding...")

    query = "What is LangChain?"

    query_vector = embeddings.embed_query(
        query
    )

    print(
        f"✓ Query: {query}"
    )

    print(
        f"✓ Query vector dimension: "
        f"{len(query_vector)}"
    )

    print(
        f"✓ Query vector first 5 values: "
        f"{query_vector[:5]}"
    )

    # ============================================================
    # Final Verification
    # ============================================================

    assert len(vectors) == len(
        translated_documents
    )

    assert all(
        len(vector) == 1024
        for vector in vectors
    )

    assert len(query_vector) == 1024

    print("\n" + "=" * 70)
    print("BGE-M3 EMBEDDING TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()