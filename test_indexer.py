"""
Real integration test for the YouTube indexing pipeline.

Flow:

YouTube URL
    ↓
Transcript extraction
    ↓
Translation / normalization
    ↓
Timestamp-aware chunking
    ↓
BGE-M3 embeddings
    ↓
Pinecone indexing
"""

from __future__ import annotations

import logging

from ingestion.indexer import YouTubeIndexer
from ingestion.translator import TranscriptTranslator
from ingestion.splitter import TranscriptSplitter
from ingestion.embeddings import BGEEmbeddings
from retrieval.vector_store import PineconeVectorStoreManager


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VIDEO_URL = "https://www.youtube.com/watch?v=etnLX7m2MiA"

INDEX_NAME = "youtube-chatbot"
NAMESPACE = "youtube-transcripts"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("REAL YOUTUBE INDEXING INTEGRATION TEST")
    print("=" * 70)

    print(f"\nYouTube URL:\n{VIDEO_URL}")

    # -------------------------------------------------------------
    # 1. Initialize embeddings
    # -------------------------------------------------------------

    print("\nInitializing BGE-M3 embeddings...")

    embeddings = BGEEmbeddings()

    print("BGE-M3 embeddings initialized ✓")

    # -------------------------------------------------------------
    # 2. Initialize translator
    # -------------------------------------------------------------

    print("\nInitializing transcript translator...")

    translator = TranscriptTranslator()

    print("Transcript translator initialized ✓")

    # -------------------------------------------------------------
    # 3. Initialize splitter
    # -------------------------------------------------------------

    print("\nInitializing transcript splitter...")

    splitter = TranscriptSplitter()

    print("Transcript splitter initialized ✓")

    # -------------------------------------------------------------
    # 4. Initialize Pinecone
    # -------------------------------------------------------------

    print("\nInitializing Pinecone vector store...")

    vector_store = PineconeVectorStoreManager(
        embedding_model=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    print("Pinecone vector store initialized ✓")

    # -------------------------------------------------------------
    # 5. Build YouTube indexer
    # -------------------------------------------------------------

    print("\nBuilding YouTube indexer...")

    indexer = YouTubeIndexer(
        translator=translator,
        splitter=splitter,
        vector_store=vector_store,
    )

    print("YouTube indexer initialized ✓")

    # -------------------------------------------------------------
    # 6. Run complete indexing pipeline
    # -------------------------------------------------------------

    print("\nRunning indexing pipeline...")

    result = indexer.index_video(
        video_url=VIDEO_URL
    )

    # -------------------------------------------------------------
    # 7. Validate indexing result
    # -------------------------------------------------------------

    print("\n" + "-" * 70)
    print("INDEXING RESULT")
    print("-" * 70)

    print("\nVideo ID:")
    print(result.video_id)

    print("\nTranscript documents:")
    print(result.transcript_documents)

    print("\nChunk documents:")
    print(result.chunk_documents)

    print("\nIndexed documents:")
    print(result.indexed_documents)

    print("\nIndexed IDs:")

    for document_id in result.indexed_ids:
        print(f"  - {document_id}")

    # -------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------

    assert result.video_id, (
        "Video ID was not extracted."
    )

    assert result.transcript_documents > 0, (
        "No transcript documents were extracted."
    )

    assert result.chunk_documents > 0, (
        "No chunks were created."
    )

    assert result.indexed_documents > 0, (
        "No documents were indexed."
    )

    assert len(result.indexed_ids) == result.indexed_documents, (
        "Indexed ID count does not match indexed document count."
    )

    print("\nREAL TRANSCRIPT EXTRACTION: PASS ✓")
    print("REAL TRANSLATION / NORMALIZATION: PASS ✓")
    print("REAL TEXT SPLITTING: PASS ✓")
    print("REAL PINECONE INDEXING: PASS ✓")

    # -------------------------------------------------------------
    # 8. Verify Pinecone retrieval
    # -------------------------------------------------------------

    print("\n" + "-" * 70)
    print("VERIFYING PINECONE RETRIEVAL")
    print("-" * 70)

    query = "What does the video say about YouTube?"

    print("\nTest query:")
    print(query)

    retrieved_documents = vector_store.similarity_search(
        query=query,
        k=3,
    )

    print(
        f"\nRetrieved documents: "
        f"{len(retrieved_documents)}"
    )

    assert retrieved_documents, (
        "Pinecone returned no documents after indexing."
    )

    # -------------------------------------------------------------
    # Display retrieved documents
    # -------------------------------------------------------------

    for i, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        print(f"\nDocument {i}")
        print("-" * 40)

        print("Content:")
        print(
            document.page_content[:500]
        )

        print("\nMetadata:")

        print(
            document.metadata
        )

    print("\nREAL PINECONE RETRIEVAL: PASS ✓")

    # -------------------------------------------------------------
    # 9. Final result
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL YOUTUBE INDEXING TEST SUCCESSFUL ✓")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()