from __future__ import annotations

from langchain_core.documents import Document

from retrieval.contextual_compressor import (
    ContextualCompressor,
    CompressionConfig,
)


def main() -> None:

    print("=" * 70)
    print("CONTEXTUAL COMPRESSION TEST")
    print("=" * 70)

    query = "What does the video say about YouTube?"

    documents = [
        Document(
            page_content=(
                "The video starts with an introduction. "
                "The creator welcomes viewers to the YouTube channel. "
                "Later, the creator discusses Python programming. "
                "The video ends with a conclusion."
            ),
            metadata={"chunk_index": 0},
        ),
        Document(
            page_content=(
                "Pinecone is a vector database used for storing "
                "and retrieving embeddings. It supports semantic search."
            ),
            metadata={"chunk_index": 1},
        ),
    ]

    print(f"\nQuery:")
    print(query)

    print(f"\nInput documents: {len(documents)}")

    # --------------------------------------------------------
    # Initialize compressor
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING COMPRESSOR")
    print("=" * 70)

    config = CompressionConfig(
        model_name="qwen3:8b",
        temperature=0.0,
        max_output_tokens=256,
    )

    compressor = ContextualCompressor(config)

    print(f"\nModel  : {config.model_name}")
    print(f"Device : Ollama")

    # --------------------------------------------------------
    # Compress
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPRESSING CONTEXT")
    print("=" * 70)

    results = compressor.compress(
        query=query,
        documents=documents,
    )

    print(f"\n✓ Documents retained: {len(results)}")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPRESSED RESULTS")
    print("=" * 70)

    for index, document in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")

        print("\nCompressed Content:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert len(results) <= len(documents)

    for document in results:
        assert document.page_content.strip()
        assert document.metadata["contextual_compressed"] is True
        assert "original_content" in document.metadata

    print("\n" + "=" * 70)
    print("CONTEXTUAL COMPRESSION TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()