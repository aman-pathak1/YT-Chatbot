from __future__ import annotations

from langchain_core.documents import Document

from generation.context_optimizer import (
    ContextOptimizerConfig,
    ContextWindowOptimizer,
)


def main() -> None:
    print("=" * 60)
    print("CONTEXT WINDOW OPTIMIZER TEST")
    print("=" * 60)

    config = ContextOptimizerConfig(
        max_context_characters=500,
        max_document_characters=250,
        min_document_characters=20,
    )

    optimizer = ContextWindowOptimizer(
        config=config
    )

    documents = [
        Document(
            page_content=(
                "Vector databases store embeddings and "
                "allow efficient similarity search."
            ),
            metadata={
                "source": "test",
                "chunk_index": 0,
                "chunk_start_time": 10.0,
            },
        ),
        Document(
            page_content=(
                "Vector databases store embeddings and "
                "allow efficient similarity search."
            ),
            metadata={
                "source": "test",
                "chunk_index": 1,
                "chunk_start_time": 20.0,
            },
        ),
        Document(
            page_content=(
                "Python is a programming language used "
                "for building many different applications. "
                "It is widely used in data science, web "
                "development, automation, and machine learning."
            ),
            metadata={
                "source": "test",
                "chunk_index": 2,
                "chunk_start_time": 30.0,
            },
        ),
        Document(
            page_content=(
                "This is a very small document."
            ),
            metadata={
                "source": "test",
                "chunk_index": 3,
                "chunk_start_time": 40.0,
            },
        ),
    ]

    print("\nBefore optimization:")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"{index}. "
            f"{len(document.page_content)} chars | "
            f"{document.page_content}"
        )

    result = optimizer.optimize(documents)

    print("\nAfter optimization:")

    for index, document in enumerate(
        result.optimized_documents,
        start=1,
    ):
        print(
            f"{index}. "
            f"{len(document.page_content)} chars | "
            f"{document.page_content}"
        )

    print("\nStatistics:")
    print(
        f"Original documents: "
        f"{result.original_document_count}"
    )

    print(
        f"Optimized documents: "
        f"{result.optimized_document_count}"
    )

    print(
        f"Original characters: "
        f"{result.original_character_count}"
    )

    print(
        f"Optimized characters: "
        f"{result.optimized_character_count}"
    )

    print(
        f"Characters saved: "
        f"{result.characters_saved}"
    )

    print(
        f"Compression ratio: "
        f"{result.compression_ratio:.2f}"
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    # Duplicate should be removed.
    assert result.optimized_document_count < len(documents)

    # Overall context budget must be respected.
    assert (
        result.optimized_character_count
        <= config.max_context_characters
    )

    # Every document must respect the per-document limit.
    for document in result.optimized_documents:
        assert (
            len(document.page_content)
            <= config.max_document_characters
        )

    # The duplicate content should appear only once.
    vector_database_documents = [
        document
        for document in result.optimized_documents
        if "Vector databases" in document.page_content
    ]

    assert len(vector_database_documents) == 1

    # Metadata must be preserved.
    for document in result.optimized_documents:
        assert "source" in document.metadata
        assert "chunk_index" in document.metadata

    print("\nDuplicate removal: PASS ✓")
    print("Context budget: PASS ✓")
    print("Per-document limit: PASS ✓")
    print("Metadata preservation: PASS ✓")
    print("No LLM/model download required: PASS ✓")

    print("\n" + "=" * 60)
    print("CONTEXT WINDOW OPTIMIZER TEST SUCCESSFUL ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()