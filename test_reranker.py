from __future__ import annotations

from langchain_core.documents import Document

from retrieval.reranker import (
    RerankerConfig,
    BGEReranker,
)


class MockReranker(BGEReranker):
    """
    Lightweight test reranker.

    The actual BGE reranker model is NOT loaded.
    Simple lexical overlap is used only to verify
    the reranking interface and pipeline.
    """

    def _load_model(self):
        # Never load the actual BGE model during this test.
        return None

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
    ):
        query_words = set(
            query.lower().split()
        )

        scored_documents = []

        for document in documents:
            document_words = set(
                document.page_content.lower().split()
            )

            overlap = len(
                query_words.intersection(document_words)
            )

            score = float(overlap)

            scored_documents.append(
                (score, document)
            )

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            (document, score)
            for score, document in scored_documents[
                : self.config.top_n
            ]
        ]


def main() -> None:
    print("=" * 60)
    print("LIGHTWEIGHT RERANKER TEST")
    print("=" * 60)

    # Configuration only.
    # The actual BGE model will NOT be downloaded.
    config = RerankerConfig(
        model_name="BAAI/bge-reranker-v2-m3",
        top_n=3,
        batch_size=8,
        max_length=512,
    )

    reranker = MockReranker(config=config)

    documents = [
        Document(
            page_content=(
                "Python is a programming language "
                "used for many applications."
            ),
            metadata={
                "source": "test",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "YouTube is a video sharing platform "
                "where users can upload videos."
            ),
            metadata={
                "source": "test",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content=(
                "Vector databases store embeddings "
                "and support similarity search."
            ),
            metadata={
                "source": "test",
                "chunk_index": 2,
            },
        ),
        Document(
            page_content=(
                "A YouTube video can contain a transcript "
                "that can be searched."
            ),
            metadata={
                "source": "test",
                "chunk_index": 3,
            },
        ),
    ]

    query = "What does the video say about YouTube?"

    print("\nQuery:")
    print(query)

    print("\nDocuments before reranking:")

    for index, document in enumerate(documents):
        print(
            f"{index + 1}. "
            f"{document.page_content}"
        )

    # ---------------------------------------------
    # Run lightweight reranking
    # ---------------------------------------------

    results = reranker.rerank_with_scores(
        query=query,
        documents=documents,
    )

    print("\nDocuments after reranking:")

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"Score={score:.2f} | "
            f"{document.page_content}"
        )

    # ---------------------------------------------
    # Assertions
    # ---------------------------------------------

    assert len(results) == 3

    # Highest ranked document should be YouTube-related.
    first_document = results[0][0]

    assert "YouTube" in first_document.page_content

    # Scores should be in descending order.
    for index in range(len(results) - 1):
        assert (
            results[index][1]
            >= results[index + 1][1]
        )

    # Configuration checks.
    assert reranker.config.top_n == 3
    assert reranker.config.batch_size == 8
    assert reranker.config.max_length == 512

    # The actual model must not be loaded.
    assert reranker._model is None

    print("\nModel downloaded: NO ✓")
    print("Actual BGE reranker loaded: NO ✓")
    print("Mock reranker executed: YES ✓")

    print("\n" + "=" * 60)
    print("LIGHTWEIGHT RERANKER TEST SUCCESSFUL ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()