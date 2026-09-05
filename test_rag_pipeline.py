from __future__ import annotations

from langchain_core.documents import Document

from pipeline.rag_pipeline import RAGPipeline


# ============================================================
# Mock Result Classes
# ============================================================

class MockDomainResult:
    def __init__(self, domain: str) -> None:
        self.original_query = ""
        self.domain = domain
        self.confidence = 0.95
        self.reasoning = "Test domain classification."


class MockRewriteResult:
    def __init__(self, query: str) -> None:
        self.original_query = query
        self.rewritten_query = query
        self.language = "english"
        self.intent = "factual"
        self.keywords = ("youtube",)


class MockMultiQueryResult:
    def __init__(self, query: str) -> None:
        self.original_query = query
        self.queries = (
            query,
            "information about YouTube",
            "what is mentioned about YouTube",
        )


class MockGroundingResult:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.is_grounded = True
        self.grounding_score = 0.95
        self.explanation = (
            "The answer is fully supported by the context."
        )
        self.cited_timestamps = ("00:00:02",)

        @property
        def passed(self) -> bool:
            return self.is_grounded


class MockGuardrailResult:
    def __init__(self) -> None:
        self.is_valid = True
        self.reason = ""


# ============================================================
# Mock Components
# ============================================================

class MockDomainRouter:

    def route(self, query: str):
        result = MockDomainResult("technical")
        result.original_query = query
        return result


class MockQueryRewriter:

    def rewrite(self, query: str, conversation_history: str = ""):
        return MockRewriteResult(query)


class MockMultiQueryGenerator:

    def generate(self, query: str):
        return MockMultiQueryResult(query)


class MockHybridRetriever:

    def __init__(self) -> None:
        self.sample_docs = [
            Document(
                page_content="YouTube is a video sharing platform where users can upload videos.",
                metadata={"video_id": "test_video", "chunk_index": 0},
            ),
            Document(
                page_content="Users can create YouTube channels and organize videos into playlists.",
                metadata={"video_id": "test_video", "chunk_index": 1},
            ),
        ]

    def retrieve(
        self,
        query: str,
        documents: list[Document] | None = None,
    ) -> list[Document]:

        if documents:
            return documents[:3]
        return self.sample_docs


class MockMMRRetriever:

    def retrieve(
        self,
        query: str,
        documents: list[Document] | None = None,
    ) -> list[Document]:

        return documents[:3] if documents else []


class MockReranker:

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        # Put YouTube-related documents first.
        youtube_documents = [
            document
            for document in documents
            if "youtube" in document.page_content.lower()
        ]

        other_documents = [
            document
            for document in documents
            if "youtube" not in document.page_content.lower()
        ]

        return youtube_documents + other_documents


class MockContextualCompressor:

    def compress(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        return documents


class MockContextOptimizerResult:

    def __init__(
        self,
        documents: list[Document],
    ) -> None:

        self.optimized_documents = documents
        self.documents = tuple(documents)

        self.original_document_count = len(documents)
        self.optimized_document_count = len(documents)

        self.original_character_count = sum(
            len(document.page_content)
            for document in documents
        )

        self.optimized_character_count = (
            self.original_character_count
        )

    @property
    def characters_saved(self) -> int:
        return 0

    @property
    def compression_ratio(self) -> float:
        return 1.0


class MockContextOptimizer:

    def optimize(
        self,
        documents: list[Document],
    ) -> MockContextOptimizerResult:

        return MockContextOptimizerResult(
            documents
        )


class MockAnswerGenerator:

    def generate(
        self,
        question: str,
        documents: list[Document],
    ) -> str:

        return (
            "YouTube is a video sharing platform "
            "where users can upload videos. "
            "[Source: 00:00:02]"
        )


class MockAnswerGrounding:

    def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ) -> MockGroundingResult:

        return MockGroundingResult(answer)


class MockGuardrails:

    def validate(
        self,
        answer: str,
        documents: list[Document],
    ) -> MockGuardrailResult:

        return MockGuardrailResult()

    def validate_or_fallback(
        self,
        answer: str,
        documents: list[Document],
    ) -> str:

        return answer


# ============================================================
# Main Test
# ============================================================

def main() -> None:

    print("=" * 60)
    print("LIGHTWEIGHT RAG PIPELINE TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create mock components
    # --------------------------------------------------------

    domain_router = MockDomainRouter()
    query_rewriter = MockQueryRewriter()
    multi_query_generator = MockMultiQueryGenerator()

    hybrid_retriever = MockHybridRetriever()
    mmr_retriever = MockMMRRetriever()
    reranker = MockReranker()

    contextual_compressor = (
        MockContextualCompressor()
    )

    context_optimizer = MockContextOptimizer()

    answer_generator = MockAnswerGenerator()
    answer_grounding = MockAnswerGrounding()
    guardrails = MockGuardrails()

    # --------------------------------------------------------
    # Create pipeline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Test documents
    # --------------------------------------------------------

    documents = [
        Document(
            page_content=(
                "YouTube is a video sharing platform "
                "where users can upload videos."
            ),
            metadata={
                "source": "test",
                "video_id": "test_video",
                "chunk_index": 0,
                "chunk_start_time": 2.0,
            },
        ),
        Document(
            page_content=(
                "Users can create YouTube channels "
                "and organize videos into playlists."
            ),
            metadata={
                "source": "test",
                "video_id": "test_video",
                "chunk_index": 1,
                "chunk_start_time": 8.0,
            },
        ),
        Document(
            page_content=(
                "Python is a programming language "
                "used for application development."
            ),
            metadata={
                "source": "test",
                "video_id": "test_video",
                "chunk_index": 2,
                "chunk_start_time": 15.0,
            },
        ),
    ]

    question = (
        "What does the video say about YouTube?"
    )

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    result = pipeline.run(
        question=question,
        documents=documents,
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\nPipeline Result:")
    print(
        f"Domain: {result.domain}"
    )

    print(
        f"Rewritten Query: "
        f"{result.rewritten_query}"
    )

    print(
        "\nGenerated Queries:"
    )

    for index, query in enumerate(
        result.generated_queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )

    print(
        "\nRetrieved Documents: "
        f"{len(result.retrieved_documents)}"
    )

    print(
        "Reranked Documents: "
        f"{len(result.reranked_documents)}"
    )

    print(
        "Compressed Documents: "
        f"{len(result.compressed_documents)}"
    )

    print(
        "Optimized Documents: "
        f"{len(result.optimized_documents)}"
    )

    print(
        f"\nGrounding Score: "
        f"{result.grounding_score:.2f}"
    )

    print(
        f"Grounded: "
        f"{result.is_grounded}"
    )

    print(
        f"Guardrails Passed: "
        f"{result.guardrails_passed}"
    )

    print(
        f"\nFinal Answer:\n"
        f"{result.answer}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    # Basic result validation.
    assert result.question == question

    assert result.answer.strip()

    # Domain routing.
    assert result.domain == "technical"

    # Query rewriting.
    assert result.rewritten_query == question

    # Multi-query generation.
    assert len(result.generated_queries) == 3

    # Retrieval.
    assert len(result.retrieved_documents) > 0

    # Reranking.
    assert len(result.reranked_documents) > 0

    assert (
        "YouTube"
        in result.reranked_documents[0].page_content
    )

    # Contextual compression.
    assert len(result.compressed_documents) > 0

    # Context optimization.
    assert len(result.optimized_documents) > 0

    # Generation.
    assert "YouTube" in result.answer

    # Citation.
    assert "[Source: 00:00:02]" in result.answer

    # Grounding.
    assert result.is_grounded is True

    assert result.grounding_score >= 0.70

    # Guardrails.
    assert result.guardrails_passed is True

    print("\n" + "=" * 60)
    print("DOMAIN ROUTING: PASS ✓")
    print("QUERY REWRITING: PASS ✓")
    print("MULTI-QUERY: PASS ✓")
    print("HYBRID RETRIEVAL: PASS ✓")
    print("MMR: PASS ✓")
    print("RERANKING: PASS ✓")
    print("CONTEXTUAL COMPRESSION: PASS ✓")
    print("CONTEXT OPTIMIZATION: PASS ✓")
    print("ANSWER GENERATION: PASS ✓")
    print("ANSWER GROUNDING: PASS ✓")
    print("GUARDRAILS: PASS ✓")
    print("NO HEAVY MODEL DOWNLOAD: PASS ✓")
    print("=" * 60)

    print(
        "\nRAG PIPELINE TEST SUCCESSFUL ✓"
    )


if __name__ == "__main__":
    main()