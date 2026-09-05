from __future__ import annotations

from retrieval.query_rewriter import (
    QueryRewriter,
    QueryRewriterError,
)


def main() -> None:

    print("=" * 70)
    print("QUERY REWRITER TEST")
    print("=" * 70)

    # ============================================================
    # 1. Initialize Query Rewriter
    # ============================================================

    print("\n[1/3] Initializing Qwen3 8B...")

    rewriter = QueryRewriter(
        model_name="qwen3:8b",
        temperature=0.0,
        num_predict=256,
    )

    print("✓ Query rewriter initialized")

    # ============================================================
    # 2. Test Queries
    # ============================================================

    test_queries = [
        "ye video kis bare me hai?",
        "what does he say about langchain?",
        "mujhe batao isme vector database kaise use hua hai",
        "what is the main purpose of using embeddings in this video?",
        "RAG system me vector database ka kya role hai?",
    ]

    print("\n[2/3] Testing query rewriting...")

    successful_tests = 0

    for index, query in enumerate(
        test_queries,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"TEST QUERY {index}")
        print("-" * 70)

        print("\nOriginal Query:")
        print(query)

        try:

            result = rewriter.rewrite(query)

            # ----------------------------------------------------
            # Validate result
            # ----------------------------------------------------

            assert result.original_query == query
            assert result.rewritten_query
            assert result.language
            assert result.intent

            # IMPORTANT:
            # Query rewriting stage must return ONE query.
            assert isinstance(
                result.rewritten_query,
                str,
            )

            # ----------------------------------------------------
            # Display
            # ----------------------------------------------------

            print("\nLanguage:")
            print(result.language)

            print("\nIntent:")
            print(result.intent)

            print("\nRewritten Query:")
            print(result.rewritten_query)

            print("\nKeywords:")

            if result.keywords:

                for keyword in result.keywords:
                    print(f"  - {keyword}")

            else:
                print("  None")

            successful_tests += 1

            print("\n✓ Query rewrite successful")

        except QueryRewriterError as exc:

            print(
                f"\n✗ Query rewrite failed: {exc}"
            )

            raise

    # ============================================================
    # 3. Simple API Test
    # ============================================================

    print("\n" + "=" * 70)
    print("SIMPLE API TEST")
    print("=" * 70)

    query = (
        "mujhe samjhao ki RAG system me "
        "Pinecone ka kya kaam hai"
    )

    print("\nOriginal Query:")
    print(query)

    rewritten_query = rewriter.rewrite_query(
        query
    )

    print("\nRewritten Query:")
    print(rewritten_query)

    assert isinstance(
        rewritten_query,
        str,
    )

    assert rewritten_query.strip()

    print("\n✓ Simple API successful")

    # ============================================================
    # Final Validation
    # ============================================================

    assert successful_tests == len(
        test_queries
    )

    print("\n" + "=" * 70)
    print(
        f"Tests Passed: "
        f"{successful_tests}/{len(test_queries)}"
    )

    print(
        "QUERY REWRITER TEST SUCCESSFUL ✓"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()