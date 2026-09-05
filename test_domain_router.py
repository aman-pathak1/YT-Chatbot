from retrieval.domain_router import (
    DomainRouter,
    DomainRouterConfig,
)


def main() -> None:
    print("=" * 60)
    print("DOMAIN ROUTER TEST")
    print("=" * 60)

    config = DomainRouterConfig(
        model_name="qwen3:8b",
        temperature=0.0,
        max_tokens=256,
    )

    router = DomainRouter(config=config)

    test_queries = [
        "What is a vector database?",
        "How does backpropagation work in neural networks?",
        "How do I implement binary search in Python?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        result = router.route(query)

        print(f"Domain: {result.domain}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")

        assert result.original_query == query
        assert result.domain in config.allowed_domains
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning.strip()

    print("\n" + "=" * 60)
    print("DOMAIN ROUTER TEST SUCCESSFUL ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()