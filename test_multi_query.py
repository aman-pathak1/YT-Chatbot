from retrieval.multi_query import (
    MultiQueryGenerator,
    MultiQueryConfig,
)


def main() -> None:

    print("=" * 70)
    print("MULTI-QUERY GENERATION TEST")
    print("=" * 70)

    query = "What does the video say about YouTube?"

    # --------------------------------------------------------
    # 1. Initialize
    # --------------------------------------------------------

    print("\n[1/3] Initializing Multi-Query Generator...")

    config = MultiQueryConfig(
        model_name="qwen3:8b",
        num_queries=3,
        temperature=0.0,
        max_tokens=512,
    )

    generator = MultiQueryGenerator(config)

    print(f"✓ Model      : {config.model_name}")
    print(f"✓ Num Queries: {config.num_queries}")
    print(f"✓ Temperature: {config.temperature}")

    # --------------------------------------------------------
    # 2. Generate
    # --------------------------------------------------------

    print("\n[2/3] Generating alternative queries...")

    result = generator.generate(query)

    print("\nOriginal Query:")
    print(result.original_query)

    print("\nGenerated Queries:")

    for index, generated_query in enumerate(
        result.queries,
        start=1,
    ):
        print(f"{index}. {generated_query}")

    # --------------------------------------------------------
    # 3. Validation
    # --------------------------------------------------------

    print("\n[3/3] Validating results...")

    assert result.original_query == query

    assert len(result.queries) == 3

    assert len(set(result.queries)) == 3

    for generated_query in result.queries:
        assert isinstance(generated_query, str)
        assert generated_query.strip()

    print("✓ Exactly 3 unique queries generated")
    print("✓ All queries are non-empty strings")

    print("\n" + "=" * 70)
    print("MULTI-QUERY GENERATION TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()