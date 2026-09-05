from generation.llm import QwenGenerator, LLMConfig


def main() -> None:
    print("=" * 70)
    print("QWEN3 LLM TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------

    config = LLMConfig(
        model_name="qwen3:8b",
        temperature=0.0,
        max_tokens=256,
    )

    print("\n[1/3] Initializing Qwen3...")

    generator = QwenGenerator(config)

    print(f"✓ Model      : {config.model_name}")
    print(f"✓ Temperature: {config.temperature}")
    print(f"✓ Max Tokens : {config.max_tokens}")

    # --------------------------------------------------------
    # 2. Generation
    # --------------------------------------------------------

    print("\n[2/3] Generating response...")

    prompt = """
Answer the following question in one short sentence.

Question:
What is YouTube?
""".strip()

    response = generator.generate(prompt)

    print("\nGenerated Answer:")
    print(response)

    # --------------------------------------------------------
    # 3. LCEL Runnable test
    # --------------------------------------------------------

    print("\n[3/3] Testing LangChain Runnable...")

    runnable = generator.runnable

    runnable_response = runnable.invoke(
        "Say 'LCEL test successful' in exactly one sentence."
    )

    print("\nRunnable Response:")
    print(runnable_response.content)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert response.strip()
    assert runnable_response.content.strip()
    assert generator.is_initialized is True

    print("\n" + "=" * 70)
    print("QWEN3 LLM TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()