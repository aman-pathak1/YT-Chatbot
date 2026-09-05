from __future__ import annotations

from langchain_core.documents import Document

from generation.answer_generator import (
    AnswerGenerator,
    AnswerGeneratorConfig,
)


def main() -> None:

    print("=" * 70)
    print("ANSWER GENERATOR TEST")
    print("=" * 70)

    question = "What does the video say about YouTube?"

    documents = [
        Document(
            page_content=(
                "The creator welcomes viewers to the YouTube channel "
                "and introduces the topic of the video."
            ),
            metadata={
                "source": "https://youtu.be/EzYaFF7ahKw",
                "video_id": "EzYaFF7ahKw",
                "chunk_start_time": 2.32,
                "chunk_end_time": 6.48,
            },
        ),
        Document(
            page_content=(
                "The creator continues discussing the YouTube playlist."
            ),
            metadata={
                "source": "https://youtu.be/EzYaFF7ahKw",
                "video_id": "EzYaFF7ahKw",
                "chunk_start_time": 8.56,
                "chunk_end_time": 12.20,
            },
        ),
    ]

    # --------------------------------------------------------
    # 1. Initialize
    # --------------------------------------------------------

    print("\n[1/3] Initializing Answer Generator...")

    config = AnswerGeneratorConfig()

    generator = AnswerGenerator(config)

    print("✓ Answer generator initialized")
    print(f"✓ Model: {config.llm.model_name}")

    # --------------------------------------------------------
    # 2. Generate answer
    # --------------------------------------------------------

    print("\n[2/3] Generating grounded answer...")

    answer = generator.generate(
        question=question,
        documents=documents,
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerated Answer:")
    print(answer)

    # --------------------------------------------------------
    # 3. Test LCEL runnable
    # --------------------------------------------------------

    print("\n[3/3] Testing complete LCEL chain...")

    context = generator.prompt_template.build_context(
        documents
    )

    lcel_answer = generator.runnable.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    print("\nLCEL Answer:")
    print(lcel_answer)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert isinstance(answer, str)
    assert answer.strip()

    assert isinstance(lcel_answer, str)
    assert lcel_answer.strip()

    assert generator.is_ready is True

    print("\n" + "=" * 70)
    print("ANSWER GENERATOR TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()