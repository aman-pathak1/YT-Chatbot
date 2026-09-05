from langchain_core.documents import Document

from generation.prompt_template import RAGPromptTemplate


def main() -> None:

    print("=" * 70)
    print("RAG PROMPT TEMPLATE TEST")
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
    # Initialize
    # --------------------------------------------------------

    prompt_template = RAGPromptTemplate()

    print("\n✓ Prompt template initialized")

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = prompt_template.build_context(documents)

    print("\n" + "=" * 70)
    print("FORMATTED CONTEXT")
    print("=" * 70)

    print(context)

    # --------------------------------------------------------
    # Build complete prompt
    # --------------------------------------------------------

    messages = prompt_template.build(
        question=question,
        documents=documents,
    )

    print("\n" + "=" * 70)
    print("GENERATED MESSAGES")
    print("=" * 70)

    for index, message in enumerate(messages, start=1):

        print(f"\n--- Message {index} ---")
        print(f"Role: {message.type}")
        print(message.content)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert len(messages) == 2

    full_prompt = "\n".join(
        message.content
        for message in messages
    )

    assert question in full_prompt
    assert "YouTube channel" in full_prompt
    assert "00:00:02" in full_prompt
    assert "00:00:08" in full_prompt
    assert "ONLY" in full_prompt

    print("\n" + "=" * 70)
    print("RAG PROMPT TEMPLATE TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()