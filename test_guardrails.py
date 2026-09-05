from __future__ import annotations

from langchain_core.documents import Document

from generation.guardrails import (
    AnswerGuardrails,
    GuardrailConfig,
)


def main() -> None:

    print("=" * 70)
    print("ANSWER GUARDRAILS TEST")
    print("=" * 70)

    documents = [
        Document(
            page_content=(
                "The creator welcomes viewers to the YouTube channel."
            ),
            metadata={
                "source": "https://youtu.be/EzYaFF7ahKw",
                "video_id": "EzYaFF7ahKw",
                "chunk_start_time": 2.32,
                "chunk_end_time": 6.48,
            },
        )
    ]

    guardrails = AnswerGuardrails(
        GuardrailConfig(
            min_answer_length=5,
            max_answer_length=1000,
        )
    )

    # --------------------------------------------------------
    # Test 1: Valid answer
    # --------------------------------------------------------

    print("\n[1/3] Testing valid answer...")

    valid_answer = (
        "The creator welcomes viewers to the YouTube channel "
        "[Source: 00:00:02]."
    )

    result = guardrails.validate(
        answer=valid_answer,
        documents=documents,
    )

    print(f"Valid       : {result.is_valid}")
    print(f"Reasons     : {result.reasons}")

    assert result.is_valid is True
    assert len(result.reasons) == 0

    print("✓ Valid answer passed")

    # --------------------------------------------------------
    # Test 2: Missing citation
    # --------------------------------------------------------

    print("\n[2/3] Testing missing citation...")

    invalid_answer = (
        "The creator welcomes viewers to the YouTube channel."
    )

    result = guardrails.validate(
        answer=invalid_answer,
        documents=documents,
    )

    print(f"Valid       : {result.is_valid}")
    print(f"Reasons     : {result.reasons}")

    assert result.is_valid is False
    assert any(
        "citation" in reason.lower()
        for reason in result.reasons
    )

    print("✓ Missing citation detected")

    # --------------------------------------------------------
    # Test 3: Empty answer
    # --------------------------------------------------------

    print("\n[3/3] Testing empty answer...")

    result = guardrails.validate(
        answer="",
        documents=documents,
    )

    print(f"Valid       : {result.is_valid}")
    print(f"Reasons     : {result.reasons}")

    assert result.is_valid is False
    assert any(
        "empty" in reason.lower()
        for reason in result.reasons
    )

    print("✓ Empty answer detected")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANSWER GUARDRAILS TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()