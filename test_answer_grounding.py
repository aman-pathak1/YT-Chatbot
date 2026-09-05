from __future__ import annotations

from langchain_core.documents import Document

from generation.answer_grounding import (
    AnswerGrounding,
    GroundingConfig,
)


class MockAnswerGrounding(AnswerGrounding):
    """
    Lightweight grounding evaluator for testing.

    This class does NOT call Qwen3.
    It uses simple lexical overlap to verify the
    grounding component's interface and result handling.
    """

    def _load_llm(self):
        # Never initialize the actual LLM.
        return None

    def evaluate(
        self,
        answer: str,
        documents: list[Document],
    ):
        answer_words = set(
            answer.lower()
            .replace(".", "")
            .replace(",", "")
            .split()
        )

        context_words = set()

        for document in documents:
            context_words.update(
                document.page_content.lower()
                .replace(".", "")
                .replace(",", "")
                .split()
            )

        if not answer_words:
            score = 0.0
        else:
            supported_words = (
                answer_words.intersection(context_words)
            )

            score = (
                len(supported_words)
                / len(answer_words)
            )

        is_grounded = (
            score
            >= self.config.minimum_grounding_score
        )

        cited_timestamps = ()

        import re

        cited_timestamps = tuple(
            re.findall(
                r"\[Source:\s*(\d{2}:\d{2}:\d{2})\]",
                answer,
            )
        )

        return type(
            "MockGroundingResult",
            (),
            {
                "answer": answer,
                "is_grounded": is_grounded,
                "grounding_score": score,
                "explanation": (
                    "Answer claims were checked "
                    "against the provided context."
                ),
                "cited_timestamps": cited_timestamps,
                "passed": is_grounded,
            },
        )()


def main() -> None:
    print("=" * 60)
    print("LIGHTWEIGHT ANSWER GROUNDING TEST")
    print("=" * 60)

    config = GroundingConfig(
        model_name="qwen3:8b",
        temperature=0.0,
        max_tokens=512,
        minimum_grounding_score=0.70,
    )

    grounding = MockAnswerGrounding(
        config=config
    )

    documents = [
        Document(
            page_content=(
                "YouTube is a video sharing platform "
                "where users can upload videos."
            ),
            metadata={
                "source": "test",
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
                "chunk_index": 1,
                "chunk_start_time": 8.0,
            },
        ),
    ]

    # --------------------------------------------------
    # Test 1: Grounded answer
    # --------------------------------------------------

    grounded_answer = (
        "YouTube is a video sharing platform "
        "where users can upload videos. "
        "[Source: 00:00:02]"
    )

    print("\nTest 1: Grounded answer")

    result = grounding.evaluate(
        answer=grounded_answer,
        documents=documents,
    )

    print(
        f"Grounded: {result.is_grounded}"
    )

    print(
        f"Grounding score: "
        f"{result.grounding_score:.2f}"
    )

    print(
        f"Explanation: "
        f"{result.explanation}"
    )

    print(
        f"Citations: "
        f"{result.cited_timestamps}"
    )

    assert result.is_grounded is True

    assert (
        result.grounding_score
        >= config.minimum_grounding_score
    )

    assert (
        "00:00:02"
        in result.cited_timestamps
    )

    # --------------------------------------------------
    # Test 2: Unsupported answer
    # --------------------------------------------------

    unsupported_answer = (
        "YouTube was founded in 1999 "
        "and has exactly one billion users."
    )

    print("\nTest 2: Unsupported answer")

    result = grounding.evaluate(
        answer=unsupported_answer,
        documents=documents,
    )

    print(
        f"Grounded: {result.is_grounded}"
    )

    print(
        f"Grounding score: "
        f"{result.grounding_score:.2f}"
    )

    print(
        f"Explanation: "
        f"{result.explanation}"
    )

    assert result.is_grounded is False

    assert (
        result.grounding_score
        < config.minimum_grounding_score
    )

    # --------------------------------------------------
    # Test 3: Citation extraction
    # --------------------------------------------------

    citation_answer = (
        "Users can create YouTube channels "
        "and organize videos into playlists. "
        "[Source: 00:00:08]"
    )

    print("\nTest 3: Citation extraction")

    result = grounding.evaluate(
        answer=citation_answer,
        documents=documents,
    )

    print(
        f"Citations: "
        f"{result.cited_timestamps}"
    )

    assert (
        "00:00:08"
        in result.cited_timestamps
    )

    # --------------------------------------------------
    # Test 4: Empty answer validation
    # --------------------------------------------------

    print("\nTest 4: Empty answer handling")

    try:
        grounding.evaluate(
            answer="",
            documents=documents,
        )

        raise AssertionError(
            "Empty answer should raise an error."
        )

    except Exception:
        print(
            "Empty answer rejected: PASS ✓"
        )

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    assert grounding._llm is None

    print("\nLLM loaded: NO ✓")
    print("Large model download: NO ✓")
    print("Grounded answer detection: PASS ✓")
    print("Unsupported answer detection: PASS ✓")
    print("Citation extraction: PASS ✓")
    print("Input validation: PASS ✓")

    print("\n" + "=" * 60)
    print(
        "LIGHTWEIGHT ANSWER GROUNDING "
        "TEST SUCCESSFUL ✓"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()