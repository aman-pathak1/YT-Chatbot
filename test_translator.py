from __future__ import annotations

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator


def main() -> None:

    url = input("Enter YouTube URL: ").strip()

    # ============================================================
    # 1. Fetch YouTube Transcript
    # ============================================================

    print("\n[1/3] Fetching transcript...")

    documents = load_youtube_transcript(url)

    print(
        f"✓ Total transcript segments: "
        f"{len(documents)}"
    )

    # ============================================================
    # 2. Take Only First 3 Segments
    # ============================================================

    test_documents = documents[:3]

    print(
        f"✓ Using first {len(test_documents)} "
        "segments for testing."
    )

    if not test_documents:
        print("✗ No transcript segments available.")
        return

    # ============================================================
    # 3. Translate Transcript using IndicTrans2
    # ============================================================

    print("\n[2/3] Translating transcript...")

    translator = TranscriptTranslator(
        model_name="ai4bharat/indictrans2-indic-en-dist-200M",
        batch_size=3,
        max_length=256,
        num_beams=1,
    )

    translated_documents = translator.translate_documents(
        test_documents
    )

    print(
        f"✓ Translated segments: "
        f"{len(translated_documents)}"
    )

    # ============================================================
    # 4. Verify Translation
    # ============================================================

    print("\n[3/3] Translation Verification")

    print("\n" + "=" * 70)
    print("TRANSLATION RESULTS")
    print("=" * 70)

    for index, document in enumerate(
        translated_documents
    ):

        print(
            f"\n--- Segment {index + 1} ---"
        )

        print("\nOriginal:")
        print(
            document.metadata.get(
                "original_text",
                "N/A",
            )
        )

        print("\nEnglish:")
        print(document.page_content)

        print("\nTimestamp:")

        start_time = document.metadata.get(
            "start_time",
            document.metadata.get(
                "chunk_start_time",
                "N/A",
            ),
        )

        end_time = document.metadata.get(
            "end_time",
            document.metadata.get(
                "chunk_end_time",
                "N/A",
            ),
        )

        print(
            f"{start_time} → {end_time}"
        )

    # ============================================================
    # 5. Final Result
    # ============================================================

    print("\n" + "=" * 70)
    print("TRANSLATOR TEST SUCCESSFUL ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()