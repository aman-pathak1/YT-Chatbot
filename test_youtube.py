from ingestion.youtube import load_youtube_transcript
from ingestion.splitter import TranscriptSplitter


def main() -> None:
    url = input("Enter YouTube URL: ").strip()

    # -----------------------------------------
    # 1. Test YouTube Transcript Ingestion
    # -----------------------------------------
    print("\n[1/2] Fetching transcript...")

    documents = load_youtube_transcript(url)

    print(f"✓ Transcript segments: {len(documents)}")

    # -----------------------------------------
    # 2. Test Text Splitting
    # -----------------------------------------
    print("\n[2/2] Splitting transcript...")

    splitter = TranscriptSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"✓ Total chunks: {len(chunks)}")

    # -----------------------------------------
    # 3. Verify First Chunk
    # -----------------------------------------
    print("\n" + "=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    first_chunk = chunks[0]

    print("\nContent:")
    print(first_chunk.page_content)

    print("\nMetadata:")
    for key, value in first_chunk.metadata.items():
        print(f"{key}: {value}")

    # -----------------------------------------
    # 4. Verify Multiple Chunks
    # -----------------------------------------
    print("\n" + "=" * 60)
    print("CHUNK VERIFICATION")
    print("=" * 60)

    for index, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {index}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(
            f"Start: "
            f"{chunk.metadata.get('chunk_start_time')}"
        )
        print(
            f"End: "
            f"{chunk.metadata.get('chunk_end_time')}"
        )
        print(
            f"Content: "
            f"{chunk.page_content[:100]}..."
        )

    # -----------------------------------------
    # 5. Final Result
    # -----------------------------------------
    print("\n" + "=" * 60)
    print("TEST SUCCESSFUL ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()