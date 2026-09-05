from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    """
    Internal representation of a timestamped transcript segment.
    """

    text: str
    start_time: float
    end_time: float


class TranscriptSplitter:
    """
    Production-oriented timestamp-aware transcript splitter.

    Chunking strategy:

        Timestamped Transcript Segments
                    ↓
        Meaningful/Semantic Boundaries
                    ↓
        Target Chunk Construction
                    ↓
        RecursiveCharacterTextSplitter
              (only when required)
                    ↓
        Accurate Timestamp Mapping
                    ↓
        Retrieval-ready Documents

    Features:
    - Timestamp-aware chunking
    - Meaningful sentence/topic boundaries
    - RecursiveCharacterTextSplitter fallback
    - Configurable chunk size and overlap
    - Accurate chunk-level timestamps
    - Preserves source/video metadata
    - Robust validation
    - Handles empty/invalid transcript segments
    - Deterministic chunk generation
    """

    # Common sentence-ending patterns.
    _SENTENCE_BOUNDARY_PATTERN = re.compile(
        r"(?<=[.!?।])\s+"
    )

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 300,
    ) -> None:
        """
        Initialize the transcript splitter.

        Args:
            chunk_size:
                Maximum target size of a chunk in characters.

            chunk_overlap:
                Number of characters to overlap between adjacent
                chunks when recursive splitting is required.

            min_chunk_size:
                Minimum preferred chunk size before creating a
                new semantic chunk.

        Raises:
            ValueError:
                If configuration values are invalid.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if min_chunk_size <= 0:
            raise ValueError(
                "min_chunk_size must be greater than 0."
            )

        if min_chunk_size > chunk_size:
            raise ValueError(
                "min_chunk_size cannot be greater than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Recursive splitter acts as the safety/fallback layer.
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "। ",
                ", ",
                " ",
                "",
            ],
            keep_separator=True,
            strip_whitespace=True,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_segment(
        document: Document,
    ) -> TranscriptSegment | None:
        """
        Convert a LangChain Document into an internal transcript segment.

        Invalid or empty documents are ignored safely.
        """

        text = document.page_content.strip()

        if not text:
            return None

        metadata = document.metadata

        try:
            start_time = float(
                metadata.get("start_time", 0.0)
            )

            end_time = float(
                metadata.get(
                    "end_time",
                    start_time,
                )
            )

        except (TypeError, ValueError):
            logger.warning(
                "Invalid timestamp metadata encountered. "
                "Using safe defaults."
            )

            start_time = 0.0
            end_time = start_time

        if start_time < 0:
            start_time = 0.0

        if end_time < start_time:
            end_time = start_time

        return TranscriptSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
        )

    def _prepare_segments(
        self,
        documents: Sequence[Document],
    ) -> list[tuple[TranscriptSegment, Document]]:
        """
        Validate, clean and chronologically sort transcript segments.
        """

        prepared: list[
            tuple[TranscriptSegment, Document]
        ] = []

        for document in documents:

            segment = self._extract_segment(document)

            if segment is None:
                continue

            prepared.append(
                (
                    segment,
                    document,
                )
            )

        if not prepared:
            return []

        # Transcript should normally already be chronological,
        # but sorting makes the splitter deterministic and robust.
        prepared.sort(
            key=lambda item: item[0].start_time
        )

        return prepared

    # ------------------------------------------------------------------
    # Semantic boundary detection
    # ------------------------------------------------------------------

    @classmethod
    def _split_into_sentences(
        cls,
        text: str,
    ) -> list[str]:
        """
        Split transcript text into sentence-like units.

        Supports common English/Hinglish/Hindi punctuation.
        """

        text = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if not text:
            return []

        sentences = cls._SENTENCE_BOUNDARY_PATTERN.split(
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # ------------------------------------------------------------------
    # Semantic grouping
    # ------------------------------------------------------------------

    def _build_semantic_groups(
        self,
        prepared_segments: list[
            tuple[TranscriptSegment, Document]
        ],
    ) -> list[dict]:
        """
        Build meaningful transcript groups while preserving
        their original timestamp boundaries.

        A group grows until the target chunk size is reached.
        We preferentially break at sentence boundaries rather
        than arbitrarily cutting transcript text.
        """

        groups: list[dict] = []

        current_text_parts: list[str] = []
        current_segments: list[TranscriptSegment] = []
        current_length = 0

        for segment, _original_document in prepared_segments:

            sentences = self._split_into_sentences(
                segment.text
            )

            # If a transcript segment does not contain recognizable
            # sentence boundaries, treat the whole segment as one unit.
            if not sentences:
                continue

            for sentence in sentences:

                sentence_length = len(sentence)

                # --------------------------------------------------
                # Case 1:
                # Current group is empty.
                # --------------------------------------------------
                if not current_text_parts:

                    current_text_parts.append(sentence)
                    current_segments.append(segment)
                    current_length = sentence_length

                    continue

                projected_length = (
                    current_length
                    + 1
                    + sentence_length
                )

                # --------------------------------------------------
                # Case 2:
                # Sentence fits into current semantic group.
                # --------------------------------------------------
                if projected_length <= self.chunk_size:

                    current_text_parts.append(sentence)
                    current_segments.append(segment)
                    current_length = projected_length

                    continue

                # --------------------------------------------------
                # Case 3:
                # Current group is large enough.
                # Finalize it at a meaningful boundary.
                # --------------------------------------------------
                if current_length >= self.min_chunk_size:

                    groups.append(
                        self._create_group(
                            current_text_parts,
                            current_segments,
                        )
                    )

                    current_text_parts = [
                        sentence
                    ]

                    current_segments = [
                        segment
                    ]

                    current_length = sentence_length

                    continue

                # --------------------------------------------------
                # Case 4:
                # Current group is still too small.
                # Prefer keeping semantically related sentences
                # together even if target size is exceeded slightly.
                # --------------------------------------------------
                current_text_parts.append(sentence)
                current_segments.append(segment)

                current_length = projected_length

                # Prevent pathological growth from a long transcript
                # segment by finalizing once it exceeds 1.5x target.
                if current_length >= int(
                    self.chunk_size * 1.5
                ):
                    groups.append(
                        self._create_group(
                            current_text_parts,
                            current_segments,
                        )
                    )

                    current_text_parts = []
                    current_segments = []
                    current_length = 0

        # Flush remaining content.
        if current_text_parts:
            groups.append(
                self._create_group(
                    current_text_parts,
                    current_segments,
                )
            )

        return groups

    @staticmethod
    def _create_group(
        text_parts: list[str],
        segments: list[TranscriptSegment],
    ) -> dict:
        """
        Create a semantic group with accurate timestamp range.
        """

        return {
            "text": " ".join(text_parts).strip(),
            "start_time": segments[0].start_time,
            "end_time": segments[-1].end_time,
        }

    # ------------------------------------------------------------------
    # Recursive fallback
    # ------------------------------------------------------------------

    def _recursively_split_group(
        self,
        group: dict,
    ) -> list[Document]:
        """
        Recursively split oversized semantic groups.

        Timestamp estimation is performed proportionally according
        to character offsets inside the original semantic group.
        """

        text = group["text"]

        if len(text) <= self.chunk_size:
            return [
                Document(
                    page_content=text,
                    metadata={
                        "chunk_start_time": group[
                            "start_time"
                        ],
                        "chunk_end_time": group[
                            "end_time"
                        ],
                    },
                )
            ]

        temporary_document = Document(
            page_content=text,
            metadata={
                "chunk_start_time": group[
                    "start_time"
                ],
                "chunk_end_time": group[
                    "end_time"
                ],
            },
        )

        chunks = self.recursive_splitter.split_documents(
            [temporary_document]
        )

        total_text_length = len(text)

        if total_text_length == 0:
            return []

        total_duration = (
            group["end_time"]
            - group["start_time"]
        )

        result: list[Document] = []

        search_start = 0

        for chunk in chunks:

            chunk_text = chunk.page_content.strip()

            if not chunk_text:
                continue

            # Locate this chunk inside the original group.
            position = text.find(
                chunk_text,
                search_start,
            )

            if position == -1:
                # Defensive fallback.
                position = search_start

            chunk_start_offset = position

            chunk_end_offset = (
                position + len(chunk_text)
            )

            # Map character position → timestamp.
            start_ratio = (
                chunk_start_offset
                / total_text_length
            )

            end_ratio = (
                min(
                    chunk_end_offset,
                    total_text_length,
                )
                / total_text_length
            )

            chunk_start_time = (
                group["start_time"]
                + total_duration * start_ratio
            )

            chunk_end_time = (
                group["start_time"]
                + total_duration * end_ratio
            )

            result.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_start_time": chunk_start_time,
                        "chunk_end_time": chunk_end_time,
                    },
                )
            )

            search_start = max(
                search_start,
                chunk_end_offset
                - self.chunk_overlap,
            )

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Convert timestamped YouTube transcript segments into
        retrieval-optimized chunks.

        The resulting Documents contain:

        - source
        - video_id
        - chunk_index
        - chunk_start_time
        - chunk_end_time
        - content_length
        """

        if not documents:
            raise ValueError(
                "No transcript documents were provided."
            )

        prepared = self._prepare_segments(
            documents
        )

        if not prepared:
            raise ValueError(
                "Transcript documents contain no usable text."
            )

        # --------------------------------------------------------------
        # Build semantic groups first.
        # --------------------------------------------------------------

        semantic_groups = self._build_semantic_groups(
            prepared
        )

        if not semantic_groups:
            raise ValueError(
                "Unable to create semantic transcript chunks."
            )

        # --------------------------------------------------------------
        # Apply recursive splitting only where necessary.
        # --------------------------------------------------------------

        raw_chunks: list[Document] = []

        for group in semantic_groups:

            group_chunks = self._recursively_split_group(
                group
            )

            raw_chunks.extend(
                group_chunks
            )

        # --------------------------------------------------------------
        # Add production metadata.
        # --------------------------------------------------------------

        first_document = prepared[0][1]

        base_metadata = {
            "source": first_document.metadata.get(
                "source"
            ),
            "video_id": first_document.metadata.get(
                "video_id"
            ),
        }

        final_chunks: list[Document] = []

        for index, chunk in enumerate(raw_chunks):

            metadata = {
                **base_metadata,
                "chunk_index": index,
                "chunk_start_time": float(
                    chunk.metadata.get(
                        "chunk_start_time",
                        0.0,
                    )
                ),
                "chunk_end_time": float(
                    chunk.metadata.get(
                        "chunk_end_time",
                        0.0,
                    )
                ),
                "content_length": len(
                    chunk.page_content
                ),
            }

            final_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Created %d timestamp-aware chunks "
            "from %d transcript segments.",
            len(final_chunks),
            len(prepared),
        )

        return final_chunks