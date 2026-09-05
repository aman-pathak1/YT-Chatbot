

import logging
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from ingestion.youtube import load_youtube_transcript
from ingestion.translator import TranscriptTranslator
from ingestion.splitter import TranscriptSplitter
from retrieval.vector_store import PineconeVectorStoreManager


logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Base exception for indexing failures."""


@dataclass
class IndexingResult:
    """Result returned after indexing a YouTube video."""

    video_url: str
    video_id: str
    transcript_documents: int
    chunk_documents: List[Document]
    indexed_documents: int
    indexed_ids: List[str]


class YouTubeIndexer:
    """
    End-to-end YouTube transcript indexing pipeline.

    Responsibilities:
    1. Fetch YouTube transcript.
    2. Translate/normalize transcript when required.
    3. Split transcript into retrieval chunks.
    4. Store chunks in Pinecone.
    """

    def __init__(
        self,
        translator: TranscriptTranslator,
        splitter: TranscriptSplitter,
        vector_store: PineconeVectorStoreManager,
    ) -> None:
        self.translator = translator
        self.splitter = splitter
        self.vector_store = vector_store

        logger.info("YouTube indexer initialized.")

    def index_video(self, video_url: str) -> IndexingResult:
        """Fetch, process, chunk and index one YouTube video."""

        if not isinstance(video_url, str) or not video_url.strip():
            raise IndexingError(
                "video_url must be a non-empty string."
            )

        video_url = video_url.strip()

        logger.info(
            "Starting YouTube indexing: %s",
            video_url,
        )

        try:
            # ---------------------------------------------------------
            # 1. Transcript extraction
            # ---------------------------------------------------------
            logger.info("Fetching YouTube transcript...")

            transcript_documents = load_youtube_transcript(video_url)

            if not transcript_documents:
                raise IndexingError(
                    "YouTube transcript extraction returned no documents."
                )

            video_id = self._extract_video_id(
                transcript_documents
            )

            logger.info(
                "Transcript fetched successfully | "
                "video_id=%s | segments=%d",
                video_id,
                len(transcript_documents),
            )

            # ---------------------------------------------------------
            # 2. Translation / language normalization
            # ---------------------------------------------------------
            logger.info("Normalizing transcript language...")

            translated_documents = (
                self.translator.translate_documents(
                    transcript_documents
                )
            )

            if not translated_documents:
                raise IndexingError(
                    "Transcript translation returned no documents."
                )

            logger.info(
                "Transcript normalization completed | documents=%d",
                len(translated_documents),
            )

            # ---------------------------------------------------------
            # 3. Timestamp-aware chunking
            # ---------------------------------------------------------
            logger.info(
                "Splitting transcript into retrieval chunks..."
            )

            chunks = self.splitter.split_documents(
                translated_documents
            )

            if not chunks:
                raise IndexingError(
                    "Transcript splitting produced no chunks."
                )

            logger.info(
                "Transcript chunking completed | chunks=%d",
                len(chunks),
            )

            # ---------------------------------------------------------
            # 4. Pinecone indexing
            # ---------------------------------------------------------
            logger.info("Indexing chunks into Pinecone...")

            indexed_ids = self.vector_store.add_documents(chunks)

            if not indexed_ids:
                raise IndexingError(
                    "Pinecone indexing returned no document IDs."
                )

            logger.info(
                "YouTube indexing completed successfully | "
                "video_id=%s | chunks=%d | indexed=%d",
                video_id,
                len(chunks),
                len(indexed_ids),
            )

            return IndexingResult(
                video_url=video_url,
                video_id=video_id,
                transcript_documents=len(transcript_documents),
                chunk_documents=list(chunks),
                indexed_documents=len(indexed_ids),
                indexed_ids=indexed_ids,
            )

        except IndexingError:
            raise

        except Exception as exc:
            logger.exception(
                "YouTube indexing failed: %s",
                exc,
            )

            raise IndexingError(
                f"Failed to index YouTube video: {exc}"
            ) from exc

    @staticmethod
    def _extract_video_id(
        documents: List[Document],
    ) -> str:
        """Extract video ID from transcript metadata."""

        for document in documents:
            video_id = document.metadata.get("video_id")

            if video_id:
                return str(video_id)

        raise IndexingError(
            "Transcript documents do not contain video_id metadata."
        )
