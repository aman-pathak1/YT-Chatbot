# ingestion/youtube.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class YouTubeError(Exception):
    """Base exception for YouTube ingestion errors."""


class InvalidYouTubeURLError(YouTubeError):
    """Raised when the supplied URL is not a valid YouTube URL."""


class TranscriptNotFoundError(YouTubeError):
    """Raised when a suitable transcript cannot be found."""


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def extract_video_id(url: str) -> str:
    """
    Extract a YouTube video ID from common YouTube URL formats.

    Supported:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - youtube.com/shorts/VIDEO_ID
    - youtube.com/embed/VIDEO_ID
    """

    if not isinstance(url, str) or not url.strip():
        raise InvalidYouTubeURLError("YouTube URL cannot be empty.")

    url = url.strip()

    # Allow users to enter just the video ID.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise InvalidYouTubeURLError("Invalid URL.") from exc

    hostname = parsed.hostname

    if not hostname:
        raise InvalidYouTubeURLError("Invalid YouTube URL.")

    hostname = hostname.lower().removeprefix("www.")

    # youtu.be/VIDEO_ID
    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    # youtube.com/...
    elif hostname in {"youtube.com", "m.youtube.com"}:

        # /watch?v=VIDEO_ID
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]

        # /shorts/VIDEO_ID
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/", 1)[1].split("/")[0]

        # /embed/VIDEO_ID
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/")[0]

        else:
            video_id = None

    else:
        raise InvalidYouTubeURLError(
            "URL must belong to youtube.com or youtu.be."
        )

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise InvalidYouTubeURLError(
            "Could not extract a valid YouTube video ID."
        )

    return video_id


def fetch_transcript(
    video_id: str,
    languages: tuple[str, ...] = ("en", "hi"),
) -> list[TranscriptSegment]:
    """
    Fetch transcript with preferred language fallback.
    """

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise InvalidYouTubeURLError("Invalid YouTube video ID.")

    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(
            video_id,
            languages=list(languages),
        )

    except Exception as exc:
        logger.exception(
            "Failed to fetch transcript for video %s",
            video_id,
        )
        raise TranscriptNotFoundError(
            f"No suitable transcript found for video: {video_id}"
        ) from exc

    segments: list[TranscriptSegment] = []

    for snippet in transcript:
        text = getattr(snippet, "text", "").strip()

        if not text:
            continue

        segments.append(
            TranscriptSegment(
                text=text,
                start=float(getattr(snippet, "start", 0.0)),
                duration=float(getattr(snippet, "duration", 0.0)),
            )
        )

    if not segments:
        raise TranscriptNotFoundError(
            f"Transcript is empty for video: {video_id}"
        )

    return segments


def transcript_to_documents(
    segments: list[TranscriptSegment],
    video_id: str,
    source_url: str,
) -> list[Document]:
    """
    Convert transcript segments into LangChain Documents.

    Each segment keeps timestamp metadata so that
    citations can later point to the relevant YouTube time.
    """

    documents: list[Document] = []

    for index, segment in enumerate(segments):

        document = Document(
            page_content=segment.text,
            metadata={
                "source": source_url,
                "video_id": video_id,
                "segment_index": index,
                "start_time": segment.start,
                "end_time": segment.end,
                "youtube_timestamp": int(segment.start),
            },
        )

        documents.append(document)

    return documents


def load_youtube_transcript(
    url: str,
    languages: tuple[str, ...] = ("en", "hi"),
) -> list[Document]:
    """
    Main entry point for YouTube transcript ingestion.
    """

    video_id = extract_video_id(url)

    segments = fetch_transcript(
        video_id=video_id,
        languages=languages,
    )

    documents = transcript_to_documents(
        segments=segments,
        video_id=video_id,
        source_url=url,
    )

    logger.info(
        "Loaded %d transcript segments from video %s",
        len(documents),
        video_id,
    )

    return documents