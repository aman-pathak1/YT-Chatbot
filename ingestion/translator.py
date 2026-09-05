from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import torch
try:
    from IndicTransToolkit.processor import IndicProcessor
except ModuleNotFoundError:
    logger.warning("IndicTransToolkit not found in current environment. Please activate venv311.")
    IndicProcessor = None
from langchain_core.documents import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

SOURCE_LANGUAGE = "hin_Deva"
TARGET_LANGUAGE = "eng_Latn"

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 256
DEFAULT_NUM_BEAMS = 1


# ============================================================
# Exceptions
# ============================================================


class TranslationError(Exception):
    """Base exception for transcript translation errors."""


class InvalidTranscriptError(TranslationError):
    """Raised when transcript input is invalid."""


class TranscriptTranslationError(TranslationError):
    """Raised when transcript translation fails."""


class TranslationModelError(TranslationError):
    """Raised when the IndicTrans2 model cannot be loaded."""


# ============================================================
# Internal Data Model
# ============================================================


@dataclass(frozen=True)
class TranscriptSegment:
    """
    Internal representation of one transcript segment/chunk.
    """

    index: int
    text: str
    start_time: float
    end_time: float


# ============================================================
# Transcript Translator
# ============================================================


class TranscriptTranslator:
    """
    Production-oriented Hindi → English transcript translator.

    Translation pipeline:

        LangChain Documents
                ↓
        Transcript Validation
                ↓
        Language Detection
                ↓
        Hindi Segment Selection
                ↓
        Batch Construction
                ↓
        IndicTrans2 Inference
                ↓
        Translation Mapping
                ↓
        English LangChain Documents

    Design goals:

        - IndicTrans2 runs locally.
        - Model is loaded only once.
        - Translation is performed in batches.
        - English text is not sent through the model.
        - Original transcript text is preserved in metadata.
        - Timestamp metadata is preserved.
        - LangChain Document remains the public data format.
        - Component can later be composed into a LangChain Runnable
          pipeline from the application layer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        device: str | None = None,
    ) -> None:
        """
        Initialize the IndicTrans2 translator.

        Args:
            model_name:
                Hugging Face IndicTrans2 model identifier.

            batch_size:
                Maximum number of transcript segments translated
                in one model inference call.

            max_length:
                Maximum generated sequence length.

            num_beams:
                Beam-search width. 1 is intentionally used by
                default for faster local CPU inference.

            device:
                Explicit device such as "cpu" or "cuda".
                If omitted, CUDA is selected when available,
                otherwise CPU is used.
        """

        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than 0."
            )

        if num_beams <= 0:
            raise ValueError(
                "num_beams must be greater than 0."
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams

        self.device = self._resolve_device(device)

        logger.info(
            "Initializing IndicTrans2 translator: "
            "model=%s, device=%s, batch_size=%d",
            self.model_name,
            self.device,
            self.batch_size,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            self.processor = IndicProcessor(
                inference=True
            )

        except Exception as exc:
            logger.exception(
                "Failed to initialize IndicTrans2 model."
            )

            raise TranslationModelError(
                "Unable to load IndicTrans2 translation model."
            ) from exc

        logger.info(
            "IndicTrans2 translator initialized successfully."
        )

    # ============================================================
    # Device
    # ============================================================

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> str:
        """
        Resolve the inference device.
        """

        if requested_device is not None:

            normalized = requested_device.strip().lower()

            if normalized not in {"cpu", "cuda"}:
                raise ValueError(
                    "device must be either 'cpu', 'cuda', or None."
                )

            if normalized == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    "CUDA was requested but is not available."
                )

            return normalized

        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ============================================================
    # Language Detection
    # ============================================================

    @staticmethod
    def _contains_devanagari(
        text: str,
    ) -> bool:
        """
        Detect Devanagari characters.

        Hindi transcript text is expected to contain
        Devanagari characters.
        """

        return bool(
            re.search(
                r"[\u0900-\u097F]",
                text,
            )
        )

    @classmethod
    def _requires_translation(
        cls,
        text: str,
    ) -> bool:
        """
        Determine whether IndicTrans2 should process the text.

        Current model supports Hindi written in Devanagari
        as the source language.

        English / non-Devanagari text is therefore preserved
        without translation.
        """

        if not text.strip():
            return False

        return cls._contains_devanagari(text)

    # ============================================================
    # Transcript Validation
    # ============================================================

    @staticmethod
    def _extract_segment(
        document: Document,
        index: int,
    ) -> TranscriptSegment | None:
        """
        Convert a LangChain Document into a validated
        TranscriptSegment.

        Supports both:

            start_time / end_time

        and chunk-level metadata:

            chunk_start_time / chunk_end_time
        """

        text = document.page_content.strip()

        if not text:
            return None

        metadata = document.metadata

        try:
            start_value = metadata.get(
                "start_time",
                metadata.get(
                    "chunk_start_time",
                    0.0,
                ),
            )

            end_value = metadata.get(
                "end_time",
                metadata.get(
                    "chunk_end_time",
                    start_value,
                ),
            )

            start_time = float(start_value)
            end_time = float(end_value)

        except (TypeError, ValueError) as exc:

            raise InvalidTranscriptError(
                f"Invalid timestamp metadata at "
                f"document index {index}."
            ) from exc

        if start_time < 0:
            raise InvalidTranscriptError(
                f"Negative start_time at document "
                f"index {index}."
            )

        if end_time < start_time:
            raise InvalidTranscriptError(
                f"end_time is earlier than start_time "
                f"at document index {index}."
            )

        return TranscriptSegment(
            index=index,
            text=text,
            start_time=start_time,
            end_time=end_time,
        )

    def _prepare_segments(
        self,
        documents: Sequence[Document],
    ) -> list[TranscriptSegment]:
        """
        Validate and chronologically order transcript documents.
        """

        if not documents:
            raise InvalidTranscriptError(
                "No transcript documents were provided."
            )

        segments: list[TranscriptSegment] = []

        for index, document in enumerate(documents):

            if not isinstance(document, Document):
                raise InvalidTranscriptError(
                    f"Expected LangChain Document at index "
                    f"{index}, got {type(document).__name__}."
                )

            segment = self._extract_segment(
                document=document,
                index=index,
            )

            if segment is not None:
                segments.append(segment)

        if not segments:
            raise InvalidTranscriptError(
                "Transcript contains no usable text."
            )

        segments.sort(
            key=lambda segment: segment.start_time
        )

        return segments

    # ============================================================
    # Batch Construction
    # ============================================================

    def _create_batches(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> list[list[TranscriptSegment]]:
        """
        Split translation candidates into fixed-size batches.
        """

        return [
            list(segments[index:index + self.batch_size])
            for index in range(
                0,
                len(segments),
                self.batch_size,
            )
        ]

    # ============================================================
    # IndicTrans2 Inference
    # ============================================================

    def _translate_batch(
        self,
        batch: Sequence[TranscriptSegment],
    ) -> dict[int, str]:
        """
        Translate one batch using IndicTrans2.

        Returns:
            Mapping:
                original segment index → English translation
        """

        if not batch:
            return {}

        texts = [
            segment.text
            for segment in batch
        ]

        segment_indices = [
            segment.index
            for segment in batch
        ]

        try:
            # ----------------------------------------------------
            # IndicTrans2 preprocessing
            # ----------------------------------------------------

            processed_batch = (
                self.processor.preprocess_batch(
                    texts,
                    src_lang=SOURCE_LANGUAGE,
                    tgt_lang=TARGET_LANGUAGE,
                )
            )

            # ----------------------------------------------------
            # Tokenization
            # ----------------------------------------------------

            inputs = self.tokenizer(
                processed_batch,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # ----------------------------------------------------
            # Generation
            # ----------------------------------------------------

            with torch.no_grad():

                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    use_cache=False,
                )

            # ----------------------------------------------------
            # Decode
            # ----------------------------------------------------

            generated_texts = (
                self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
            )

            # ----------------------------------------------------
            # IndicTrans2 postprocessing
            # ----------------------------------------------------

            translations = (
                self.processor.postprocess_batch(
                    generated_texts,
                    lang=TARGET_LANGUAGE,
                )
            )

        except Exception as exc:

            logger.exception(
                "IndicTrans2 inference failed for "
                "batch containing segment indices %s.",
                segment_indices,
            )

            raise TranscriptTranslationError(
                "IndicTrans2 failed to translate a transcript batch."
            ) from exc

        if len(translations) != len(batch):

            raise TranscriptTranslationError(
                "IndicTrans2 returned an unexpected number "
                "of translations. "
                f"Expected {len(batch)}, "
                f"received {len(translations)}."
            )

        translation_map: dict[int, str] = {}

        for segment, translation in zip(
            batch,
            translations,
        ):

            translated_text = translation.strip()

            if not translated_text:

                raise TranscriptTranslationError(
                    "IndicTrans2 returned an empty translation "
                    f"for segment {segment.index}."
                )

            translation_map[
                segment.index
            ] = translated_text

        return translation_map

    # ============================================================
    # Public Translation API
    # ============================================================

    def translate_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Translate Hindi transcript documents into English.

        English documents are preserved without model inference.

        The returned Documents preserve:

            - source
            - video_id
            - timestamps
            - chunk metadata
            - original transcript text

        and add:

            - language_normalized
            - translation_model
        """

        segments = self._prepare_segments(
            documents
        )

        # --------------------------------------------------------
        # Build lookup once.
        #
        # This avoids repeatedly scanning the complete document
        # list for every segment.
        # --------------------------------------------------------

        document_map = {
            index: document
            for index, document in enumerate(documents)
        }

        # --------------------------------------------------------
        # Separate Hindi and already-English segments.
        # --------------------------------------------------------

        translation_candidates = [
            segment
            for segment in segments
            if self._requires_translation(
                segment.text
            )
        ]

        translation_map: dict[int, str] = {}

        # --------------------------------------------------------
        # Translate Hindi segments.
        # --------------------------------------------------------

        if translation_candidates:

            batches = self._create_batches(
                translation_candidates
            )

            logger.info(
                "Translating %d Hindi segments "
                "in %d batches.",
                len(translation_candidates),
                len(batches),
            )

            for batch_number, batch in enumerate(
                batches,
                start=1,
            ):

                logger.info(
                    "Processing translation batch "
                    "%d/%d (%d segments).",
                    batch_number,
                    len(batches),
                    len(batch),
                )

                batch_translation = (
                    self._translate_batch(
                        batch
                    )
                )

                translation_map.update(
                    batch_translation
                )

        # --------------------------------------------------------
        # Build final Documents.
        # --------------------------------------------------------

        translated_documents: list[Document] = []

        for segment in segments:

            original_document = document_map[
                segment.index
            ]

            if segment.index in translation_map:

                english_text = translation_map[
                    segment.index
                ]

            else:

                # Already English / unsupported non-Devanagari
                # text is preserved as-is.
                english_text = segment.text

            metadata = dict(
                original_document.metadata
            )

            metadata.update(
                {
                    "original_text": segment.text,
                    "language_normalized": True,
                    "translation_model": (
                        self.model_name
                        if segment.index
                        in translation_map
                        else None
                    ),
                    "source_language": (
                        SOURCE_LANGUAGE
                        if segment.index
                        in translation_map
                        else "eng_Latn"
                    ),
                    "target_language": TARGET_LANGUAGE,
                }
            )

            translated_documents.append(
                Document(
                    page_content=english_text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Transcript translation completed: "
            "%d documents processed, %d translated.",
            len(translated_documents),
            len(translation_map),
        )

        return translated_documents