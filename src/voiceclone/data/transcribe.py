"""Transcript generation and verification, via either a local faster-whisper model or
OpenAI's hosted Whisper API.

Two uses regardless of backend: (1) transcribing unscripted/conversational chunks that
have no ground-truth text, and (2) sanity-checking scripted chunks (e.g. Harvard
sentences) against the text you meant to read, so a misread sentence doesn't silently
poison the training manifest with a text/audio mismatch.

Backend tradeoff: local (faster-whisper large-v3) is free but CPU-bound — on a modern
laptop CPU it runs close to real-time, so an hour of audio takes roughly an hour.
OpenAI's API is cloud/GPU-backed and requests are parallelized here, so the same hour
of audio typically finishes in a few minutes, at a cost of roughly $0.006/minute of
audio (openai.com/pricing — verify current rate before relying on it for a much
larger dataset). Needs `OPENAI_API_KEY` set in the environment (or `.env`); never pass
it as a function argument or log it.
"""

from __future__ import annotations

import difflib
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from voiceclone.logging_utils import get_logger

if typing.TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from openai import OpenAI

logger = get_logger(__name__)

_LOCAL_MODEL_CACHE: dict[str, WhisperModel] = {}
_OPENAI_CLIENT: OpenAI | None = None

DEFAULT_LOCAL_MODEL = "large-v3"
DEFAULT_OPENAI_MODEL = "whisper-1"


def _get_local_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    from faster_whisper import WhisperModel

    key = f"{model_size}:{device}:{compute_type}"
    if key not in _LOCAL_MODEL_CACHE:
        logger.info("loading faster-whisper model %s", key)
        _LOCAL_MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _LOCAL_MODEL_CACHE[key]


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI

        # Reads OPENAI_API_KEY from the environment itself — never pass the key
        # through our own code/config so it can't end up in a log line or a
        # committed .env-adjacent file by accident.
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


@dataclass(frozen=True)
class TranscriptMismatch:
    utt_id: str
    expected: str
    transcribed: str
    similarity: float


def transcribe(
    wav_path: Path,
    backend: str = "local",
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
) -> str:
    """Transcribe one WAV. `model` defaults per backend (large-v3 local, whisper-1 OpenAI)."""
    if backend == "local":
        m = _get_local_model(model or DEFAULT_LOCAL_MODEL, device, compute_type)
        segments, _info = m.transcribe(str(wav_path), language=language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    if backend == "openai":
        client = _get_openai_client()
        with open(wav_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model or DEFAULT_OPENAI_MODEL, file=f, language=language
            )
        return result.text.strip()
    raise ValueError(f"unknown transcribe backend '{backend}', expected 'local' or 'openai'")


def transcribe_batch(
    wav_paths: list[Path],
    backend: str = "local",
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    max_workers: int = 8,
) -> dict[str, str]:
    """utt_id (filename stem) -> transcript, for chunks with no pre-existing script text.

    OpenAI's backend fans out concurrently (I/O-bound API calls) since that's most of
    the speed advantage over the local backend; local runs sequentially since a single
    CPU-bound faster-whisper model instance isn't meaningfully parallelizable this way.
    """
    if backend == "local":
        return {p.stem: transcribe(p, backend, model, device, compute_type, language) for p in wav_paths}

    if backend == "openai":
        out: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(transcribe, p, backend, model, device, compute_type, language): p
                for p in wav_paths
            }
            for future in as_completed(futures):
                path = futures[future]
                out[path.stem] = future.result()
        return out

    raise ValueError(f"unknown transcribe backend '{backend}', expected 'local' or 'openai'")


def _normalize_for_compare(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()


def verify_against_script(
    utt_to_wav: dict[str, Path],
    utt_to_expected_text: dict[str, str],
    backend: str = "local",
    model: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    similarity_threshold: float = 0.85,
) -> list[TranscriptMismatch]:
    """Flag chunks whose audio doesn't match the sentence you intended to read.

    Doesn't correct anything itself — returns mismatches so you can decide per-case
    whether to re-record, drop the utterance, or accept the ASR transcript instead
    of the script text (which is the right call for spontaneous speech read imperfectly).
    """
    mismatches: list[TranscriptMismatch] = []
    for utt_id, wav_path in utt_to_wav.items():
        expected = utt_to_expected_text.get(utt_id)
        if expected is None:
            continue
        transcribed = transcribe(wav_path, backend, model, device, compute_type, language)
        ratio = difflib.SequenceMatcher(
            None, _normalize_for_compare(expected), _normalize_for_compare(transcribed)
        ).ratio()
        if ratio < similarity_threshold:
            mismatches.append(TranscriptMismatch(utt_id, expected, transcribed, ratio))
    if mismatches:
        logger.warning("%d/%d utterances failed script verification", len(mismatches), len(utt_to_wav))
    return mismatches
