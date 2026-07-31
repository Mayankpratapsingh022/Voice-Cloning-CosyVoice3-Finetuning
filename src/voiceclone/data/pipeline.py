"""Local orchestration: raw recording sessions -> chunked, transcribed, split manifests.

Runs entirely on your machine, no GPU/Modal needed — matches the plan's rule that only
the CosyVoice-repo-dependent stages (speech-token extraction, training) pay for rented
compute. Output of this module is what gets uploaded to the Modal dataset Volume.
"""

from __future__ import annotations

from pathlib import Path

from voiceclone.config import Settings
from voiceclone.data.audio_preprocess import (
    SUPPORTED_SESSION_EXTENSIONS,
    chunk_recording,
    convert_to_wav,
    probe_duration_s,
)
from voiceclone.data.manifest import prepare_speaker_manifests
from voiceclone.data.transcribe import transcribe_batch, verify_against_script
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)


def _transcribe_session(
    session_prefix: str,
    chunk_paths: list[Path],
    expected_sentences: list[str] | None,
    transcribe_backend: str,
    transcribe_model: str | None,
    whisper_device: str,
) -> dict[str, str]:
    """Prefer clean script text over raw ASR when we're confident the two line up.

    If VAD produced exactly as many chunks as sentences in the script (the common
    case for a Harvard-sentences-style session read with a pause between each one),
    zip them in order, verify each chunk's audio actually matches via ASR, and use the
    script text — it's cleaner and better-punctuated than raw ASR output. Any chunk
    that fails verification, or a chunk count mismatch, falls back to plain ASR so a
    single misalignment doesn't cost the whole session.
    """
    ordered_paths = sorted(chunk_paths, key=lambda p: p.stem)

    if expected_sentences is None or len(expected_sentences) != len(ordered_paths):
        if expected_sentences is not None:
            logger.warning(
                "%s: %d chunks vs %d expected sentences, falling back to ASR-only transcripts",
                session_prefix,
                len(ordered_paths),
                len(expected_sentences),
            )
        return transcribe_batch(
            ordered_paths, backend=transcribe_backend, model=transcribe_model, device=whisper_device
        )

    candidate_text = {p.stem: sentence for p, sentence in zip(ordered_paths, expected_sentences, strict=True)}
    utt_to_wav = {p.stem: p for p in ordered_paths}
    mismatches = verify_against_script(
        utt_to_wav, candidate_text, backend=transcribe_backend, model=transcribe_model, device=whisper_device
    )
    mismatched_ids = {m.utt_id for m in mismatches}
    if mismatched_ids:
        logger.warning(
            "%s: %d/%d chunks didn't match their expected sentence, re-transcribing those via ASR: %s",
            session_prefix,
            len(mismatched_ids),
            len(ordered_paths),
            sorted(mismatched_ids),
        )
        fallback = transcribe_batch(
            [utt_to_wav[u] for u in mismatched_ids],
            backend=transcribe_backend,
            model=transcribe_model,
            device=whisper_device,
        )
        candidate_text.update(fallback)
    return candidate_text


def run_local_data_prep(
    raw_sessions_dir: Path,
    dest_root: Path,
    settings: Settings,
    scripted_sentences: dict[str, list[str]] | None = None,
    transcribe_backend: str = "local",
    transcribe_model: str | None = None,
    whisper_device: str = "cpu",
) -> dict[str, Path]:
    """Chunk every recording under `raw_sessions_dir`, transcribe/verify, and write
    train, cv, and holdout manifests under `dest_root` (see `manifest.split_dataset`).

    `raw_sessions_dir` should contain one (or more) long-form recordings per session —
    e.g. `harvard_session1.wav`, `youtube_tutorial_3.mp3`. Each session gets its own
    utterance-id prefix so chunks never collide across sessions. Formats other than
    WAV (mp3/m4a/flac/ogg — see `SUPPORTED_SESSION_EXTENSIONS`) are transcoded to WAV
    first via ffmpeg; this assumes the source is your voice essentially alone on the
    track (no background music/other speakers) — VAD chunking has no way to separate
    your voice from anything else mixed into the same channel.

    `scripted_sentences`, if given, maps a *session file stem* to the ordered list of
    sentences you read in that session, enabling script-text-over-ASR (see
    `_transcribe_session`). Omit an entry (or pass None entirely) for unscripted /
    conversational sessions — those are transcribed from scratch via ASR.

    `transcribe_backend` is `"local"` (free, CPU-bound faster-whisper) or `"openai"`
    (paid, fast, requires `OPENAI_API_KEY` in the environment) — see transcribe.py's
    module docstring for the speed/cost tradeoff.
    """
    session_paths = sorted(
        p for ext in SUPPORTED_SESSION_EXTENSIONS for p in raw_sessions_dir.glob(f"*{ext}")
    )
    if not session_paths:
        raise FileNotFoundError(
            f"no session recordings ({', '.join(SUPPORTED_SESSION_EXTENSIONS)}) found under {raw_sessions_dir}"
        )

    conversion_staging = dest_root / "_converted"
    chunks_dir = dest_root / "chunks"
    utt_to_wav: dict[str, Path] = {}
    utt_to_text: dict[str, str] = {}

    for session_path in session_paths:
        session_prefix = f"{settings.speaker_id}_{session_path.stem}"

        if session_path.suffix.lower() != ".wav":
            wav_path = conversion_staging / f"{session_path.stem}.wav"
            logger.info("converting %s -> %s", session_path.name, wav_path)
            convert_to_wav(session_path, wav_path, settings.target_sample_rate)
            session_path = wav_path

        written = chunk_recording(
            session_path,
            chunks_dir,
            utt_prefix=session_prefix,
            target_sample_rate=settings.target_sample_rate,
            min_utterance_s=settings.min_utterance_seconds,
            max_utterance_s=settings.max_utterance_seconds,
        )
        if not written:
            continue

        expected = (scripted_sentences or {}).get(session_path.stem)
        session_text = _transcribe_session(
            session_prefix, written, expected, transcribe_backend, transcribe_model, whisper_device
        )
        utt_to_text.update(session_text)
        for p in written:
            utt_to_wav[p.stem] = p

    logger.info("chunked %d sessions into %d utterances total", len(session_paths), len(utt_to_wav))
    total_hours = sum(probe_duration_s(p) for p in utt_to_wav.values()) / 3600
    logger.info("total usable audio: %.2f hours", total_hours)

    return prepare_speaker_manifests(
        utt_to_wav,
        utt_to_text,
        settings.speaker_id,
        dest_root,
        settings.cv_fraction,
        settings.eval_holdout_fraction,
    )
