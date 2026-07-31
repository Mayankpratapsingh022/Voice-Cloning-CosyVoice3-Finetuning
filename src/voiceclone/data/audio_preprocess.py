"""Turn raw recording sessions into clean, CosyVoice3-ready utterance chunks.

CosyVoice3's speech-token extractor (tools/extract_speech_token.py in the upstream
repo) silently drops any utterance longer than 30 seconds rather than erroring, which
would otherwise mean losing training data without a clear signal. This module enforces
that ceiling (and a floor to discard unusably short fragments) at chunk time, using
WebRTC VAD to split on natural speech/silence boundaries instead of hard-cutting
mid-word.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import webrtcvad
from scipy import signal

from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

VAD_SAMPLE_RATE = 16000  # only rate/frame combo webrtcvad's aggressiveness modes support well
VAD_FRAME_MS = 30
SILENCE_MERGE_GAP_S = 0.3  # gaps shorter than this get bridged so mid-sentence pauses don't fragment an utterance
TARGET_LUFS = -23.0

# Extensions run_local_data_prep will pick up as session recordings, in addition to
# .wav. Anything not already .wav gets run through `convert_to_wav` first — soundfile's
# format support varies by platform/libsndfile version, so we don't rely on it for
# compressed formats; ffmpeg is the one dependency guaranteed to handle all of these.
SUPPORTED_SESSION_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def sanitize_id(text: str) -> str:
    """Make a string safe to use as (part of) a CosyVoice3 utterance id.

    wav.scp/text/utt2spk/spk2utt are whitespace-delimited, and critically,
    CosyVoice3's own tools/extract_embedding.py and extract_speech_token.py parse
    them with a plain `line.split()` (no maxsplit) -- so any whitespace inside an
    id or path silently truncates it rather than erroring loudly. This showed up in
    practice: chunk filenames derived directly from source video titles (e.g.
    "Implementing DeepSeek LLM from Scratch...") broke embedding extraction with a
    "file not found" pointing at a path truncated at the first space.

    Keeps only alphanumerics, underscore, and hyphen; collapses everything else
    (spaces, brackets, unicode punctuation) to a single underscore.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return re.sub(r"_+", "_", slug).strip("_")


def convert_to_wav(src_path: Path, dest_path: Path, sample_rate: int) -> Path:
    """Transcode any ffmpeg-readable audio file (mp3/m4a/flac/...) to mono WAV.

    Used for session recordings that don't already arrive as WAV — e.g. MP3s pulled
    from existing narration/video content. Runs before VAD chunking, not as part of
    it, so `chunk_recording` only ever has to deal with one input format.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-ac", "1", "-ar", str(sample_rate),
        str(dest_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {src_path} -> {dest_path}:\n{result.stderr}")
    return dest_path


@dataclass(frozen=True)
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    num_samples = round(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples).astype(np.float32)


def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    aggressiveness: int = 2,
) -> list[SpeechSegment]:
    """Frame-level VAD over a mono float32 signal, merged into speech segments.

    `aggressiveness` is webrtcvad's own 0-3 scale (0 = most permissive about calling
    something speech, 3 = strictest); 2 is a reasonable default for a quiet-room home
    recording setup.
    """
    vad_audio = _resample(audio, sample_rate, VAD_SAMPLE_RATE)
    pcm = _to_pcm16_bytes(vad_audio)

    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 2 bytes/sample
    frames = [pcm[i : i + frame_len] for i in range(0, len(pcm) - frame_len, frame_len)]

    flags = [vad.is_speech(f, VAD_SAMPLE_RATE) for f in frames]

    segments: list[SpeechSegment] = []
    seg_start: float | None = None
    for i, is_speech in enumerate(flags):
        t = i * VAD_FRAME_MS / 1000
        if is_speech and seg_start is None:
            seg_start = t
        elif not is_speech and seg_start is not None:
            segments.append(SpeechSegment(seg_start, t))
            seg_start = None
    if seg_start is not None:
        segments.append(SpeechSegment(seg_start, len(flags) * VAD_FRAME_MS / 1000))

    return _merge_close_segments(segments)


def _merge_close_segments(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start_s - merged[-1].end_s <= SILENCE_MERGE_GAP_S:
            merged[-1] = SpeechSegment(merged[-1].start_s, seg.end_s)
        else:
            merged.append(seg)
    return merged


def _split_overlong(seg: SpeechSegment, max_s: float) -> list[SpeechSegment]:
    """Hard-split a segment that exceeds CosyVoice3's utterance length cap.

    There's no silence to split on inside a segment by construction (VAD already
    merged only speech-contiguous spans), so this cuts at fixed-length boundaries.
    Prefer catching this at the recording stage (pause between sentences) over relying
    on this fallback, since a mid-word cut can teach the model bad prosody.
    """
    if seg.duration_s <= max_s:
        return [seg]
    logger.warning(
        "segment %.1fs exceeds %.1fs cap, hard-splitting — re-record with pauses if this recurs",
        seg.duration_s,
        max_s,
    )
    n_chunks = int(np.ceil(seg.duration_s / max_s))
    chunk_len = seg.duration_s / n_chunks
    return [
        SpeechSegment(seg.start_s + i * chunk_len, seg.start_s + (i + 1) * chunk_len)
        for i in range(n_chunks)
    ]


def normalize_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return audio  # near-silent chunk; leave as-is rather than dividing by zero-ish loudness
    return pyln.normalize.loudness(audio, loudness, target_lufs).astype(np.float32)


def chunk_recording(
    src_path: Path,
    out_dir: Path,
    utt_prefix: str,
    target_sample_rate: int,
    min_utterance_s: float,
    max_utterance_s: float,
    vad_aggressiveness: int = 2,
) -> list[Path]:
    """Split one raw recording session into normalized, CosyVoice3-sized utterance WAVs.

    Returns the paths written, named `{utt_prefix}_{index:05d}.wav`, ready to be picked
    up by manifest.py.
    """
    audio, sr = sf.read(src_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix any accidental stereo capture

    segments = detect_speech_segments(audio, sr, vad_aggressiveness)
    segments = [s for seg in segments for s in _split_overlong(seg, max_utterance_s)]
    segments = [s for s in segments if s.duration_s >= min_utterance_s]

    if not segments:
        logger.warning("no usable speech segments found in %s", src_path)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, seg in enumerate(segments):
        start_sample = int(seg.start_s * sr)
        end_sample = int(seg.end_s * sr)
        chunk = audio[start_sample:end_sample]
        chunk = _resample(chunk, sr, target_sample_rate)
        chunk = normalize_loudness(chunk, target_sample_rate)

        out_path = out_dir / f"{utt_prefix}_{i:05d}.wav"
        sf.write(out_path, chunk, target_sample_rate, subtype="PCM_16")
        written.append(out_path)

    logger.info("%s -> %d utterance chunks (%s)", src_path.name, len(written), out_dir)
    return written


def probe_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
