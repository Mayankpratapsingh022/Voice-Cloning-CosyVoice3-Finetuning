"""Unit tests for the deterministic segment-processing logic in audio_preprocess.py.

webrtcvad's actual speech/non-speech classification isn't exercised here — that needs
real speech fixtures to be meaningful, not synthetic tones, and belongs in a manual
pilot-run sanity check (see PLAN.md Section 5), not a fast unit test. What's tested
here is the segment merging/splitting/normalization logic that runs regardless of
what the VAD decided.
"""

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from voiceclone.data.audio_preprocess import (
    SpeechSegment,
    _merge_close_segments,
    _split_overlong,
    convert_to_wav,
    normalize_loudness,
    probe_duration_s,
    sanitize_id,
)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_sanitize_id_strips_whitespace_and_punctuation() -> None:
    # the exact real-world case that broke extract_embedding.py: CosyVoice3's own
    # manifest readers use a plain `line.split()`, so any whitespace in an id or path
    # silently truncates it rather than erroring
    raw = "Implementing DeepSeek LLM from Scratch in Pytorch (Transformer + RVQ + CTC) [L-E3WJvztZ0]"
    result = sanitize_id(raw)
    assert " " not in result
    assert "(" not in result and ")" not in result
    assert "[" not in result and "]" not in result
    assert "L-E3WJvztZ0" in result  # the hyphen survives -- it's an allowed character


def test_sanitize_id_collapses_repeats_and_strips_edges() -> None:
    assert sanitize_id("  hello   world  ") == "hello_world"
    assert sanitize_id("a___b") == "a_b"


def test_sanitize_id_preserves_already_safe_strings() -> None:
    assert sanitize_id("mayank_session1_00042") == "mayank_session1_00042"


def test_merge_close_segments_bridges_short_gaps() -> None:
    segments = [SpeechSegment(0.0, 1.0), SpeechSegment(1.2, 2.0), SpeechSegment(5.0, 6.0)]
    merged = _merge_close_segments(segments)  # SILENCE_MERGE_GAP_S = 0.3, so 0.2s gap bridges, 3.0s doesn't
    assert merged == [SpeechSegment(0.0, 2.0), SpeechSegment(5.0, 6.0)]


def test_merge_close_segments_empty_input() -> None:
    assert _merge_close_segments([]) == []


def test_split_overlong_leaves_short_segments_alone() -> None:
    seg = SpeechSegment(0.0, 10.0)
    assert _split_overlong(seg, max_s=30.0) == [seg]


def test_split_overlong_splits_into_equal_chunks_under_cap() -> None:
    seg = SpeechSegment(0.0, 65.0)
    chunks = _split_overlong(seg, max_s=30.0)
    assert len(chunks) == 3  # ceil(65/30)
    assert all(c.duration_s <= 30.0 for c in chunks)
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == pytest.approx(65.0)
    # contiguous, no gaps or overlaps introduced
    for a, b in zip(chunks, chunks[1:], strict=False):  # deliberately different lengths (pairwise iteration)
        assert a.end_s == pytest.approx(b.start_s)


def test_normalize_loudness_hits_target(tmp_path) -> None:
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    normalized = normalize_loudness(audio, sr, target_lufs=-23.0)

    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(normalized)
    assert measured == pytest.approx(-23.0, abs=0.5)


def test_normalize_loudness_handles_near_silence_without_crashing() -> None:
    sr = 24000
    silence = np.zeros(sr, dtype=np.float32)
    result = normalize_loudness(silence, sr)
    assert result.shape == silence.shape


@requires_ffmpeg
def test_convert_to_wav_transcodes_mp3(tmp_path) -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    src_wav = tmp_path / "tone.wav"
    sf.write(src_wav, tone, sr)

    src_mp3 = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_wav), str(src_mp3)],
        check=True, capture_output=True,
    )

    dest_wav = tmp_path / "converted" / "tone.wav"
    result = convert_to_wav(src_mp3, dest_wav, sample_rate=24000)

    assert result == dest_wav
    assert dest_wav.exists()
    info = sf.info(str(dest_wav))
    assert info.samplerate == 24000
    assert info.channels == 1
    assert probe_duration_s(dest_wav) == pytest.approx(1.0, abs=0.15)  # mp3 encoding pads slightly


@requires_ffmpeg
def test_convert_to_wav_raises_on_bad_input(tmp_path) -> None:
    bogus = tmp_path / "not_audio.mp3"
    bogus.write_text("this is not an audio file")

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        convert_to_wav(bogus, tmp_path / "out.wav", sample_rate=24000)
