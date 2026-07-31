import os

from voiceclone.config import Settings


def test_settings_defaults_are_sane() -> None:
    s = Settings(_env_file=None)
    assert 0 < s.cv_fraction < 1
    assert 0 < s.eval_holdout_fraction < 1
    assert s.cv_fraction + s.eval_holdout_fraction < 1
    assert s.max_utterance_seconds <= 30.0  # CosyVoice3's speech-token extractor hard cap
    assert s.min_utterance_seconds < s.max_utterance_seconds


def test_settings_reads_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("VOICECLONE_SPEAKER_ID", "test_speaker_123")
    s = Settings(_env_file=None)
    assert s.speaker_id == "test_speaker_123"
    os.environ.pop("VOICECLONE_SPEAKER_ID", None)
