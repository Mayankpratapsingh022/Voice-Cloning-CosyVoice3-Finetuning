"""Central configuration, loaded from environment / .env via pydantic-settings.

Every module that needs a path, volume name, or hyperparameter default reads it from
here rather than hardcoding it, so the whole pipeline can be retargeted (new speaker,
new base checkpoint, new Modal volume names) by editing one .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICECLONE_", env_file=".env", extra="ignore")

    speaker_id: str = Field(default="speaker", description="utt/spk id prefix used throughout the data pipeline")
    base_model_id: str = Field(default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

    dataset_volume: str = "voiceclone-dataset"
    checkpoint_volume: str = "voiceclone-checkpoints"
    pretrained_volume: str = "voiceclone-pretrained"
    hf_cache_volume: str = "voiceclone-hf-cache"

    # CosyVoice3's speech-token extractor hard-caps input length; anything longer is
    # silently skipped by the upstream script rather than erroring, so we enforce the
    # limit ourselves at chunking time to avoid silently losing training data.
    max_utterance_seconds: float = 30.0
    min_utterance_seconds: float = 1.0
    target_sample_rate: int = 24000  # CosyVoice3's native sample_rate (see conf/cosyvoice3.yaml)
    asr_sample_rate: int = 16000  # rate expected by campplus/whisper-based feature extractors

    # Two distinct splits held out from training, serving two different purposes:
    # `cv` feeds validation loss during training (used by average_model.py --val_best),
    # `holdout` is never touched until final objective/human evaluation. Keeping them
    # separate avoids picking a checkpoint and reporting its quality on the same data.
    cv_fraction: float = 0.05
    eval_holdout_fraction: float = 0.05

    modal_app_name: str = "voiceclone-cosyvoice3"


def get_settings() -> Settings:
    return Settings()


class Paths:
    """Container-side paths used consistently across every Modal function.

    Kept as plain class attributes (not env-configurable) because these are mount
    points *inside* the Modal container image, not something a user should retarget.
    """

    REPO_ROOT = Path("/root/CosyVoice")
    PRETRAINED_DIR = Path("/pretrained")
    DATASET_DIR = Path("/dataset")
    CHECKPOINT_DIR = Path("/checkpoints")
    HF_CACHE_DIR = Path("/hf-cache")
