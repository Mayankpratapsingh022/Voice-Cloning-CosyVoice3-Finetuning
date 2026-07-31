"""Central configuration, loaded from environment / .env via pydantic-settings.

Every module that needs a path, identifier, or hyperparameter default reads it from
here rather than hardcoding it, so the whole pipeline can be retargeted (new speaker,
new base checkpoint, new RunPod resources) by editing one .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICECLONE_", env_file=".env", extra="ignore")

    speaker_id: str = Field(default="speaker", description="utt/spk id prefix used throughout the data pipeline")
    base_model_id: str = Field(default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

    # Private HuggingFace dataset repo the prepared training data is pushed to and
    # pulled from. If left blank, defaults to "{hf_username}/voiceclone-{speaker_id}-dataset"
    # at upload time (see data/hf_dataset.py), using whoami() against HF_TOKEN.
    hf_dataset_repo_id: str = ""

    # A RunPod Network Volume created once via the RunPod web console (runpodctl has
    # no volume-creation command) and reused across pods/sessions. Required before any
    # `voiceclone pod ...` / `voiceclone train ...` command that touches RunPod.
    runpod_network_volume_id: str = ""
    # 24GB comfortably covers CosyVoice3's 0.5B params (trained one component at a
    # time, under AMP), and this is the best price/availability tradeoff of what's on
    # RunPod's "latest generation" list for this account/region at time of writing.
    # Re-verify with `voiceclone pod gpus` before a real run, pricing/availability shift.
    runpod_gpu_type: str = "NVIDIA GeForce RTX 4090"
    runpod_pod_name: str = "voiceclone"

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


def get_settings() -> Settings:
    return Settings()


class RemotePaths:
    """Paths as they exist on the RunPod pod's Network Volume, mounted at /workspace.

    Kept as plain class attributes (not env-configurable) since these are fixed mount
    points inside the remote environment, not something a user should retarget.
    """

    WORKSPACE = Path("/workspace")
    REPO_ROOT = WORKSPACE / "CosyVoice"
    VENV = WORKSPACE / "venv"
    PRETRAINED_DIR = WORKSPACE / "pretrained"
    DATASET_DIR = WORKSPACE / "dataset"
    CHECKPOINT_DIR = WORKSPACE / "checkpoints"
    HF_CACHE_DIR = WORKSPACE / "hf-cache"


def cosyvoice_pythonpath() -> list[str]:
    """The two entries CosyVoice3 needs on sys.path to be importable.

    It is a cloned repo, not a pip-installed package, so `import cosyvoice` only
    resolves if the repo root is on the path; Matcha-TTS is a git submodule the code
    imports directly (upstream's own example.py does `sys.path.append` for it). This
    mirrors what `examples/libritts/cosyvoice3/path.sh` exports before running
    anything.
    """
    return [str(RemotePaths.REPO_ROOT), str(RemotePaths.REPO_ROOT / "third_party" / "Matcha-TTS")]


def cosyvoice_subprocess_env() -> dict[str, str]:
    """os.environ plus the PYTHONPATH CosyVoice3's scripts need.

    Setting cwd=REPO_ROOT is not sufficient and this is an easy trap: torch's
    distributed launcher spawns the training script as a subprocess, where sys.path[0]
    becomes the *script's* directory (cosyvoice/bin/) rather than the working
    directory. Confirmed live as ModuleNotFoundError: No module named 'cosyvoice'.
    The feature-extraction scripts appeared to work without this only because none of
    them import the cosyvoice package.
    """
    env = os.environ.copy()
    entries = cosyvoice_pythonpath()
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env
