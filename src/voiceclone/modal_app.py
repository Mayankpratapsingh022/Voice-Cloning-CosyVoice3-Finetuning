"""Single source of truth for the Modal App, container image, and Volumes.

Every other module (data pipeline, training, evaluation, inference) imports `app`,
`cosyvoice_image`, and the volume handles from here rather than redefining them, so
there is exactly one container image to build/cache and volumes never accidentally
diverge by name between modules.

Pin `COSYVOICE_COMMIT` to a specific commit once you've validated the pipeline against
it — tracking a moving branch is fine for initial exploration but is the single easiest
way to have a fine-tuning run silently break under you between sessions.
"""

from __future__ import annotations

import modal

from voiceclone.config import get_settings

settings = get_settings()

app = modal.App(settings.modal_app_name)

COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_COMMIT = "main"  # TODO: pin to a specific commit hash before your real training runs

GPU_A100 = "A100-80GB"
GPU_H100 = "H100"
GPU_CHEAP_INFERENCE = "L4"

cosyvoice_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "sox", "libsox-dev", "ffmpeg", "build-essential")
    .run_commands(
        f"git clone --depth 1 {COSYVOICE_REPO_URL} /root/CosyVoice",
        "cd /root/CosyVoice && git submodule update --init --recursive",
    )
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "cd /root/CosyVoice && pip install --no-deps -r requirements.txt || "
        "pip install -r requirements.txt"
    )
    .pip_install("onnxruntime-gpu==1.18.0")
    .env(
        {
            "PYTHONPATH": "/root/CosyVoice:/root/CosyVoice/third_party/Matcha-TTS",
            "PYTHONIOENCODING": "UTF-8",
        }
    )
    # Makes our own `voiceclone` package importable inside the container — every
    # remote function below does `from voiceclone... import ...`, so without this the
    # container has no idea what `voiceclone` is even though the driver process does.
    .add_local_python_source("voiceclone")
)

# Evaluation needs an ASR model (WER) and speaker/naturalness models that are unrelated
# to CosyVoice3 itself — kept in a separate lighter image so eval runs don't pay to
# rebuild/pull the full training image when only metric code changes.
eval_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "faster-whisper>=1.0",
        "jiwer>=3.0",
        "transformers>=4.51,<4.52",
        "pandas>=2.2",
        "soundfile>=0.12",
    )
    .run_commands(
        # Pre-download UTMOS (torch.hub) and WavLM-SV (transformers) weights at build
        # time so every eval run doesn't redownload them on a cold start.
        "python -c \"import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)\"",
        "python -c \"from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor; "
        "WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv'); "
        "Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')\"",
    )
    .add_local_python_source("voiceclone")
)

dataset_volume = modal.Volume.from_name(settings.dataset_volume, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(settings.checkpoint_volume, create_if_missing=True)
pretrained_volume = modal.Volume.from_name(settings.pretrained_volume, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(settings.hf_cache_volume, create_if_missing=True)

STANDARD_VOLUMES = {
    "/dataset": dataset_volume,
    "/checkpoints": checkpoint_volume,
    "/pretrained": pretrained_volume,
    "/hf-cache": hf_cache_volume,
}
