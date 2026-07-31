"""Upload the locally-prepared dataset to Modal, then run CosyVoice3's own
embedding/token/parquet extraction scripts against it inside the training image.

These three extraction stages are deliberately *not* reimplemented — they're invoked
as subprocesses against the actual upstream scripts (tools/extract_embedding.py,
tools/extract_speech_token.py, tools/make_parquet_list.py) so behavior always matches
what CosyVoice3's own training recipe produces. Re-verify these paths/args against
`examples/libritts/cosyvoice3/run.sh` in the pinned commit if upstream changes them.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from voiceclone.config import Paths, Settings, get_settings
from voiceclone.data.manifest import remap_wav_scp
from voiceclone.logging_utils import get_logger
from voiceclone.modal_app import (
    GPU_CHEAP_INFERENCE,
    STANDARD_VOLUMES,
    app,
    cosyvoice_image,
    dataset_volume,
    pretrained_volume,
)

logger = get_logger(__name__)

TRAIN_SPLITS = ("train", "cv")  # holdout is eval-only, never goes through CosyVoice3 feature extraction


def upload_dataset(local_dest_root: Path, settings: Settings | None = None) -> str:
    """Stage remapped manifests + chunk audio and push them to the dataset Volume.

    Runs locally (no GPU needed). Returns the remote prefix (e.g. `mayank`) everything
    was uploaded under, which the extraction/training functions below take as input.
    """
    settings = settings or get_settings()
    remote_prefix = settings.speaker_id
    container_chunks_dir = f"{Paths.DATASET_DIR}/{remote_prefix}/chunks"

    with tempfile.TemporaryDirectory(prefix="voiceclone-upload-") as tmp:
        staging = Path(tmp)
        for split in ("train", "cv", "holdout"):
            split_dir = local_dest_root / split
            if not split_dir.exists():
                continue
            staged_split = staging / split
            shutil.copytree(split_dir, staged_split)
            remap_wav_scp(split_dir / "wav.scp", staged_split / "wav.scp", container_chunks_dir)

        with dataset_volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_dest_root / "chunks"), f"{remote_prefix}/chunks")
            for split in ("train", "cv", "holdout"):
                staged_split = staging / split
                if staged_split.exists():
                    batch.put_directory(str(staged_split), f"{remote_prefix}/{split}")

    logger.info("uploaded dataset for speaker=%s to volume=%s", remote_prefix, settings.dataset_volume)
    return remote_prefix


@app.function(image=cosyvoice_image, volumes={"/pretrained": pretrained_volume}, timeout=3600)
def download_pretrained(base_model_id: str) -> None:
    """One-time fetch of the base checkpoint we fine-tune from, e.g.
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, into the shared pretrained Volume.
    """
    from modelscope import snapshot_download

    dest = str(Paths.PRETRAINED_DIR)
    logger.info("downloading %s -> %s", base_model_id, dest)
    snapshot_download(base_model_id, local_dir=dest)
    pretrained_volume.commit()


@app.function(
    image=cosyvoice_image,
    volumes=STANDARD_VOLUMES,
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def extract_speaker_embedding(speaker_prefix: str, split: str) -> None:
    """campplus speaker-embedding extraction — CPU-only in upstream (onnxruntime
    CPUExecutionProvider is hardcoded in tools/extract_embedding.py), so this runs
    on a plain CPU container rather than paying for a GPU here.
    """
    manifest_dir = Paths.DATASET_DIR / speaker_prefix / split
    onnx_path = Paths.PRETRAINED_DIR / "campplus.onnx"
    cmd = [
        "python",
        str(Paths.REPO_ROOT / "tools" / "extract_embedding.py"),
        "--dir",
        str(manifest_dir),
        "--onnx_path",
        str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Paths.REPO_ROOT))
    dataset_volume.commit()


@app.function(
    image=cosyvoice_image,
    volumes=STANDARD_VOLUMES,
    gpu=GPU_CHEAP_INFERENCE,
    timeout=3600,
)
def extract_speech_tokens(speaker_prefix: str, split: str) -> None:
    """Discrete speech-token extraction via CosyVoice3's speech_tokenizer_v3.onnx.

    Upstream hardcodes CUDAExecutionProvider (tools/extract_speech_token.py), so this
    needs a GPU — a cheap one (L4) is plenty since it's pure ONNX inference, not
    training; don't burn A100/H100 budget on this stage.
    """
    manifest_dir = Paths.DATASET_DIR / speaker_prefix / split
    onnx_path = Paths.PRETRAINED_DIR / "speech_tokenizer_v3.onnx"
    cmd = [
        "python",
        str(Paths.REPO_ROOT / "tools" / "extract_speech_token.py"),
        "--dir",
        str(manifest_dir),
        "--onnx_path",
        str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Paths.REPO_ROOT))
    dataset_volume.commit()


@app.function(
    image=cosyvoice_image,
    volumes=STANDARD_VOLUMES,
    cpu=8.0,
    memory=16384,
    timeout=3600,
)
def package_parquet(speaker_prefix: str, split: str, num_utts_per_parquet: int = 1000) -> None:
    manifest_dir = Paths.DATASET_DIR / speaker_prefix / split
    parquet_dir = manifest_dir / "parquet"
    cmd = [
        "python",
        str(Paths.REPO_ROOT / "tools" / "make_parquet_list.py"),
        "--num_utts_per_parquet",
        str(num_utts_per_parquet),
        "--num_processes",
        "4",
        "--src_dir",
        str(manifest_dir),
        "--des_dir",
        str(parquet_dir),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Paths.REPO_ROOT))
    dataset_volume.commit()


@app.function(image=cosyvoice_image, volumes=STANDARD_VOLUMES, timeout=7200)
def run_full_data_prep(speaker_prefix: str) -> None:
    """Chain embedding -> speech-token -> parquet extraction for train + cv.

    Each stage commits the Volume before the next reads it, since Modal Volumes are
    not automatically synced across separate function invocations mid-flight.
    """
    for split in TRAIN_SPLITS:
        extract_speaker_embedding.remote(speaker_prefix, split)
        extract_speech_tokens.remote(speaker_prefix, split)
        package_parquet.remote(speaker_prefix, split)
    logger.info("data prep complete for splits=%s", TRAIN_SPLITS)
