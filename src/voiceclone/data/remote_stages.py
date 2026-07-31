"""CosyVoice3 feature extraction stages: embedding, speech-token, and parquet
packaging. Runs on the RunPod pod, not locally.

These three stages are deliberately *not* reimplemented, they shell out to the
actual upstream scripts (tools/extract_embedding.py, tools/extract_speech_token.py,
tools/make_parquet_list.py) so behavior always matches what CosyVoice3's own training
recipe produces. Re-verify these paths/args against
`examples/libritts/cosyvoice3/run.sh` in the pinned commit if upstream changes them.

Invoked remotely via `voiceclone remote extract-embedding / extract-speech-tokens /
package-parquet` (see cli.py's `remote` command group) over SSH from the local driver
in data/orchestrate.py.
"""

from __future__ import annotations

import subprocess

from voiceclone.config import RemotePaths
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

TRAIN_SPLITS = ("train", "cv")  # holdout is eval-only, never goes through CosyVoice3 feature extraction


def extract_speaker_embedding(speaker_prefix: str, split: str) -> None:
    """campplus speaker-embedding extraction. CPU-only in upstream (onnxruntime
    CPUExecutionProvider is hardcoded in tools/extract_embedding.py).
    """
    manifest_dir = RemotePaths.DATASET_DIR / speaker_prefix / split
    onnx_path = RemotePaths.PRETRAINED_DIR / "campplus.onnx"
    cmd = [
        "python", str(RemotePaths.REPO_ROOT / "tools" / "extract_embedding.py"),
        "--dir", str(manifest_dir),
        "--onnx_path", str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT))


def extract_speech_tokens(speaker_prefix: str, split: str) -> None:
    """Discrete speech-token extraction via CosyVoice3's speech_tokenizer_v3.onnx.

    Upstream hardcodes CUDAExecutionProvider (tools/extract_speech_token.py) so this
    needs the pod's GPU.
    """
    manifest_dir = RemotePaths.DATASET_DIR / speaker_prefix / split
    onnx_path = RemotePaths.PRETRAINED_DIR / "speech_tokenizer_v3.onnx"
    cmd = [
        "python", str(RemotePaths.REPO_ROOT / "tools" / "extract_speech_token.py"),
        "--dir", str(manifest_dir),
        "--onnx_path", str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT))


def package_parquet(speaker_prefix: str, split: str, num_utts_per_parquet: int = 1000) -> None:
    manifest_dir = RemotePaths.DATASET_DIR / speaker_prefix / split
    parquet_dir = manifest_dir / "parquet"
    # make_parquet_list.py writes into --des_dir but never creates it; upstream's
    # run.sh does `mkdir -p data/$x/parquet` on the line before invoking it. Worse,
    # the failure is quiet: the per-shard writes happen inside a multiprocessing pool
    # via apply_async, whose exceptions are swallowed unless .get() is called, so the
    # progress bar runs to 240/240 "successfully" and only the main process's final
    # data.list write surfaces the error.
    parquet_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", str(RemotePaths.REPO_ROOT / "tools" / "make_parquet_list.py"),
        "--num_utts_per_parquet", str(num_utts_per_parquet),
        "--num_processes", "4",
        "--src_dir", str(manifest_dir),
        "--des_dir", str(parquet_dir),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT))


def run_full_data_prep(speaker_prefix: str) -> None:
    """Chain embedding -> speech-token -> parquet extraction for train + cv."""
    for split in TRAIN_SPLITS:
        extract_speaker_embedding(speaker_prefix, split)
        extract_speech_tokens(speaker_prefix, split)
        package_parquet(speaker_prefix, split)
    logger.info("data prep complete for splits=%s", TRAIN_SPLITS)


def download_pretrained(base_model_id: str) -> None:
    """One-time fetch of the base checkpoint to fine-tune from, e.g.
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, into RemotePaths.PRETRAINED_DIR.
    """
    from modelscope import snapshot_download

    dest = str(RemotePaths.PRETRAINED_DIR)
    logger.info("downloading %s -> %s", base_model_id, dest)
    snapshot_download(base_model_id, local_dir=dest)
