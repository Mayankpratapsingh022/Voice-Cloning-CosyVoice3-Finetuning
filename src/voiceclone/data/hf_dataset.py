"""Push the locally-prepared dataset to a private HuggingFace dataset repo, and pull
it back down on the RunPod pod.

This is the data-transfer mechanism between your laptop and the pod: rather than
rsync-ing directly from your machine (which requires your laptop to be reachable for
the whole transfer, and re-transfers everything on every session), the prepared
dataset is pushed once to a private HF dataset repo and the pod pulls from there. It
also means the prepared dataset has a durable home independent of any single machine.

Requires `HF_TOKEN` in the environment (never pass it as a function argument or log
it) with write access for upload, read access for download. A private repo is the
default and is not changed by this module, nothing here makes the repo public.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from voiceclone.data.manifest import remap_wav_scp
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

DATASET_SPLITS = ("train", "cv", "holdout")


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Add it to .env (never pass it as a function argument or log it).")
    return token


def default_repo_id(speaker_id: str) -> str:
    from huggingface_hub import HfApi

    username = HfApi(token=_hf_token()).whoami()["name"]
    return f"{username}/voiceclone-{speaker_id}-dataset"


def upload_dataset(
    local_dest_root: Path,
    speaker_id: str,
    repo_id: str | None = None,
    remote_chunks_dir: str | None = None,
) -> str:
    """Push `local_dest_root` (the output of `data.pipeline.run_local_data_prep`) to a
    private HF dataset repo, with wav.scp paths remapped to wherever the pod will
    actually download the audio to (see `remote_chunks_dir`).

    Returns the repo id used (either `repo_id`, or the constructed default).
    """
    from huggingface_hub import HfApi

    token = _hf_token()
    repo_id = repo_id or default_repo_id(speaker_id)
    remote_chunks_dir = remote_chunks_dir or f"/workspace/dataset/{speaker_id}/chunks"

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="voiceclone-hf-upload-") as tmp:
        staging = Path(tmp)
        shutil.copytree(local_dest_root / "chunks", staging / "chunks")
        for split in DATASET_SPLITS:
            split_dir = local_dest_root / split
            if not split_dir.exists():
                continue
            staged_split = staging / split
            shutil.copytree(split_dir, staged_split)
            remap_wav_scp(split_dir / "wav.scp", staged_split / "wav.scp", remote_chunks_dir)

        logger.info("uploading dataset (%s) -> %s", local_dest_root, repo_id)
        api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="dataset")

    logger.info("uploaded dataset for speaker=%s -> https://huggingface.co/datasets/%s", speaker_id, repo_id)
    return repo_id


def download_dataset(repo_id: str, local_dir: Path) -> Path:
    """Run on the pod: pull the dataset down from the private HF repo into
    `local_dir` (typically RemotePaths.DATASET_DIR / speaker_id).
    """
    from huggingface_hub import snapshot_download

    token = _hf_token()
    logger.info("downloading dataset %s -> %s", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir), token=token)
    return local_dir
