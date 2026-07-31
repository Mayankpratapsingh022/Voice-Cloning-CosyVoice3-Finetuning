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
import sys
from pathlib import Path

from voiceclone.config import RemotePaths, cosyvoice_subprocess_env
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
        sys.executable, str(RemotePaths.REPO_ROOT / "tools" / "extract_embedding.py"),
        "--dir", str(manifest_dir),
        "--onnx_path", str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT), env=cosyvoice_subprocess_env())


def extract_speech_tokens(speaker_prefix: str, split: str) -> None:
    """Discrete speech-token extraction via CosyVoice3's speech_tokenizer_v3.onnx.

    NOT called by run_full_data_prep, deliberately. Kept because it is a faithful port
    of the upstream recipe stage and may be useful for inspection, but packing its
    output into the parquet actively breaks `flow` training. See
    run_full_data_prep's docstring for why.

    Upstream hardcodes CUDAExecutionProvider (tools/extract_speech_token.py) so this
    needs the pod's GPU.
    """
    manifest_dir = RemotePaths.DATASET_DIR / speaker_prefix / split
    onnx_path = RemotePaths.PRETRAINED_DIR / "speech_tokenizer_v3.onnx"
    cmd = [
        sys.executable, str(RemotePaths.REPO_ROOT / "tools" / "extract_speech_token.py"),
        "--dir", str(manifest_dir),
        "--onnx_path", str(onnx_path),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT), env=cosyvoice_subprocess_env())


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

    # make_parquet_list.py includes a `speech_token` column iff utt2speech_token.pt
    # exists in src_dir. A leftover file from an earlier run would silently reintroduce
    # the offline tokens (and the flow-training size mismatch) even though
    # run_full_data_prep no longer generates them, so drop it explicitly rather than
    # depending on the directory being clean.
    stale_tokens = manifest_dir / "utt2speech_token.pt"
    if stale_tokens.exists():
        logger.warning("removing stale %s so tokens are extracted online during training", stale_tokens)
        stale_tokens.unlink()

    cmd = [
        sys.executable, str(RemotePaths.REPO_ROOT / "tools" / "make_parquet_list.py"),
        "--num_utts_per_parquet", str(num_utts_per_parquet),
        "--num_processes", "4",
        "--src_dir", str(manifest_dir),
        "--des_dir", str(parquet_dir),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT), env=cosyvoice_subprocess_env())
    _verify_parquet_shards(parquet_dir)


def _verify_parquet_shards(parquet_dir: Path) -> None:
    """Fail loudly if data.list references shards that were never actually written.

    make_parquet_list.py exits 0 in this case: data.list is written by the main
    process and always succeeds, while the shards themselves are written by pool
    workers via apply_async, whose exceptions are discarded because .get() is never
    called. Hit for real when pyarrow was missing at packaging time -- every worker
    failed, data.list still listed all the shard paths, and the damage only surfaced
    much later as training silently loading zero batches and CosyVoice3's
    executor.cv() dying on `KeyError: 'tag'`.
    """
    data_list = parquet_dir / "data.list"
    if not data_list.exists():
        raise RuntimeError(f"{data_list} was not written; parquet packaging did not complete")

    referenced = [Path(line.strip()) for line in data_list.read_text().splitlines() if line.strip()]
    missing = [p for p in referenced if not p.exists()]
    if missing or not referenced:
        raise RuntimeError(
            f"parquet packaging reported success but {len(missing)}/{len(referenced)} shards are "
            f"missing under {parquet_dir}. This usually means the pool workers failed silently "
            f"(a missing pyarrow is the classic cause). First missing: {missing[:3]}"
        )
    logger.info("verified %d parquet shard(s) in %s", len(referenced), parquet_dir)


def run_full_data_prep(speaker_prefix: str) -> None:
    """Chain speaker-embedding extraction -> parquet packaging for train + cv.

    Speech-token extraction is deliberately NOT part of this chain, even though
    upstream's run.sh has it as a stage. Packing offline speech tokens into the
    parquet breaks `flow` training with a tensor size mismatch, because:

      - compute_fbank pads audio UP to a multiple of 960 samples before computing the
        mel (`num_frames: 960` in cosyvoice3.yaml), so mel length is always 2x the
        implied 25Hz token count.
      - extract_speech_token.py runs on the ORIGINAL unpadded audio, so for any
        utterance whose length is not already a multiple of 960 samples it yields one
        token fewer.
      - flow.forward() only extracts tokens online when `speech_token` is absent from
        the batch; if we supply it, it uses ours, and the flow decoder gets mu of
        length 2*(k-1) against a mel of length 2*k. Observed as:
        "Expected size 956 but got size 954".

    Upstream says as much in run.sh, which I ported past on the first pass:
    "NOTE embedding/token extraction is not necessary now as we support online feature
    extraction, but training speed will be influenced". Omitting the tokens takes the
    online path, which is aligned by construction. The cost is slower training steps
    (ONNX tokenization per batch); correctness first, and this dataset is small.

    Speaker embeddings are kept offline: they are one vector per utterance with no
    time alignment, so they carry no such risk and save real work per step.
    """
    for split in TRAIN_SPLITS:
        extract_speaker_embedding(speaker_prefix, split)
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
