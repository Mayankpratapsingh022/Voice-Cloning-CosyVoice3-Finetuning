"""Runs CosyVoice3's own `cosyvoice/bin/train.py` and `cosyvoice/bin/average_model.py`
per experiment, then assembles an inference-ready checkpoint directory. Runs on the
RunPod pod, not locally.

Two upstream quirks this module works around, found by reading the actual recipe
(`examples/libritts/cosyvoice3/run.sh`) rather than assuming:

1. Training is per-component: `--model {llm,flow,hifigan}` trains exactly one of the
   three sub-models per invocation, each from its own pretrained checkpoint. A "full
   fine-tune" experiment runs all three; a "CFM-only" experiment runs just `flow` and
   the other two are carried through from the pretrained checkpoint unchanged.
2. `average_model.py` writes its output as `{component}.pt` -- but the inference loader
   (`cosyvoice/cli/cosyvoice.py`) reads the vocoder weights from `hift.pt`, not
   `hifigan.pt`. `run.sh`'s own stage 6 doesn't rename this (and separately points at
   `exp/cosyvoice/...` while stage 5 trained into `exp/cosyvoice3/...`, an apparent
   path typo in the upstream script). `assemble_inference_checkpoint` below renames
   explicitly rather than silently inheriting that mismatch.

Invoked remotely via `voiceclone remote run-experiment` (see cli.py) over SSH from
the local driver in training/orchestrate.py. Experiments in a sweep run sequentially,
not concurrently -- there is one GPU on the pod, so parallel experiments would just
contend with each other rather than actually run faster.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from voiceclone.config import RemotePaths
from voiceclone.logging_utils import get_logger
from voiceclone.training.experiment import ExperimentConfig, render_experiment_config

logger = get_logger(__name__)

# Component -> pretrained checkpoint / final export filename. Confirmed against
# cosyvoice/cli/cosyvoice.py's CosyVoice.__init__ (loads llm.pt/flow.pt/hift.pt) and
# examples/libritts/cosyvoice3/run.sh (trains/checkpoints as llm/flow/hifigan). Verify
# these still match once you've actually downloaded a pretrained_models/ snapshot --
# upstream naming has shifted between CosyVoice/CosyVoice2/CosyVoice3 before.
COMPONENT_CHECKPOINT_NAME = {"llm": "llm.pt", "flow": "flow.pt", "hifigan": "hifigan.pt"}
COMPONENT_EXPORT_NAME = {"llm": "llm.pt", "flow": "flow.pt", "hifigan": "hift.pt"}

# Static, architecture-only assets that don't change during fine-tuning, copied
# through from the pretrained snapshot into every experiment's exported checkpoint dir.
# Maps the pretrained snapshot's filename -> what the inference loader
# (cosyvoice/cli/cosyvoice.py's CosyVoice.__init__) actually looks for in model_dir.
# Confirmed live against a real downloaded FunAudioLLM/Fun-CosyVoice3-0.5B-2512
# snapshot: it ships its config as `cosyvoice3.yaml`, but CosyVoice.__init__ hardcodes
# `{model_dir}/cosyvoice.yaml` -- same class of naming mismatch as hifigan.pt/hift.pt
# above, just for the config file instead of a weights file. spk2info.pt was not
# present in that snapshot at all; handled gracefully below (warn + skip) either way.
STATIC_ASSETS = {
    "cosyvoice3.yaml": "cosyvoice.yaml",
    "campplus.onnx": "campplus.onnx",
    "speech_tokenizer_v3.onnx": "speech_tokenizer_v3.onnx",
    "spk2info.pt": "spk2info.pt",
    "CosyVoice-BlankEN": "CosyVoice-BlankEN",
}

TRAIN_ENGINE = "torch_ddp"
DIST_BACKEND = "nccl"
AVERAGE_NUM = 5


def _experiment_root(speaker_prefix: str, experiment_name: str) -> Path:
    return RemotePaths.CHECKPOINT_DIR / speaker_prefix / experiment_name


def train_component(speaker_prefix: str, experiment_name: str, component: str, config_yaml_relpath: str) -> None:
    """Fine-tune one component (llm/flow/hifigan) via CosyVoice3's own train.py.

    Single-GPU: `torchrun --nproc_per_node=1` rather than shelling out to run.sh's
    4-GPU default, matching the plan's "one rented GPU" compute model.
    """
    exp_root = _experiment_root(speaker_prefix, experiment_name)
    model_dir = exp_root / component / TRAIN_ENGINE
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = exp_root / "tensorboard" / component

    dataset_root = RemotePaths.DATASET_DIR / speaker_prefix
    train_data_list = dataset_root / "train" / "parquet" / "data.list"
    cv_data_list = dataset_root / "cv" / "parquet" / "data.list"
    pretrained_checkpoint = RemotePaths.PRETRAINED_DIR / COMPONENT_CHECKPOINT_NAME[component]
    config_path = RemotePaths.CHECKPOINT_DIR / config_yaml_relpath

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--rdzv_id=voiceclone",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:1234",
        str(RemotePaths.REPO_ROOT / "cosyvoice" / "bin" / "train.py"),
        "--train_engine", TRAIN_ENGINE,
        "--config", str(config_path),
        "--train_data", str(train_data_list),
        "--cv_data", str(cv_data_list),
        "--qwen_pretrain_path", str(RemotePaths.PRETRAINED_DIR / "CosyVoice-BlankEN"),
        "--onnx_path", str(RemotePaths.PRETRAINED_DIR),
        "--model", component,
        "--checkpoint", str(pretrained_checkpoint),
        "--model_dir", str(model_dir),
        "--tensorboard_dir", str(tensorboard_dir),
        "--ddp.dist_backend", DIST_BACKEND,
        "--num_workers", "4",
        "--prefetch", "100",
        "--pin_memory",
        "--use_amp",
    ]
    logger.info("training %s/%s: %s", experiment_name, component, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT))


def average_and_export_component(speaker_prefix: str, experiment_name: str, component: str) -> None:
    """average_model.py --val_best over the trained component's checkpoints, written
    into the experiment's `exported/` dir under the inference-expected filename.
    """
    exp_root = _experiment_root(speaker_prefix, experiment_name)
    model_dir = exp_root / component / TRAIN_ENGINE
    exported_dir = exp_root / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)
    dst = exported_dir / COMPONENT_EXPORT_NAME[component]

    cmd = [
        "python", str(RemotePaths.REPO_ROOT / "cosyvoice" / "bin" / "average_model.py"),
        "--dst_model", str(dst),
        "--src_path", str(model_dir),
        "--num", str(AVERAGE_NUM),
        "--val_best",
    ]
    logger.info("averaging %s/%s: %s", experiment_name, component, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RemotePaths.REPO_ROOT))


def assemble_inference_checkpoint(
    speaker_prefix: str, experiment_name: str, trained_components: tuple[str, ...]
) -> str:
    """Build a self-contained, `AutoModel`-loadable directory for this experiment:
    static assets copied from the pretrained snapshot, weights from `exported/` for
    every trained component, and pretrained weights verbatim for any component this
    experiment left untouched (e.g. `llm`/`hifigan` in a CFM-only run).

    Returns the path to the assembled directory.
    """
    exp_root = _experiment_root(speaker_prefix, experiment_name)
    exported_dir = exp_root / "exported"
    final_dir = exp_root / "inference_ready"
    final_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in STATIC_ASSETS.items():
        src = RemotePaths.PRETRAINED_DIR / src_name
        dst = final_dir / dst_name
        if not src.exists():
            logger.warning("expected static asset %s not found in pretrained snapshot, skipping", src)
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    for component, export_name in COMPONENT_EXPORT_NAME.items():
        if component in trained_components:
            src = exported_dir / export_name
        else:
            src = RemotePaths.PRETRAINED_DIR / COMPONENT_CHECKPOINT_NAME[component]
        shutil.copy2(src, final_dir / export_name)

    logger.info("assembled inference-ready checkpoint for %s -> %s", experiment_name, final_dir)
    return str(final_dir)


def run_experiment(speaker_prefix: str, experiment: ExperimentConfig) -> str:
    """End-to-end for one experiment: render its config, train each specified
    component, average, and assemble the final inference-ready checkpoint.
    """
    base_yaml = RemotePaths.REPO_ROOT / "examples" / "libritts" / "cosyvoice3" / "conf" / "cosyvoice3.yaml"
    rendered_relpath = f"{speaker_prefix}/{experiment.name}/cosyvoice3.patched.yaml"
    render_experiment_config(base_yaml, experiment, RemotePaths.CHECKPOINT_DIR / rendered_relpath)

    for component in experiment.components:
        train_component(speaker_prefix, experiment.name, component, rendered_relpath)
        average_and_export_component(speaker_prefix, experiment.name, component)

    return assemble_inference_checkpoint(speaker_prefix, experiment.name, experiment.components)


def run_experiment_sweep(speaker_prefix: str, experiments: list[ExperimentConfig]) -> dict[str, str]:
    """Run every experiment in the sweep in sequence on this pod's one GPU -- this is
    the "2-4 configs" comparison the project plan calls for, not a single
    train-once-and-ship run.
    """
    return {exp.name: run_experiment(speaker_prefix, exp) for exp in experiments}
