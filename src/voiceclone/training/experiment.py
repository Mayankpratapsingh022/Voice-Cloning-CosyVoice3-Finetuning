"""Experiment definitions and the plain-scalar YAML patcher used to vary
learning-rate/epoch-count per experiment without touching CosyVoice3's HyperPyYAML
`!new:`/`!ref`-tagged model-definition sections.

CosyVoice3 trains three components independently (`llm`, `flow`, `hifigan` — see
`cosyvoice/bin/train.py`'s `--model` argument), each initialized from its own
pretrained checkpoint. "Full fine-tune" means running all three; the "CFM-only"
comparison from the project plan means training just `flow` (the Conditional Flow
Matching / DiT module that turns semantic tokens into acoustic features) and leaving
`llm` and `hifigan` at their pretrained weights.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

ALL_COMPONENTS = ("llm", "flow", "hifigan")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    components: tuple[str, ...]  # subset of ALL_COMPONENTS actually trained this run
    learning_rate: float | None = None  # overrides train_conf.optim_conf.lr (llm/flow)
    gan_learning_rate: float | None = None  # overrides train_conf_gan.optim_conf(_d).lr (hifigan)
    max_epoch: int | None = None  # applied to whichever block(s) the run touches
    notes: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.components) - set(ALL_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown component(s) {unknown}, must be subset of {ALL_COMPONENTS}")
        if not self.components:
            raise ValueError("experiment must train at least one component")


DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="pilot",
        components=ALL_COMPONENTS,
        max_epoch=5,
        notes=(
            "Pipeline validation only, NOT a quality candidate — run this first against ~20-30min "
            "of pilot recordings (PLAN.md Section 5) to confirm data prep -> train -> eval -> "
            "inference works end to end before recording the full dataset or running the real sweep."
        ),
    ),
    ExperimentConfig(
        name="full_ft_default",
        components=ALL_COMPONENTS,
        max_epoch=30,
        notes="Full fine-tune of llm+flow+hifigan at the recipe's default SFT lr (1e-5). Primary candidate.",
    ),
    ExperimentConfig(
        name="full_ft_higher_lr",
        components=ALL_COMPONENTS,
        learning_rate=3e-5,
        max_epoch=20,
        notes="Faster adaptation, higher overfitting risk — compare against full_ft_default via cv loss/eval.",
    ),
    ExperimentConfig(
        name="cfm_only",
        components=("flow",),
        max_epoch=30,
        notes="Timbre/acoustics-targeted fine-tune; llm and hifigan stay at pretrained weights.",
    ),
]


def load_experiments(path: Path) -> list[ExperimentConfig]:
    """Load experiments from a plain (non-HyperPyYAML) YAML file — see
    configs/experiments.yaml for the format. This, not `DEFAULT_EXPERIMENTS`, is what
    the CLI reads by default, so changing the sweep doesn't require editing source;
    `DEFAULT_EXPERIMENTS` remains as the fallback if the file is missing and as the
    content `save_default_experiments` writes out.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = []
    for entry in raw["experiments"]:
        entry = dict(entry)
        entry["components"] = tuple(entry["components"])
        experiments.append(ExperimentConfig(**entry))
    return experiments


def save_default_experiments(path: Path) -> None:
    """Write `DEFAULT_EXPERIMENTS` out as YAML — used to (re)generate
    configs/experiments.yaml, not called as part of the normal pipeline.
    """
    raw = {"experiments": [dict(asdict(e), components=list(e.components)) for e in DEFAULT_EXPERIMENTS]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _patch_block_scalar(lines: list[str], block_name: str, key: str, new_value: str) -> list[str]:
    """Replace `key: <value>` with `key: <new_value>`, scoped to the indented block
    that starts at a top-level `{block_name}:` line and ends at the next top-level
    (zero-indentation) key. Leaves every other line — including HyperPyYAML `!new:`
    tagged sections — untouched.
    """
    out = list(lines)
    in_block = False
    key_pattern = re.compile(rf"^(\s+){re.escape(key)}:\s*.*$")
    patched = False
    for i, line in enumerate(out):
        if line.startswith(f"{block_name}:"):
            in_block = True
            continue
        if in_block and line and not line[0].isspace():
            in_block = False  # left the block: hit the next top-level key
        if in_block:
            m = key_pattern.match(line)
            if m:
                out[i] = f"{m.group(1)}{key}: {new_value}"
                patched = True
    if not patched:
        logger.warning("could not find `%s.%s` to patch — check the base yaml hasn't changed shape", block_name, key)
    return out


def render_experiment_config(base_yaml_path: Path, experiment: ExperimentConfig, out_path: Path) -> Path:
    """Write a per-experiment copy of cosyvoice3.yaml with lr/max_epoch overridden.

    Only `train_conf` (llm/flow) and `train_conf_gan` (hifigan) are touched; the model
    architecture definitions are copied through verbatim.
    """
    lines = base_yaml_path.read_text(encoding="utf-8").splitlines()

    if experiment.learning_rate is not None:
        lines = _patch_block_scalar(lines, "train_conf", "lr", str(experiment.learning_rate))
    if experiment.gan_learning_rate is not None:
        # single call: patches both optim_conf.lr and optim_conf_d.lr, since
        # `_patch_block_scalar` matches every `lr:` line inside the block, not just the first
        lines = _patch_block_scalar(lines, "train_conf_gan", "lr", str(experiment.gan_learning_rate))
    if experiment.max_epoch is not None:
        if any(c in experiment.components for c in ("llm", "flow")):
            lines = _patch_block_scalar(lines, "train_conf", "max_epoch", str(experiment.max_epoch))
        if "hifigan" in experiment.components:
            lines = _patch_block_scalar(lines, "train_conf_gan", "max_epoch", str(experiment.max_epoch))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("rendered experiment config '%s' -> %s", experiment.name, out_path)
    return out_path
