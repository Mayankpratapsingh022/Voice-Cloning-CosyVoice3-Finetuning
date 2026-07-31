from pathlib import Path

import pytest

from voiceclone.training.experiment import (
    DEFAULT_EXPERIMENTS,
    ExperimentConfig,
    load_experiments,
    render_experiment_config,
    save_default_experiments,
)

# A trimmed-down stand-in for conf/cosyvoice3.yaml: real HyperPyYAML `!new:`/`!ref` tags
# in the model sections (must survive untouched) plus the plain-scalar train_conf /
# train_conf_gan blocks (the only thing render_experiment_config is allowed to touch).
FAKE_COSYVOICE3_YAML = """\
sample_rate: 24000
llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    llm_input_size: 896
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8

train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    max_epoch: 200
    grad_clip: 5

train_conf_gan:
    optim: adam
    optim_conf:
        lr: 0.0002
    optim_d: adam
    optim_conf_d:
        lr: 0.0002
    max_epoch: 200
"""


def test_experiment_config_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        ExperimentConfig(name="bad", components=("vocoder",))


def test_experiment_config_rejects_empty_components() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        ExperimentConfig(name="bad", components=())


def test_render_patches_only_targeted_scalars(tmp_path: Path) -> None:
    base_yaml = tmp_path / "cosyvoice3.yaml"
    base_yaml.write_text(FAKE_COSYVOICE3_YAML)

    exp = ExperimentConfig(
        name="test_exp",
        components=("llm", "flow", "hifigan"),
        learning_rate=3e-5,
        gan_learning_rate=0.0005,
        max_epoch=20,
    )
    out_path = tmp_path / "rendered.yaml"
    render_experiment_config(base_yaml, exp, out_path)
    rendered = out_path.read_text()

    rendered_lines = rendered.splitlines()

    # train_conf (llm/flow) patched
    assert "    max_epoch: 20" in rendered_lines
    assert any(line.strip() == "lr: 3e-05" for line in rendered_lines)
    # train_conf_gan patched — both optim_conf.lr and optim_conf_d.lr, and its own max_epoch
    assert sum(1 for line in rendered_lines if line.strip() == "lr: 0.0005") == 2
    assert rendered_lines.count("    max_epoch: 20") == 2  # one per block (train_conf, train_conf_gan)
    # untouched HyperPyYAML tags survive verbatim
    assert "!new:cosyvoice.llm.llm.CosyVoice3LM" in rendered
    assert "!name:cosyvoice.utils.common.ras_sampling" in rendered
    assert "top_p: 0.8" in rendered
    # original values fully gone, not just shadowed as a substring of the new ones
    assert "lr: 1e-5" not in rendered
    assert "lr: 0.0002" not in rendered
    assert "max_epoch: 200" not in rendered


def test_render_leaves_gan_untouched_when_component_not_trained(tmp_path: Path) -> None:
    base_yaml = tmp_path / "cosyvoice3.yaml"
    base_yaml.write_text(FAKE_COSYVOICE3_YAML)

    exp = ExperimentConfig(name="cfm_only", components=("flow",), max_epoch=15)
    out_path = tmp_path / "rendered.yaml"
    render_experiment_config(base_yaml, exp, out_path)
    rendered = out_path.read_text()

    assert "max_epoch: 15" in rendered  # train_conf patched (flow lives there)
    assert "max_epoch: 200" in rendered  # train_conf_gan untouched (hifigan not trained)


def test_load_and_save_experiments_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiments.yaml"
    save_default_experiments(path)
    loaded = load_experiments(path)
    assert [e.name for e in loaded] == [e.name for e in DEFAULT_EXPERIMENTS]
    assert [e.components for e in loaded] == [e.components for e in DEFAULT_EXPERIMENTS]
