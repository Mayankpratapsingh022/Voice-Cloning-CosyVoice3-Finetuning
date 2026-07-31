"""Single CLI tying the whole pipeline together: `voiceclone <group> <command>`.

Every command that touches Modal opens its own ephemeral `with app.run():` context —
the same pattern `modal run` uses under the hood — so this file works as a plain
script entrypoint (`voiceclone ...` or `python -m voiceclone.cli ...`) without
requiring the Modal CLI directly, though `modal run`/`modal deploy` remain available
for anyone who prefers driving it that way.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from voiceclone.config import get_settings
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

# pydantic-settings reads .env into our own Settings object, but doesn't export it to
# os.environ — this does, so OPENAI_API_KEY (and anything else third-party SDKs read
# straight from the environment) works when set in .env instead of the shell.
load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Fine-tune and serve a personal CosyVoice3 voice clone.")
data_cli = typer.Typer(no_args_is_help=True, help="Local + Modal dataset preparation.")
pretrained_cli = typer.Typer(no_args_is_help=True, help="Base checkpoint management.")
train_cli = typer.Typer(no_args_is_help=True, help="Run fine-tuning experiments.")
eval_cli = typer.Typer(no_args_is_help=True, help="Score fine-tuned checkpoints.")
serve_cli = typer.Typer(no_args_is_help=True, help="Call a deployed inference endpoint.")

app.add_typer(data_cli, name="data")
app.add_typer(pretrained_cli, name="pretrained")
app.add_typer(train_cli, name="train")
app.add_typer(eval_cli, name="eval")
app.add_typer(serve_cli, name="serve")


@data_cli.command("prepare")
def data_prepare(
    raw_sessions_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Dir of long-form session WAVs"),
    dest: Path = typer.Option(..., help="Local output root for chunks + train/cv/holdout manifests"),
    transcribe_backend: str = typer.Option(
        "local", help="'local' (free, CPU-bound faster-whisper) or 'openai' (paid, fast, needs OPENAI_API_KEY)"
    ),
    transcribe_model: str = typer.Option(
        None, help="Defaults to large-v3 (local) or whisper-1 (openai)"
    ),
    whisper_device: str = typer.Option("cpu", help="local backend only: 'cpu' or 'cuda'"),
) -> None:
    """Chunk, transcribe, and split raw recordings into manifests. Runs locally, no GPU
    required (the 'openai' transcribe backend uses OpenAI's cloud API instead of local compute)."""
    from voiceclone.data.pipeline import run_local_data_prep

    settings = get_settings()
    dirs = run_local_data_prep(
        raw_sessions_dir,
        dest,
        settings,
        transcribe_backend=transcribe_backend,
        transcribe_model=transcribe_model,
        whisper_device=whisper_device,
    )
    for split, path in dirs.items():
        typer.echo(f"{split}: {path}")


@data_cli.command("upload")
def data_upload(dest: Path = typer.Option(..., exists=True, help="The --dest root used in `data prepare`")) -> None:
    """Push prepared manifests + audio to the Modal dataset Volume."""
    from voiceclone.data.modal_stages import upload_dataset

    prefix = upload_dataset(dest)
    typer.echo(f"uploaded under remote prefix: {prefix}")


@data_cli.command("extract")
def data_extract() -> None:
    """Run CosyVoice3's embedding/speech-token/parquet extraction on Modal (train + cv splits)."""
    from voiceclone.data.modal_stages import run_full_data_prep
    from voiceclone.modal_app import app as modal_app

    settings = get_settings()
    with modal_app.run():
        run_full_data_prep.remote(settings.speaker_id)
    typer.echo("data extraction complete")


@pretrained_cli.command("download")
def pretrained_download(model_id: str = typer.Option(None, help="Defaults to VOICECLONE_BASE_MODEL_ID")) -> None:
    """Fetch the base CosyVoice3 checkpoint into the shared pretrained Volume (one-time)."""
    from voiceclone.data.modal_stages import download_pretrained
    from voiceclone.modal_app import app as modal_app

    settings = get_settings()
    with modal_app.run():
        download_pretrained.remote(model_id or settings.base_model_id)
    typer.echo("pretrained checkpoint ready")


DEFAULT_EXPERIMENTS_CONFIG = Path("configs/experiments.yaml")


def _load_experiments(config: Path):
    from voiceclone.training.experiment import DEFAULT_EXPERIMENTS, load_experiments

    if config.exists():
        return load_experiments(config)
    logger.warning("%s not found, falling back to training.experiment.DEFAULT_EXPERIMENTS", config)
    return DEFAULT_EXPERIMENTS


@train_cli.command("list")
def train_list(config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read")) -> None:
    """Show the configured experiments (see configs/experiments.yaml)."""
    for exp in _load_experiments(config):
        typer.echo(
            f"{exp.name}: components={exp.components} lr={exp.learning_rate} "
            f"max_epoch={exp.max_epoch} — {exp.notes}"
        )


@train_cli.command("run")
def train_run(
    experiment: str = typer.Argument(..., help="Experiment name from `train list`"),
    config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read"),
) -> None:
    """Run a single fine-tuning experiment end to end (train -> average -> assemble)."""
    from voiceclone.modal_app import app as modal_app
    from voiceclone.training.train import run_experiment

    settings = get_settings()
    experiments = _load_experiments(config)
    matches = [e for e in experiments if e.name == experiment]
    if not matches:
        available = ", ".join(e.name for e in experiments)
        raise typer.BadParameter(f"unknown experiment '{experiment}', available: {available}")

    with modal_app.run():
        result_path = run_experiment.remote(settings.speaker_id, matches[0])
    typer.echo(f"checkpoint ready: {result_path}")


@train_cli.command("sweep")
def train_sweep(config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read")) -> None:
    """Run every configured experiment concurrently — the comparison the plan calls for."""
    from voiceclone.modal_app import app as modal_app
    from voiceclone.training.train import run_experiment_sweep

    settings = get_settings()
    experiments = _load_experiments(config)
    with modal_app.run():
        results = run_experiment_sweep.remote(settings.speaker_id, experiments)
    for name, path in results.items():
        typer.echo(f"{name}: {path}")


@eval_cli.command("run")
def eval_run(
    experiments: list[str] = typer.Argument(
        ..., help="Experiment names to generate + score, e.g. full_ft_default cfm_only"
    ),
) -> None:
    """Generate holdout samples for each experiment and write the comparison report."""
    from voiceclone.evaluation.run_eval import evaluate_sweep
    from voiceclone.modal_app import app as modal_app

    settings = get_settings()
    with modal_app.run():
        report_path = evaluate_sweep.remote(settings.speaker_id, experiments)
    typer.echo(f"report: {report_path}")


@serve_cli.command("synthesize")
def serve_synthesize(
    text: str = typer.Argument(...),
    out: Path = typer.Option(Path("output.wav")),
    experiment: str = typer.Option("full_ft_default"),
) -> None:
    """Call the (already-deployed) inference service and save the result locally.

    Requires `modal deploy src/voiceclone/inference/serve.py` to have been run first —
    see README for the deploy step and the resulting shareable web demo URL.
    """
    from voiceclone.inference.serve import InferenceService

    settings = get_settings()
    service = InferenceService(experiment_name=experiment, speaker_prefix=settings.speaker_id)
    wav_bytes = service.synthesize.remote(text)
    out.write_bytes(wav_bytes)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
