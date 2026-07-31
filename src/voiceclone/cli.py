"""Single CLI tying the whole pipeline together: `voiceclone <group> <command>`.

Two kinds of commands live here:

- Local commands (data, pod, pretrained, train, eval, serve): run on your machine,
  orchestrate a RunPod pod (create/reuse it, sync this project onto it, SSH in) to do
  the actual GPU work. This is what you run.
- Remote commands (under `remote`): run on the pod itself, invoked by the local
  commands over SSH via orchestrate.py's `remote_voiceclone`. You would not normally
  run these by hand, though nothing stops you from SSHing in and doing so.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from voiceclone.config import get_settings
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

# pydantic-settings reads .env into our own Settings object, but doesn't export it to
# os.environ. This does, so RUNPOD_API_KEY / HF_TOKEN / OPENAI_API_KEY (everything
# third-party SDKs and runpodctl read straight from the environment) work when set in
# .env instead of the shell.
load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Fine-tune and serve a personal CosyVoice3 voice clone.")
data_cli = typer.Typer(no_args_is_help=True, help="Local dataset preparation and HF dataset transfer.")
pod_cli = typer.Typer(no_args_is_help=True, help="RunPod pod lifecycle.")
pretrained_cli = typer.Typer(no_args_is_help=True, help="Base checkpoint management.")
train_cli = typer.Typer(no_args_is_help=True, help="Run fine-tuning experiments.")
eval_cli = typer.Typer(no_args_is_help=True, help="Score fine-tuned checkpoints.")
serve_cli = typer.Typer(no_args_is_help=True, help="Serve a fine-tuned checkpoint.")
remote_cli = typer.Typer(no_args_is_help=True, help="Runs on the pod. Not normally invoked directly.")

app.add_typer(data_cli, name="data")
app.add_typer(pod_cli, name="pod")
app.add_typer(pretrained_cli, name="pretrained")
app.add_typer(train_cli, name="train")
app.add_typer(eval_cli, name="eval")
app.add_typer(serve_cli, name="serve")
app.add_typer(remote_cli, name="remote")

DEFAULT_EXPERIMENTS_CONFIG = Path("configs/experiments.yaml")


def _load_experiments(config: Path):
    from voiceclone.training.experiment import DEFAULT_EXPERIMENTS, load_experiments

    if config.exists():
        return load_experiments(config)
    logger.warning("%s not found, falling back to training.experiment.DEFAULT_EXPERIMENTS", config)
    return DEFAULT_EXPERIMENTS


# --- local: data ---------------------------------------------------------------------


@data_cli.command("prepare")
def data_prepare(
    raw_sessions_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, help="Dir of long-form session recordings"
    ),
    dest: Path = typer.Option(..., help="Local output root for chunks + train/cv/holdout manifests"),
    transcribe_backend: str = typer.Option(
        "local", help="'local' (free, CPU-bound faster-whisper) or 'openai' (paid, fast, needs OPENAI_API_KEY)"
    ),
    transcribe_model: str = typer.Option(None, help="Defaults to large-v3 (local) or whisper-1 (openai)"),
    whisper_device: str = typer.Option("cpu", help="local backend only: 'cpu' or 'cuda'"),
) -> None:
    """Chunk, transcribe, and split raw recordings into manifests. Runs locally, no GPU
    required (the 'openai' transcribe backend uses OpenAI's cloud API instead of local compute)."""
    from voiceclone.data.pipeline import run_local_data_prep

    settings = get_settings()
    dirs = run_local_data_prep(
        raw_sessions_dir, dest, settings,
        transcribe_backend=transcribe_backend, transcribe_model=transcribe_model, whisper_device=whisper_device,
    )
    for split, path in dirs.items():
        typer.echo(f"{split}: {path}")


@data_cli.command("upload")
def data_upload(
    dest: Path = typer.Option(..., exists=True, help="The --dest root used in `data prepare`"),
    repo_id: str = typer.Option(None, help="Defaults to '{hf_username}/voiceclone-{speaker_id}-dataset'"),
) -> None:
    """Push the prepared dataset to a private HuggingFace dataset repo."""
    from voiceclone.data.hf_dataset import upload_dataset

    settings = get_settings()
    used_repo_id = upload_dataset(dest, settings.speaker_id, repo_id=repo_id)
    typer.echo(f"uploaded to: https://huggingface.co/datasets/{used_repo_id}")


@data_cli.command("extract")
def data_extract(repo_id: str = typer.Option(None, help="Defaults to the same default as `data upload`")) -> None:
    """On the pod: download the dataset from HF, then run CosyVoice3's embedding /
    speech-token / parquet extraction (train + cv splits)."""
    from voiceclone import orchestrate
    from voiceclone.data.hf_dataset import default_repo_id

    settings = get_settings()
    pod = orchestrate.ensure_ready(settings)
    used_repo_id = repo_id or default_repo_id(settings.speaker_id)
    orchestrate.remote_voiceclone(
        pod, ["download-dataset", "--repo-id", used_repo_id, "--speaker", settings.speaker_id]
    )
    orchestrate.remote_voiceclone(pod, ["extract-data", "--speaker", settings.speaker_id])
    typer.echo("data extraction complete")


# --- local: pod ------------------------------------------------------------------------


@pod_cli.command("status")
def pod_status() -> None:
    """Show the current pod's status, if one exists for this project."""
    from voiceclone import orchestrate

    settings = get_settings()
    pod_id = orchestrate.find_existing_pod(settings.runpod_pod_name)
    if pod_id is None:
        typer.echo("no pod found")
        return
    from voiceclone import runpod_app

    info = runpod_app.get_pod(pod_id)
    typer.echo(f"{info.pod_id}: {info.status} ({info.ip}:{info.ssh_port})")


@pod_cli.command("gpus")
def pod_gpus() -> None:
    """List RunPod's currently available GPU types/pricing (`runpodctl get cloud`)."""
    from voiceclone import runpod_app

    runpod_app.ensure_authenticated()
    typer.echo(runpod_app.list_gpu_types())


@pod_cli.command("stop")
def pod_stop() -> None:
    """Stop the pod (keeps the Network Volume, stops GPU billing)."""
    from voiceclone import orchestrate, runpod_app

    settings = get_settings()
    pod_id = orchestrate.find_existing_pod(settings.runpod_pod_name)
    if pod_id is None:
        typer.echo("no pod found")
        return
    runpod_app.stop_pod(pod_id)
    typer.echo(f"stopped {pod_id}")


@pod_cli.command("remove")
def pod_remove() -> None:
    """Terminate the pod entirely (Network Volume and its contents are unaffected)."""
    from voiceclone import orchestrate, runpod_app

    settings = get_settings()
    pod_id = orchestrate.find_existing_pod(settings.runpod_pod_name)
    if pod_id is None:
        typer.echo("no pod found")
        return
    runpod_app.remove_pod(pod_id)
    typer.echo(f"removed {pod_id}")


# --- local: pretrained / train / eval / serve -----------------------------------------


@pretrained_cli.command("download")
def pretrained_download(model_id: str = typer.Option(None, help="Defaults to VOICECLONE_BASE_MODEL_ID")) -> None:
    """On the pod: fetch the base CosyVoice3 checkpoint (one-time)."""
    from voiceclone import orchestrate

    settings = get_settings()
    pod = orchestrate.ensure_ready(settings)
    orchestrate.remote_voiceclone(pod, ["download-pretrained", "--model-id", model_id or settings.base_model_id])
    typer.echo("pretrained checkpoint ready")


@train_cli.command("list")
def train_list(config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read")) -> None:
    """Show the configured experiments (see configs/experiments.yaml). Local only."""
    for exp in _load_experiments(config):
        typer.echo(
            f"{exp.name}: components={exp.components} lr={exp.learning_rate} "
            f"max_epoch={exp.max_epoch}. {exp.notes}"
        )


@train_cli.command("run")
def train_run(
    experiment: str = typer.Argument(..., help="Experiment name from `train list`"),
    config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read"),
) -> None:
    """On the pod: run a single fine-tuning experiment end to end (train -> average -> assemble)."""
    from voiceclone import orchestrate

    settings = get_settings()
    experiments = _load_experiments(config)
    if experiment not in {e.name for e in experiments}:
        available = ", ".join(e.name for e in experiments)
        raise typer.BadParameter(f"unknown experiment '{experiment}', available: {available}")

    pod = orchestrate.ensure_ready(settings)
    orchestrate.remote_voiceclone(pod, ["run-experiment", "--speaker", settings.speaker_id, "--experiment", experiment])
    typer.echo(f"experiment '{experiment}' complete")


@train_cli.command("sweep")
def train_sweep(config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG, help="experiments.yaml to read")) -> None:
    """On the pod: run every configured experiment in sequence (one GPU, so not
    concurrent). This is the multi-config comparison the project plan calls for."""
    from voiceclone import orchestrate

    settings = get_settings()
    experiments = _load_experiments(config)
    pod = orchestrate.ensure_ready(settings)
    names = [e.name for e in experiments]
    orchestrate.remote_voiceclone(pod, ["run-sweep", "--speaker", settings.speaker_id, *names])
    typer.echo(f"sweep complete: {', '.join(names)}")


@eval_cli.command("run")
def eval_run(
    experiments: list[str] = typer.Argument(
        ..., help="Experiment names to generate + score, e.g. full_ft_default cfm_only"
    ),
) -> None:
    """On the pod: generate holdout samples for each experiment, score them, and pull
    the comparison report back to this machine."""
    from voiceclone import orchestrate

    settings = get_settings()
    pod = orchestrate.ensure_ready(settings)
    orchestrate.remote_voiceclone(pod, ["evaluate", "--speaker", settings.speaker_id, *experiments])

    from voiceclone import runpod_app

    local_report_dir = Path("eval_reports") / settings.speaker_id
    runpod_app.sync_from_pod(pod, f"/workspace/checkpoints/{settings.speaker_id}/eval_report", local_report_dir)
    typer.echo(f"report: {local_report_dir / 'eval_summary.md'}")


@serve_cli.command("start")
def serve_start(experiment: str = typer.Option("full_ft_default")) -> None:
    """On the pod: start the inference web demo in the background and print its
    public URL (RunPod's proxy on the pod's exposed HTTP port)."""
    from voiceclone import orchestrate

    settings = get_settings()
    pod = orchestrate.ensure_ready(settings)
    orchestrate.remote_voiceclone(
        pod,
        ["serve", "--speaker", settings.speaker_id, "--experiment", experiment, "--background"],
        stream=False,
    )
    typer.echo(f"demo starting at: https://{pod.pod_id}-8000.proxy.runpod.net")
    typer.echo("(RunPod's proxy URL pattern, confirm against your pod's actual public URL in the RunPod dashboard)")


# --- remote: invoked over SSH by the local commands above, runs on the pod -----------


@remote_cli.command("download-pretrained")
def remote_download_pretrained(model_id: str = typer.Option(...)) -> None:
    from voiceclone.data.remote_stages import download_pretrained

    download_pretrained(model_id)


@remote_cli.command("download-dataset")
def remote_download_dataset(repo_id: str = typer.Option(...), speaker: str = typer.Option(...)) -> None:
    from voiceclone.config import RemotePaths
    from voiceclone.data.hf_dataset import download_dataset

    download_dataset(repo_id, RemotePaths.DATASET_DIR / speaker)


@remote_cli.command("extract-data")
def remote_extract_data(speaker: str = typer.Option(...)) -> None:
    from voiceclone.data.remote_stages import run_full_data_prep

    run_full_data_prep(speaker)


@remote_cli.command("run-experiment")
def remote_run_experiment(
    speaker: str = typer.Option(...),
    experiment: str = typer.Option(...),
    config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG),
) -> None:
    from voiceclone.training.train import run_experiment as run_experiment_fn

    experiments = _load_experiments(config)
    matches = [e for e in experiments if e.name == experiment]
    if not matches:
        raise typer.BadParameter(f"unknown experiment '{experiment}'")
    run_experiment_fn(speaker, matches[0])


@remote_cli.command("run-sweep")
def remote_run_sweep(
    speaker: str = typer.Option(...),
    experiments: list[str] = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_EXPERIMENTS_CONFIG),
) -> None:
    from voiceclone.training.train import run_experiment_sweep

    all_experiments = {e.name: e for e in _load_experiments(config)}
    selected = [all_experiments[name] for name in experiments if name in all_experiments]
    run_experiment_sweep(speaker, selected)


@remote_cli.command("evaluate")
def remote_evaluate(speaker: str = typer.Option(...), experiments: list[str] = typer.Argument(...)) -> None:
    from voiceclone.evaluation.run_eval import evaluate_sweep

    evaluate_sweep(speaker, experiments)


@remote_cli.command("serve")
def remote_serve(
    speaker: str = typer.Option(...),
    experiment: str = typer.Option(...),
    background: bool = typer.Option(False),
) -> None:
    from voiceclone.inference.serve import run_server

    if background:
        import subprocess
        import sys

        subprocess.Popen(  # noqa: S603 -- fixed argv built from typed CLI options, not untrusted input
            [sys.executable, "-m", "voiceclone.cli", "remote", "serve",
             "--speaker", speaker, "--experiment", experiment],
            start_new_session=True,
        )
        return
    run_server(speaker, experiment)


if __name__ == "__main__":
    app()
