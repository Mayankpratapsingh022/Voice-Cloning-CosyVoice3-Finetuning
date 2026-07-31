"""Local-side driver: turns "run this stage" into "make sure a pod exists, get this
project onto it, and SSH in to run it there."

Every `voiceclone <group> <command>` that needs the GPU goes through `ensure_ready`
once per session (idempotent: skips work already done) and then `remote_voiceclone`
per stage, which just runs `voiceclone remote <stage>` on the pod itself, since the
pod has this same package installed in its own venv (see scripts/pod_setup.sh).
"""

from __future__ import annotations

import shlex
from pathlib import Path

from voiceclone import runpod_app
from voiceclone.config import Settings
from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../voiceclone-project (repo root, two up from src/voiceclone/)
REMOTE_PROJECT_DIR = "/workspace/voiceclone-project"


def find_existing_pod(pod_name: str) -> str | None:
    """Best-effort lookup of an already-running pod from a previous session, by name,
    via `runpodctl get pod`. Returns None (rather than raising) on any parse failure,
    since falling through to "create a new pod" is the safe default and this is
    optimizing for reuse, not required for correctness.
    """
    import subprocess

    try:
        result = subprocess.run(["runpodctl", "get", "pod", "-a"], check=True, capture_output=True, text=True)
        rows = runpod_app._parse_pod_table(result.stdout)
        for row in rows:
            name = row.get("NAME") or row.get("Name")
            pod_id = row.get("ID") or row.get("Id")
            if name == pod_name and pod_id:
                return pod_id
    except Exception:
        logger.warning("could not check for an existing pod, will create a new one", exc_info=True)
    return None


def ensure_ready(settings: Settings) -> runpod_app.PodInfo:
    """Ensure a running pod with this project synced and set up exists. Safe to call
    at the start of every command that needs the GPU; each step is a no-op if already
    done (existing pod reused, setup script skips completed work).
    """
    runpod_app.ensure_authenticated()
    runpod_app.ensure_ssh_key()

    if not settings.runpod_network_volume_id:
        raise RuntimeError(
            "VOICECLONE_RUNPOD_NETWORK_VOLUME_ID not set. Create a Network Volume via the RunPod web "
            "console (Storage -> New Network Volume) and put its id in .env."
        )

    pod_id = find_existing_pod(settings.runpod_pod_name)
    if pod_id is None:
        pod_id = runpod_app.create_pod(
            name=settings.runpod_pod_name,
            gpu_type=settings.runpod_gpu_type,
            network_volume_id=settings.runpod_network_volume_id,
            env={"HF_TOKEN": _env_passthrough("HF_TOKEN")},
        )
    else:
        logger.info("reusing existing pod %s", pod_id)

    pod = runpod_app.wait_for_running(pod_id)

    runpod_app.run_remote(pod, f"mkdir -p {REMOTE_PROJECT_DIR}")
    runpod_app.sync_to_pod(pod, PROJECT_ROOT / "src", f"{REMOTE_PROJECT_DIR}/src")
    runpod_app.sync_to_pod(pod, PROJECT_ROOT / "scripts", f"{REMOTE_PROJECT_DIR}/scripts")
    runpod_app.sync_to_pod(pod, PROJECT_ROOT / "configs", f"{REMOTE_PROJECT_DIR}/configs")
    _scp_file(pod, PROJECT_ROOT / "pyproject.toml", f"{REMOTE_PROJECT_DIR}/pyproject.toml")

    setup_cmd = f"chmod +x {REMOTE_PROJECT_DIR}/scripts/pod_setup.sh && {REMOTE_PROJECT_DIR}/scripts/pod_setup.sh"
    runpod_app.stream_remote(pod, setup_cmd)
    return pod


def _scp_file(pod: runpod_app.PodInfo, local: Path, remote: str) -> None:
    import subprocess

    subprocess.run(
        ["scp", "-i", str(runpod_app.SSH_KEY_PATH), "-P", str(pod.ssh_port),
         "-o", "StrictHostKeyChecking=accept-new",
         str(local), f"root@{pod.ip}:{remote}"],
        check=True,
    )


def _env_passthrough(name: str) -> str:
    import os

    return os.environ.get(name, "")


def remote_voiceclone(pod: runpod_app.PodInfo, args: list[str], stream: bool = True) -> str:
    """Run `voiceclone remote <args>` on the pod, inside its venv."""
    arg_str = " ".join(shlex.quote(a) for a in args)
    command = f"source /workspace/venv/bin/activate && cd {REMOTE_PROJECT_DIR} && voiceclone remote {arg_str}"
    if stream:
        runpod_app.stream_remote(pod, command)
        return ""
    return runpod_app.run_remote(pod, command).stdout
