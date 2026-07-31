"""RunPod pod lifecycle and remote execution: this project's equivalent of
modal_app.py, but for RunPod instead of Modal.

RunPod has no "decorate a function, call it, it runs remotely" abstraction the way
Modal does. The pattern here instead is: create one GPU pod for a work session,
rsync this project + the dataset onto its Network Volume, run setup once, then run
each pipeline stage as a real shell command over SSH. `run_remote` /
`sync_to_pod` / `sync_from_pod` are the three primitives every other module in this
package builds on.

Two things this module deliberately does NOT try to do, and why:

- Create the Network Volume via API. `runpodctl` has no volume-creation command, and
  guessing at RunPod's REST/GraphQL schema for this one-time, rarely-repeated action
  without being able to test it live is a bad trade against just doing it once by
  hand. Create it via the RunPod web console (Storage -> New Network Volume) and put
  its id in `.env` as `VOICECLONE_RUNPOD_NETWORK_VOLUME_ID`.
- Parse `runpodctl get pod` output blind. It has no JSON output flag as of this
  writing, only a wider `-a/--allfields` table. `_parse_pod_table` below parses that
  table defensively (from its header row, not fixed column indices), but has not been
  run against real output yet, verify it against a live pod before trusting it for
  anything unattended.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_IMAGE = "nvidia/cuda:12.1.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 40
DEFAULT_SSH_PORT_LABEL = "22/tcp"
SERVE_PORT_LABEL = "8000/http"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Add it to .env (never pass it as a function argument or log it)."
        )
    return key


def ensure_authenticated() -> None:
    """One-time `runpodctl config --apiKey ...`, reading the key from the environment
    so it never appears in a command a shell history/log could capture as an argument
    list elsewhere in the codebase, this is the single place the raw key touches a
    subprocess argv. Does not use check=True: subprocess.CalledProcessError's default
    repr includes the full argv (the key included), so failure is handled explicitly
    with a redacted message instead.
    """
    result = subprocess.run(
        ["runpodctl", "config", "--apiKey", _api_key()],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"runpodctl authentication failed (command redacted): {result.stderr.strip()}")
    logger.info("runpodctl authenticated")


@dataclass(frozen=True)
class PodInfo:
    pod_id: str
    status: str
    ip: str | None = None
    ssh_port: int | None = None


def list_gpu_types() -> str:
    """Raw `runpodctl get cloud` output, for picking a real, currently-available GPU
    type/region rather than trusting the default in Settings.runpod_gpu_type blind.
    """
    result = subprocess.run(["runpodctl", "get", "cloud"], check=True, capture_output=True, text=True)
    return result.stdout


def create_pod(
    name: str,
    gpu_type: str,
    network_volume_id: str,
    image: str = DEFAULT_IMAGE,
    container_disk_gb: int = CONTAINER_DISK_GB,
    ports: str = f"{SERVE_PORT_LABEL},{DEFAULT_SSH_PORT_LABEL}",
    env: dict[str, str] | None = None,
    cost_ceiling: float | None = None,
    cloud: str | None = None,
) -> str:
    """Start a pod, return its id. Does not wait for it to be running (see
    `wait_for_running`), pod boot + volume attach typically takes a minute or two.

    `cloud` is `"secure"`, `"community"`, or `None` (no constraint, either pool,
    maximizes the odds of finding stock, which matters when a GPU type is showing
    "Low" stock for the volume's datacenter). Only force a specific pool if you have
    a reason to (secure cloud for a long unattended training run you don't want
    preempted, say) at the cost of a smaller pool to draw from.
    """
    cmd = [
        "runpodctl", "create", "pod",
        "--name", name,
        "--gpuType", gpu_type,
        "--imageName", image,
        "--networkVolumeId", network_volume_id,
        "--containerDiskSize", str(container_disk_gb),
        "--ports", ports,
    ]
    if cloud == "secure":
        cmd.append("--secureCloud")
    elif cloud == "community":
        cmd.append("--communityCloud")
    if cost_ceiling is not None:
        cmd += ["--cost", str(cost_ceiling)]

    # env vars (e.g. HF_TOKEN) are appended separately and never included in any
    # exception raised below -- subprocess.CalledProcessError's default repr includes
    # the full argv, which would otherwise leak them into logs/tracebacks.
    full_cmd = list(cmd)
    for k, v in (env or {}).items():
        full_cmd += ["--env", f"{k}={v}"]

    logger.info("creating pod '%s' (%s)", name, gpu_type)
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pod creation failed (command redacted): {result.stderr.strip()}")
    pod_id = _extract_pod_id(result.stdout)
    logger.info("pod created: %s", pod_id)
    return pod_id


def _extract_pod_id(create_output: str) -> str:
    """`runpodctl create pod` prints a human-readable confirmation, not JSON. Pod ids
    are RunPod's short alphanumeric ids; this looks for the id embedded in whichever
    phrasing the CLI uses ("pod ... created with ID <id>" in known versions) and falls
    back to the last bare alphanumeric token on the last non-empty line, since exact
    wording has changed between runpodctl versions before.
    """
    import re

    match = re.search(r"\bID[:\s]+([a-z0-9]{8,})\b", create_output, re.IGNORECASE)
    if match:
        return match.group(1)
    lines = [line.strip() for line in create_output.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"could not parse pod id from empty create output: {create_output!r}")
    tokens = lines[-1].split()
    candidate = tokens[-1].strip(".,:;")
    logger.warning(
        "pod id extraction fell back to last token of last output line (%r), "
        "verify this against real `runpodctl create pod` output and fix this parser if wrong",
        candidate,
    )
    return candidate


def _parse_pod_table(table_output: str) -> list[dict[str, str]]:
    """Header-driven parser for `runpodctl get pod -a`'s table output: splits the
    header row on 2+ spaces to find column boundaries, then slices each data row at
    those same boundaries. More robust than fixed-width or single-space-split parsing
    against a table whose column widths vary with content, but still unverified
    against real output, check this against `runpodctl get pod -a` directly if pod
    status lookups ever silently return wrong/empty fields.
    """
    lines = [line for line in table_output.splitlines() if line.strip()]
    if not lines:
        return []
    header_line = lines[0]
    col_starts = [0] + [m.end() for m in __import__("re").finditer(r"\S+\s{2,}", header_line)]
    headers = [header_line[a:b].strip() for a, b in zip(col_starts, col_starts[1:] + [None], strict=True)]

    rows = []
    for line in lines[1:]:
        values = [line[a:b].strip() for a, b in zip(col_starts, col_starts[1:] + [None], strict=True)]
        rows.append(dict(zip(headers, values, strict=True)))
    return rows


def get_pod(pod_id: str) -> PodInfo:
    result = subprocess.run(["runpodctl", "get", "pod", pod_id, "-a"], check=True, capture_output=True, text=True)
    rows = _parse_pod_table(result.stdout)
    if not rows:
        raise RuntimeError(f"pod {pod_id} not found in `runpodctl get pod` output")
    row = rows[0]
    status = row.get("STATUS") or row.get("DesiredStatus") or row.get("Status", "UNKNOWN")
    ip = row.get("IP") or row.get("PublicIP") or None
    ssh_port_raw = row.get("SSH Port") or row.get("SSHPort") or None
    return PodInfo(pod_id=pod_id, status=status, ip=ip or None, ssh_port=int(ssh_port_raw) if ssh_port_raw else None)


def wait_for_running(pod_id: str, timeout_s: int = 300, poll_interval_s: int = 10) -> PodInfo:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = get_pod(pod_id)
        logger.info("pod %s status: %s", pod_id, info.status)
        if info.status.upper() in ("RUNNING",) and info.ip and info.ssh_port:
            return info
        time.sleep(poll_interval_s)
    raise TimeoutError(f"pod {pod_id} did not reach RUNNING with SSH details within {timeout_s}s")


def stop_pod(pod_id: str) -> None:
    subprocess.run(["runpodctl", "stop", "pod", pod_id], check=True, capture_output=True, text=True)
    logger.info("stopped pod %s", pod_id)


def remove_pod(pod_id: str) -> None:
    subprocess.run(["runpodctl", "remove", "pod", pod_id], check=True, capture_output=True, text=True)
    logger.info("removed pod %s", pod_id)


# --- SSH execution -----------------------------------------------------------------

SSH_KEY_PATH = Path.home() / ".ssh" / "voiceclone_runpod_ed25519"


def ensure_ssh_key() -> Path:
    """Generate a dedicated keypair for this project if one doesn't exist yet, and
    register the public half with RunPod. Separate from any other SSH key already on
    this machine (including a stray `runpod_id_ed25519.pub` found earlier with no
    matching private key) so this project's access doesn't depend on unrelated key
    material.
    """
    if not SSH_KEY_PATH.exists():
        logger.info("generating SSH keypair at %s", SSH_KEY_PATH)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY_PATH), "-N", "", "-C", "voiceclone-runpod"],
            check=True, capture_output=True, text=True,
        )
    subprocess.run(
        ["runpodctl", "ssh", "add-key", "--key-file", str(SSH_KEY_PATH) + ".pub"],
        check=True, capture_output=True, text=True,
    )
    logger.info("SSH key registered with RunPod")
    return SSH_KEY_PATH


def ssh_base_args(pod: PodInfo) -> list[str]:
    return [
        "ssh",
        "-i", str(SSH_KEY_PATH),
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(pod.ssh_port),
        f"root@{pod.ip}",
    ]


def run_remote(
    pod: PodInfo, command: str, timeout_s: int | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run one shell command on the pod over SSH, streaming nothing back but capturing
    stdout/stderr, for long training runs, prefer `stream_remote` so progress is
    visible rather than silent until the whole thing finishes.
    """
    cmd = [*ssh_base_args(pod), command]
    logger.info("[pod %s] %s", pod.pod_id, command)
    return subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True, check=check)


def stream_remote(pod: PodInfo, command: str, check: bool = True) -> int:
    """Like `run_remote`, but inherits stdout/stderr so output streams live, use this
    for anything that takes more than a few seconds (training, feature extraction).
    """
    cmd = [*ssh_base_args(pod), command]
    logger.info("[pod %s] %s", pod.pod_id, command)
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def sync_to_pod(pod: PodInfo, local_path: Path, remote_path: str) -> None:
    remote = f"root@{pod.ip}:{remote_path}"
    cmd = [
        "rsync", "-az", "--progress",
        "-e", f"ssh -i {shlex.quote(str(SSH_KEY_PATH))} -o StrictHostKeyChecking=accept-new -p {pod.ssh_port}",
        f"{local_path}/",
        remote,
    ]
    logger.info("syncing %s -> pod %s:%s", local_path, pod.pod_id, remote_path)
    subprocess.run(cmd, check=True)


def sync_from_pod(pod: PodInfo, remote_path: str, local_path: Path) -> None:
    remote = f"root@{pod.ip}:{remote_path}"
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-az", "--progress",
        "-e", f"ssh -i {shlex.quote(str(SSH_KEY_PATH))} -o StrictHostKeyChecking=accept-new -p {pod.ssh_port}",
        f"{remote}/",
        str(local_path),
    ]
    logger.info("syncing pod %s:%s -> %s", pod.pod_id, remote_path, local_path)
    subprocess.run(cmd, check=True)


def run_remote_json(pod: PodInfo, command: str) -> object:
    """Run a remote command expected to print exactly one JSON value on stdout, and
    parse it, used for stage scripts that need to return structured results (e.g. the
    evaluation report path) rather than just a pass/fail exit code.
    """
    result = run_remote(pod, command)
    return json.loads(result.stdout.strip().splitlines()[-1])
