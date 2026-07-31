#!/usr/bin/env bash
# Runs on the RunPod pod (over SSH) to prepare /workspace once per Network Volume.
# Idempotent: skips work that is already done, since the volume persists across pod
# create/destroy cycles within a session and across sessions.
set -euo pipefail

WORKSPACE=/workspace
VENV="$WORKSPACE/venv"
REPO_ROOT="$WORKSPACE/CosyVoice"
COSYVOICE_COMMIT="main"  # TODO: pin to a specific commit once the pipeline is validated (see runpod_app.py's DEFAULT_IMAGE comment)

mkdir -p "$WORKSPACE"

if [ ! -d "$VENV" ]; then
  echo "[pod_setup] installing system packages"
  apt-get update -qq
  apt-get install -y -qq git git-lfs sox libsox-dev ffmpeg build-essential python3.10 python3.10-venv python3-pip >/dev/null

  echo "[pod_setup] creating venv at $VENV"
  python3.10 -m venv "$VENV"
else
  echo "[pod_setup] venv already exists, skipping system package install"
fi

source "$VENV/bin/activate"

if [ ! -d "$REPO_ROOT" ]; then
  echo "[pod_setup] cloning CosyVoice3"
  git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$REPO_ROOT"
  (cd "$REPO_ROOT" && git submodule update --init --recursive)
else
  echo "[pod_setup] CosyVoice repo already present, skipping clone"
fi

if [ ! -f "$WORKSPACE/.setup_complete" ]; then
  # No -q anywhere in this block: normal pip output (download progress, "Collecting X",
  # wheel builds) stays visible. Deliberate -- this block can run 10-25 min, dominated
  # by torch/tensorrt downloads and deepspeed compiling CUDA extensions from source,
  # and a silent multi-minute gap is indistinguishable from a hang without it.
  echo "[pod_setup] (1/5) upgrading pip"
  pip install --upgrade pip

  echo "[pod_setup] (2/5) installing torch + torchaudio (cu121) -- large download, several minutes"
  pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

  echo "[pod_setup] (3/5) installing CosyVoice3 requirements.txt -- includes deepspeed, which"
  echo "[pod_setup]     compiles CUDA extensions from source and can take 5-15 min on its own"
  pip install -r "$REPO_ROOT/requirements.txt" || true  # some deps in this file are platform-conditional; non-fatal

  echo "[pod_setup] (4/5) installing onnxruntime-gpu"
  pip install onnxruntime-gpu==1.18.0

  echo "[pod_setup] (5/5) installing voiceclone package"
  pip install -e "$WORKSPACE/voiceclone-project"

  touch "$WORKSPACE/.setup_complete"
  echo "[pod_setup] setup complete"
else
  echo "[pod_setup] dependencies already installed, skipping"
fi

echo "[pod_setup] done"
