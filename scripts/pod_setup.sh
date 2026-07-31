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
  echo "[pod_setup] (1/8) upgrading pip"
  pip install --upgrade pip

  # setuptools>=81 dropped the pkg_resources module entirely, which openai-whisper's
  # setup.py needs. Pinning it in *this* venv is necessary but NOT sufficient: pip
  # builds PEP 517 packages (anything with a pyproject.toml-based build, which
  # openai-whisper uses) in a separate, temporary, isolated environment by default,
  # with its own freshly-resolved setuptools -- our pin here never reaches that
  # isolated build env. Confirmed live: pinning setuptools<81 here changed nothing,
  # the build still failed from inside /tmp/pip-build-env-*/.
  #
  # wheel is needed alongside it: with --no-build-isolation (below), pip no longer
  # auto-installs a package's declared build tools into an isolated env for you --
  # you're on the hook for having them in the real venv yourself. Without wheel
  # present, setuptools doesn't know the `bdist_wheel` command, and the build fails
  # with "error: invalid command 'bdist_wheel'" -- confirmed live, one layer past the
  # pkg_resources failure once that part was actually fixed.
  echo "[pod_setup] (2/8) pinning setuptools<81 and installing wheel in this venv"
  pip install "setuptools<81" wheel

  echo "[pod_setup] (3/8) installing torch + torchaudio (cu121) -- large download, several minutes"
  pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

  # The actual fix: --no-build-isolation makes pip build using *this* venv's already-
  # installed packages (including the setuptools<81 pin above) instead of spinning up
  # an isolated build env with its own unpinned, too-new setuptools. Installed here,
  # its own explicit step, before requirements.txt: once it's already satisfied at the
  # pinned version, pip won't try to rebuild it again (with the same isolation problem)
  # when it hits openai-whisper's line in requirements.txt below.
  echo "[pod_setup] (4/8) installing openai-whisper with build isolation disabled"
  pip install --no-build-isolation openai-whisper==20231117

  echo "[pod_setup] (5/8) installing CosyVoice3 requirements.txt -- includes deepspeed, which"
  echo "[pod_setup]     compiles CUDA extensions from source and can take 5-15 min on its own"
  # No `|| true` here. It used to be here on the (wrong) assumption that this file's
  # platform-conditional lines (e.g. `onnxruntime==...; sys_platform == "darwin"`)
  # needed protecting -- they don't, pip's own marker evaluation already skips those
  # correctly on its own (confirmed live: "Ignoring onnxruntime: markers ... don't
  # match your environment"). What `|| true` actually did was mask a real failure
  # (openai-whisper's build breaking) and let the script limp to the end and touch
  # .setup_complete anyway -- so a later, actually-fixed rerun kept seeing that stale
  # marker and skipping this whole block, silently, for every run since. If this line
  # fails now, the script stops (set -e) and .setup_complete is never written, so the
  # next run properly retries instead of lying about having succeeded.
  pip install -r "$REPO_ROOT/requirements.txt"

  # The --extra-index-url is load-bearing, not decoration: plain PyPI's
  # onnxruntime-gpu 1.18.0 is built against CUDA 11, so on this CUDA 12 image it
  # fails to load libcublasLt.so.11 and silently falls back to CPU (confirmed live:
  # speech-token extraction still completed, just far slower than it should have).
  # This index is Microsoft's CUDA 12 build feed, and is exactly why CosyVoice3's
  # own requirements.txt carries the same URL -- dropping it when hoisting this into
  # its own explicit step is what caused the fallback.
  echo "[pod_setup] (6/8) installing onnxruntime-gpu (CUDA 12 build)"
  pip install onnxruntime-gpu==1.18.0 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

  # modelscope is in requirements.txt too, but if *any* package in that combined
  # install fails to resolve, pip can silently drop others from the same batch
  # rather than partially installing -- exactly what happened here (openai-whisper's
  # build failure took modelscope down with it, and download_pretrained()'s `from
  # modelscope import snapshot_download` broke as a result). Its own dedicated,
  # unprotected step now, so a requirements.txt failure elsewhere can't hide this.
  echo "[pod_setup] (7/8) installing modelscope (needed by download_pretrained, not just requirements.txt)"
  pip install modelscope==1.20.0

  echo "[pod_setup] (8/8) installing voiceclone package"
  pip install -e "$WORKSPACE/voiceclone-project"

  touch "$WORKSPACE/.setup_complete"
  echo "[pod_setup] setup complete"
else
  echo "[pod_setup] dependencies already installed, skipping"
fi

echo "[pod_setup] done"
