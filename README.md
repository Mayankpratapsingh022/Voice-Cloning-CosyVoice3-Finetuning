# voiceclone

Full-parameter fine-tuning of [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) for personal
voice cloning, with an objective evaluation harness and a deployed inference demo.

## Motivation

Most voice cloning demos rely on zero-shot conditioning: a foundation model generates speech in a
target voice from a short reference clip, with no training involved. This project takes a different,
more rigorous approach: it fine-tunes CosyVoice3's model weights directly on a personal voice dataset,
compares multiple training configurations against each other with objective metrics, and ships the
result as a working, deployed service rather than a one-off notebook.

The goal is to demonstrate an end-to-end applied ML workflow: data collection and preprocessing, model
selection backed by benchmark evidence, a real training and evaluation loop, and production-style
deployment, all for a task (voice cloning) that is central to how voice AI products actually work.

## Why CosyVoice3

CosyVoice3 was chosen after comparing it against several other open-source TTS models (Fish
Speech/OpenAudio, F5-TTS, Higgs Audio v3) on two criteria: reported speaker-similarity and
intelligibility benchmarks, and whether the model supports genuine full-parameter fine-tuning rather
than zero-shot-only cloning. CosyVoice3 has the strongest published speaker-similarity numbers among
the models with a documented full fine-tuning path, is Apache 2.0 licensed, and its architecture
(a language model, a flow-matching acoustic model, and a vocoder trained as three separable
components) allows for a direct comparison between full fine-tuning and a targeted, cheaper
fine-tune of just the acoustic component. The full reasoning, including why the alternatives were
ruled out, is in `PLAN.md` (kept local, not part of the published repository).

## Architecture

CosyVoice3's pipeline has three components, each trainable independently:

1. `llm`: a Qwen2-based autoregressive model that converts text into discrete speech tokens.
2. `flow`: a conditional flow-matching model (DiT-based) that converts speech tokens and a speaker
   embedding into acoustic features. This is the component most responsible for voice identity.
3. `hift`: a HiFi-GAN-derived vocoder (with source-filter harmonic excitation and multi-band iSTFT
   generation) that renders the acoustic features into an audio waveform.

A "full fine-tune" trains all three; a targeted fine-tune trains only `flow`, leaving the language
model and vocoder at their pretrained weights. Both are run as separate experiments and compared.

```
src/voiceclone/
    config.py            Runtime settings (speaker id, RunPod/HF identifiers, thresholds) via .env
    cli.py                Command-line entry point: voiceclone <group> <command>
    runpod_app.py           Pod lifecycle (create, wait, stop, remove) and SSH/rsync execution
    orchestrate.py            Local driver: ensures a pod is ready, syncs this project onto it
    data/
        audio_preprocess.py   Voice-activity-based chunking, loudness normalization, format conversion
        transcribe.py           Transcription via local faster-whisper or the OpenAI API
        manifest.py               Training manifest format, train/cv/holdout split
        pipeline.py                 Local orchestration: raw recordings to manifests
        hf_dataset.py                 Push/pull the prepared dataset via a private HF dataset repo
        remote_stages.py               CosyVoice3 feature extraction, runs on the pod
    training/
        experiment.py            Experiment configuration and per-run hyperparameter overrides
        train.py                   Fine-tuning, checkpoint averaging, and export, runs on the pod
    evaluation/
        metrics.py                Word error rate, speaker similarity, predicted naturalness (UTMOS)
        report.py                   Aggregated comparison table across experiments
        run_eval.py                  Holdout generation and scoring, runs on the pod
    inference/
        engine.py                Inference wrapper around the fine-tuned checkpoint
        serve.py                   Web demo (FastAPI + Gradio), runs on the pod
tests/                          Unit tests for the components that do not require a GPU
configs/experiments.yaml       The set of fine-tuning experiments to run
scripts/
    record_checklist.md      Practical guide for recording a training dataset
    pod_setup.sh                One-time environment setup run on the pod (idempotent)
```

Compute runs on a rented [RunPod](https://runpod.io) GPU pod, not locally. One pod is used per work
session: `voiceclone` commands that need the GPU create or reuse it, sync this project onto its
Network Volume, and run the actual work over SSH. The prepared dataset moves between your machine and
the pod through a private HuggingFace dataset repo, not a direct transfer, so the pod does not depend
on your machine staying reachable during a long run.

## Data

Training data can come from either dedicated recordings (see `scripts/record_checklist.md` for a
recording protocol covering phoneme coverage, emotional range, and natural conversational speech) or
existing narration, such as a personal video archive, provided it is a clean single-speaker track.
Audio is chunked on detected speech boundaries, transcribed, and split into three sets:

- `train`: used for gradient updates.
- `cv`: held out from training, used during training to select which checkpoints to average together.
- `holdout`: never used for training or checkpoint selection, reserved for final evaluation only.

Keeping `cv` and `holdout` separate matters: if the same data were used to pick the best checkpoint
and to report that checkpoint's quality, the reported numbers would be optimistic.

## Evaluation

Every trained checkpoint is scored on the `holdout` set with three objective metrics:

- Word error rate, via an independent ASR model, as a proxy for intelligibility.
- Speaker similarity, via WavLM speaker-verification embeddings, as a proxy for how closely the
  cloned voice matches the target speaker.
- Predicted naturalness (UTMOS), a reference-free mean-opinion-score estimate.

Results across all configured experiments are written as a single comparison table, not just as
audio samples, so model selection is based on evidence rather than casual listening.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[data,eval,dev]"
cp .env.example .env
```

Fill in `.env`: your speaker id, a RunPod API key, a HuggingFace token (with write access, used for
both the private dataset repo and reading the base checkpoint), and a RunPod Network Volume id.
The Network Volume has to be created once by hand, through the RunPod web console (Storage -> New
Network Volume), since RunPod's CLI has no command for it. Secrets are read from the environment,
never committed, and never printed by any command here.

## Usage

```bash
# Chunk, transcribe, and split raw recordings into training manifests (local, no GPU required)
voiceclone data prepare --raw-sessions-dir raw_sessions --dest data/<speaker>

# Push the prepared dataset to a private HuggingFace dataset repo
voiceclone data upload --dest data/<speaker>

# From here on, each command creates or reuses a pod, syncs this project onto it, and
# runs the actual work over SSH
voiceclone pretrained download
voiceclone data extract

voiceclone train list
voiceclone train run <experiment-name>
voiceclone train sweep

voiceclone eval run <experiment-name> [<experiment-name> ...]

voiceclone serve start --experiment <experiment-name>

# Pod management
voiceclone pod status
voiceclone pod gpus     # currently available GPU types/pricing for your account
voiceclone pod stop     # stop billing, keep the Network Volume
voiceclone pod remove   # terminate the pod entirely
```

## Testing

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Tests cover the components that do not require a GPU or a live checkpoint: manifest generation and
splitting, the training-configuration file format, audio chunking and normalization, format
conversion, and evaluation-report aggregation.

## Status

This is an active project. The data pipeline, training orchestration, and evaluation harness are
implemented and unit tested. Full pod-based execution has not yet been run end to end against live
RunPod infrastructure; a full fine-tuning run and its results are in progress.

## License

MIT. See `LICENSE`.
