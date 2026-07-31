"""Generate speech for the held-out sentences with a fine-tuned checkpoint, then
score it against the plan's three objective metrics and write the comparison report.

The holdout set (see data/manifest.py's `split_dataset`) is never touched by training
or by checkpoint averaging — this is the only place in the pipeline that reads it, and
it does so purely to measure generalization.
"""

from __future__ import annotations

from pathlib import Path

from voiceclone.config import Paths, Settings, get_settings
from voiceclone.data.manifest import pick_enrollment_utterance, read_kv_file
from voiceclone.evaluation.report import EvalResult, save_report
from voiceclone.logging_utils import get_logger
from voiceclone.modal_app import (
    GPU_CHEAP_INFERENCE,
    STANDARD_VOLUMES,
    app,
    checkpoint_volume,
    cosyvoice_image,
    eval_image,
)

logger = get_logger(__name__)

ENROLLMENT_SPK_ID = "voiceclone_holdout_eval"


@app.function(image=cosyvoice_image, volumes=STANDARD_VOLUMES, gpu=GPU_CHEAP_INFERENCE, timeout=3600)
def generate_holdout_samples(speaker_prefix: str, experiment_name: str) -> str:
    """Synthesize every holdout sentence with this experiment's fine-tuned checkpoint.

    Returns the container-side directory the generated WAVs were written to.
    """
    from voiceclone.inference.engine import VoiceCloneEngine

    checkpoint_dir = Paths.CHECKPOINT_DIR / speaker_prefix / experiment_name / "inference_ready"
    train_dir = Paths.DATASET_DIR / speaker_prefix / "train"
    holdout_dir = Paths.DATASET_DIR / speaker_prefix / "holdout"
    out_dir = Paths.CHECKPOINT_DIR / speaker_prefix / experiment_name / "eval_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = VoiceCloneEngine(str(checkpoint_dir))
    _enrollment_utt, enrollment_text, enrollment_wav = pick_enrollment_utterance(train_dir)
    engine.enroll(enrollment_text, enrollment_wav, spk_id=ENROLLMENT_SPK_ID)

    holdout_text = read_kv_file(holdout_dir / "text")
    for utt_id, tts_text in holdout_text.items():
        engine.synthesize_to_file(tts_text, out_dir / f"{utt_id}_0.wav", spk_id=ENROLLMENT_SPK_ID)

    checkpoint_volume.commit()
    logger.info("generated %d holdout samples for %s -> %s", len(holdout_text), experiment_name, out_dir)
    return str(out_dir)


@app.function(image=eval_image, volumes=STANDARD_VOLUMES, gpu=GPU_CHEAP_INFERENCE, timeout=3600)
def evaluate_experiment(speaker_prefix: str, experiment_name: str) -> list[EvalResult]:
    """Score one experiment's holdout generations: WER, speaker similarity (against
    the real holdout recording of the same speaker), and UTMOS naturalness.
    """
    from voiceclone.evaluation.metrics import naturalness_mos, speaker_similarity, word_error_rate

    device = "cuda"
    holdout_dir = Paths.DATASET_DIR / speaker_prefix / "holdout"
    holdout_text = read_kv_file(holdout_dir / "text")
    holdout_wav = read_kv_file(holdout_dir / "wav.scp")
    samples_dir = Paths.CHECKPOINT_DIR / speaker_prefix / experiment_name / "eval_samples"

    results: list[EvalResult] = []
    for utt_id, reference_text in holdout_text.items():
        generated_path = samples_dir / f"{utt_id}_0.wav"
        if not generated_path.exists():
            logger.warning("no generated sample for %s, skipping (did generate_holdout_samples run?)", utt_id)
            continue
        reference_wav = Path(holdout_wav[utt_id])
        results.append(
            EvalResult(
                experiment_name=experiment_name,
                utt_id=utt_id,
                wer=word_error_rate(reference_text, generated_path, device=device),
                speaker_similarity=speaker_similarity(generated_path, reference_wav, device=device),
                utmos=naturalness_mos(generated_path, device=device),
            )
        )
    return results


@app.function(image=eval_image, volumes={"/checkpoints": checkpoint_volume}, cpu=1.0, timeout=6 * 3600)
def evaluate_sweep(speaker_prefix: str, experiment_names: list[str], settings: Settings | None = None) -> str:
    """Generate + score every experiment in the sweep, write the comparison report.

    Returns the container-side path to the markdown summary.
    """
    settings = settings or get_settings()
    for name in experiment_names:
        generate_holdout_samples.remote(speaker_prefix, name)

    all_results: list[EvalResult] = []
    for name in experiment_names:
        all_results.extend(evaluate_experiment.remote(speaker_prefix, name))

    out_dir = Paths.CHECKPOINT_DIR / speaker_prefix / "eval_report"
    md_path = save_report(all_results, out_dir)
    checkpoint_volume.commit()
    return str(md_path)
