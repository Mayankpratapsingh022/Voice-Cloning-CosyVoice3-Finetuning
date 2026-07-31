"""Aggregate per-utterance eval results into the experiment-comparison table that's
the actual point of running a multi-config sweep in the first place — see plan
Section 7: "this table, not just the final audio, is what makes the project read as
rigorous work."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvalResult:
    experiment_name: str
    utt_id: str
    wer: float
    speaker_similarity: float
    utmos: float


def to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def summarize(results: list[EvalResult]) -> pd.DataFrame:
    """One row per experiment: mean +/- std for each metric, sorted by speaker
    similarity descending since that's the plan's primary metric (see PLAN.md
    Section 7 / the CosyVoice3 selection rationale)."""
    df = to_dataframe(results)
    summary = df.groupby("experiment_name").agg(
        n_utterances=("utt_id", "count"),
        wer_mean=("wer", "mean"),
        wer_std=("wer", "std"),
        speaker_similarity_mean=("speaker_similarity", "mean"),
        speaker_similarity_std=("speaker_similarity", "std"),
        utmos_mean=("utmos", "mean"),
        utmos_std=("utmos", "std"),
    )
    return summary.sort_values("speaker_similarity_mean", ascending=False)


def to_markdown(summary: pd.DataFrame) -> str:
    rows = ["| experiment | n | WER ↓ | speaker similarity ↑ | UTMOS ↑ |", "|---|---|---|---|---|"]
    for name, row in summary.iterrows():
        rows.append(
            f"| {name} | {int(row.n_utterances)} "
            f"| {row.wer_mean:.3f} ± {row.wer_std:.3f} "
            f"| {row.speaker_similarity_mean:.3f} ± {row.speaker_similarity_std:.3f} "
            f"| {row.utmos_mean:.2f} ± {row.utmos_std:.2f} |"
        )
    return "\n".join(rows)


def save_report(results: list[EvalResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    to_dataframe(results).to_csv(out_dir / "eval_per_utterance.csv", index=False)

    summary = summarize(results)
    summary.to_csv(out_dir / "eval_summary.csv")
    md_path = out_dir / "eval_summary.md"
    md_path.write_text(to_markdown(summary) + "\n", encoding="utf-8")

    logger.info("wrote eval report -> %s", out_dir)
    return md_path
