from pathlib import Path

import pytest

from voiceclone.evaluation.report import EvalResult, save_report, summarize, to_markdown


def _results() -> list[EvalResult]:
    return [
        EvalResult("exp_a", "utt1", wer=0.10, speaker_similarity=0.80, utmos=3.5),
        EvalResult("exp_a", "utt2", wer=0.20, speaker_similarity=0.70, utmos=3.7),
        EvalResult("exp_b", "utt1", wer=0.05, speaker_similarity=0.90, utmos=4.0),
        EvalResult("exp_b", "utt2", wer=0.05, speaker_similarity=0.92, utmos=4.2),
    ]


def test_summarize_aggregates_per_experiment() -> None:
    summary = summarize(_results())
    assert set(summary.index) == {"exp_a", "exp_b"}
    assert summary.loc["exp_a", "n_utterances"] == 2
    assert summary.loc["exp_a", "wer_mean"] == pytest.approx(0.15)


def test_summarize_sorts_by_speaker_similarity_descending() -> None:
    summary = summarize(_results())
    assert list(summary.index) == ["exp_b", "exp_a"]  # exp_b has higher mean speaker_similarity


def test_to_markdown_contains_all_experiments() -> None:
    md = to_markdown(summarize(_results()))
    assert "exp_a" in md
    assert "exp_b" in md
    assert md.startswith("| experiment |")


def test_save_report_writes_all_three_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "report"
    md_path = save_report(_results(), out_dir)

    assert md_path == out_dir / "eval_summary.md"
    assert (out_dir / "eval_summary.md").exists()
    assert (out_dir / "eval_summary.csv").exists()
    assert (out_dir / "eval_per_utterance.csv").exists()
    assert "exp_b" in md_path.read_text()
