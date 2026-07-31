"""Write CosyVoice3-format manifests (wav.scp/text/utt2spk/spk2utt[/instruct]).

CosyVoice3's own `local/prepare_data.py` only knows how to parse a LibriTTS-shaped
directory tree, which our recordings don't follow. These four (or five) flat,
whitespace-separated files are the actual contract every downstream stage reads
(confirmed against tools/extract_embedding.py, tools/extract_speech_token.py, and
tools/make_parquet_list.py in the upstream repo) — so we write them directly instead
of routing through the upstream script.
"""

from __future__ import annotations

import random
from pathlib import Path

from voiceclone.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def _write_kv_lines(path: Path, items: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k in sorted(items):
            f.write(f"{k} {items[k]}\n")


def write_manifest(
    dest_dir: Path,
    utt_to_wav: dict[str, Path],
    utt_to_text: dict[str, str],
    speaker_id: str,
    instruct_text: str | None = DEFAULT_INSTRUCT,
) -> None:
    """Write wav.scp/text/utt2spk/spk2utt[/instruct] for one manifest directory.

    All utterances are attributed to a single `speaker_id` — this pipeline is built
    for single-speaker personal voice cloning, not multi-speaker corpora.
    """
    missing_text = sorted(set(utt_to_wav) - set(utt_to_text))
    if missing_text:
        raise ValueError(f"{len(missing_text)} utterances have audio but no transcript: {missing_text[:5]}...")

    dest_dir.mkdir(parents=True, exist_ok=True)
    utt_ids = sorted(utt_to_wav)

    _write_kv_lines(dest_dir / "wav.scp", {u: str(utt_to_wav[u]) for u in utt_ids})
    _write_kv_lines(dest_dir / "text", {u: utt_to_text[u] for u in utt_ids})
    _write_kv_lines(dest_dir / "utt2spk", {u: speaker_id for u in utt_ids})
    (dest_dir / "spk2utt").write_text(f"{speaker_id} {' '.join(utt_ids)}\n", encoding="utf-8")

    if instruct_text:
        _write_kv_lines(dest_dir / "instruct", {u: instruct_text for u in utt_ids})

    logger.info("wrote manifest for %d utterances -> %s", len(utt_ids), dest_dir)


def read_kv_file(path: Path) -> dict[str, str]:
    """Read one of wav.scp/text/utt2spk (`{utt_id} {value...}` per line) into a dict."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        utt, value = line.split(maxsplit=1)
        out[utt] = value
    return out


def pick_enrollment_utterance(manifest_dir: Path) -> tuple[str, str, str]:
    """Pick a single utterance from a manifest dir to use as a zero-shot voice prompt:
    the longest available recording, on the theory that more audio gives the
    frontend's speaker-conditioning more to work with.

    Returns (utt_id, text, wav_path). Callers doing evaluation should point this at
    the `train` manifest, never `holdout` — the enrollment prompt is part of the
    inference pipeline, not something that should overlap with what's being scored.
    """
    wav_scp = read_kv_file(manifest_dir / "wav.scp")
    text = read_kv_file(manifest_dir / "text")
    longest_utt = max(wav_scp, key=lambda u: Path(wav_scp[u]).stat().st_size)
    return longest_utt, text[longest_utt], wav_scp[longest_utt]


def remap_wav_scp(src_wav_scp: Path, dest_wav_scp: Path, container_audio_dir: str) -> None:
    """Rewrite wav.scp's audio paths from local filesystem paths to their eventual
    location under a Modal Volume mount.

    wav.scp is written locally with real, inspectable local paths (see `write_manifest`)
    so the manifest is useful for local debugging on its own. Those paths are obviously
    wrong once the same bytes live in a container at `/dataset/...` — this produces a
    second copy of wav.scp with paths rewritten to `container_audio_dir/{filename}`,
    relying on chunk filenames being unique (they are: `{session_prefix}_{index}.wav`).
    """
    lines = src_wav_scp.read_text(encoding="utf-8").splitlines()
    remapped = []
    for line in lines:
        utt, local_path = line.split(maxsplit=1)
        filename = Path(local_path).name
        remapped.append(f"{utt} {container_audio_dir.rstrip('/')}/{filename}")
    dest_wav_scp.parent.mkdir(parents=True, exist_ok=True)
    dest_wav_scp.write_text("\n".join(remapped) + "\n", encoding="utf-8")


def split_dataset(
    utt_ids: list[str],
    cv_fraction: float,
    holdout_fraction: float,
    seed: int = 1986,
    min_cv: int = 10,
    min_holdout: int = 20,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic three-way split: train / cv / holdout.

    `cv` feeds validation loss during training and checkpoint averaging
    (`average_model.py --val_best`); `holdout` is never touched until final
    evaluation. Keeping them separate matters: selecting a checkpoint and then
    reporting its quality on the *same* data would overstate how good it actually is.
    """
    shuffled = sorted(utt_ids)  # sort first so the shuffle is reproducible regardless of dict/glob ordering
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_cv = max(min_cv, round(len(shuffled) * cv_fraction))
    n_holdout = max(min_holdout, round(len(shuffled) * holdout_fraction))
    if n_cv + n_holdout >= len(shuffled):
        raise ValueError(
            f"cv+holdout ({n_cv + n_holdout}) >= total utterances ({len(shuffled)}); record more data or "
            f"lower cv_fraction/eval_holdout_fraction"
        )

    holdout = shuffled[:n_holdout]
    cv = shuffled[n_holdout : n_holdout + n_cv]
    train = shuffled[n_holdout + n_cv :]
    logger.info(
        "split: %d train / %d cv / %d holdout (%.1f%% / %.1f%% held out)",
        len(train),
        len(cv),
        len(holdout),
        100 * n_cv / len(shuffled),
        100 * n_holdout / len(shuffled),
    )
    return train, cv, holdout


def prepare_speaker_manifests(
    utt_to_wav: dict[str, Path],
    utt_to_text: dict[str, str],
    speaker_id: str,
    dest_root: Path,
    cv_fraction: float,
    holdout_fraction: float,
    instruct_text: str | None = DEFAULT_INSTRUCT,
    seed: int = 1986,
) -> dict[str, Path]:
    """End-to-end: split + write the `train`, `cv`, and `holdout` manifest directories.

    Returns {"train": ..., "cv": ..., "holdout": ...}.
    """
    train_ids, cv_ids, holdout_ids = split_dataset(list(utt_to_wav), cv_fraction, holdout_fraction, seed)

    dirs = {"train": dest_root / "train", "cv": dest_root / "cv", "holdout": dest_root / "holdout"}
    id_groups = {"train": train_ids, "cv": cv_ids, "holdout": holdout_ids}

    for split, ids in id_groups.items():
        write_manifest(
            dirs[split],
            {u: utt_to_wav[u] for u in ids},
            {u: utt_to_text[u] for u in ids},
            speaker_id,
            # instruct is only meaningful for training data; cv still trains-adjacent (used for
            # val loss under the same conditioning scheme) so it keeps it, holdout does not.
            instruct_text=instruct_text if split in ("train", "cv") else None,
        )
    return dirs
