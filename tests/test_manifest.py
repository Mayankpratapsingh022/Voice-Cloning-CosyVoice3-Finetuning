from pathlib import Path

import pytest

from voiceclone.data.manifest import (
    read_kv_file,
    remap_wav_scp,
    split_dataset,
    write_manifest,
)


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    utt_to_wav = {"spk_00001": tmp_path / "a.wav", "spk_00002": tmp_path / "b.wav"}
    utt_to_text = {"spk_00001": "hello there", "spk_00002": "general kenobi"}

    dest = tmp_path / "manifest"
    write_manifest(dest, utt_to_wav, utt_to_text, speaker_id="spk", instruct_text="do the thing")

    assert read_kv_file(dest / "text") == utt_to_text
    assert read_kv_file(dest / "utt2spk") == {"spk_00001": "spk", "spk_00002": "spk"}
    assert (dest / "spk2utt").read_text().strip() == "spk spk_00001 spk_00002"
    assert read_kv_file(dest / "instruct") == {"spk_00001": "do the thing", "spk_00002": "do the thing"}


def test_write_manifest_rejects_missing_transcript(tmp_path: Path) -> None:
    utt_to_wav = {"spk_00001": tmp_path / "a.wav"}
    with pytest.raises(ValueError, match="no transcript"):
        write_manifest(tmp_path / "manifest", utt_to_wav, {}, speaker_id="spk")


def test_split_dataset_is_deterministic_and_disjoint() -> None:
    utt_ids = [f"utt_{i:03d}" for i in range(200)]

    train1, cv1, holdout1 = split_dataset(utt_ids, cv_fraction=0.05, holdout_fraction=0.05, seed=7)
    train2, cv2, holdout2 = split_dataset(utt_ids, cv_fraction=0.05, holdout_fraction=0.05, seed=7)

    assert (train1, cv1, holdout1) == (train2, cv2, holdout2)
    assert set(train1) & set(cv1) == set()
    assert set(train1) & set(holdout1) == set()
    assert set(cv1) & set(holdout1) == set()
    assert set(train1) | set(cv1) | set(holdout1) == set(utt_ids)


def test_split_dataset_different_seed_gives_different_split() -> None:
    utt_ids = [f"utt_{i:03d}" for i in range(200)]
    _, _, holdout_a = split_dataset(utt_ids, 0.05, 0.05, seed=1)
    _, _, holdout_b = split_dataset(utt_ids, 0.05, 0.05, seed=2)
    assert holdout_a != holdout_b


def test_split_dataset_rejects_too_small_dataset() -> None:
    utt_ids = [f"utt_{i}" for i in range(10)]
    with pytest.raises(ValueError, match="record more data"):
        split_dataset(utt_ids, cv_fraction=0.5, holdout_fraction=0.5, min_cv=10, min_holdout=10)


def test_remap_wav_scp_swaps_prefix_keeps_filename(tmp_path: Path) -> None:
    src = tmp_path / "wav.scp"
    src.write_text(
        "utt_00001 /Users/me/project/chunks/utt_00001.wav\n"
        "utt_00002 /Users/me/project/chunks/utt_00002.wav\n"
    )

    dest = tmp_path / "remapped" / "wav.scp"
    remap_wav_scp(src, dest, "/dataset/mayank/chunks")

    assert read_kv_file(dest) == {
        "utt_00001": "/dataset/mayank/chunks/utt_00001.wav",
        "utt_00002": "/dataset/mayank/chunks/utt_00002.wav",
    }
