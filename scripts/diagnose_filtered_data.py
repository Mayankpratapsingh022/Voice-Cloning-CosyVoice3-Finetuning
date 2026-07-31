#!/usr/bin/env python
"""Report which of CosyVoice3's training-data filters drop which utterances.

Motivation: CosyVoice3's executor.cv() sets info_dict["tag"] only inside its
per-batch loop, then reads it afterwards, so an empty dataloader surfaces as
`KeyError: 'tag'` rather than anything mentioning data. If training dies there, the
real question is "what got filtered out and why", which is what this answers.

Mirrors the checks in cosyvoice/dataset/processor.py::filter against a packaged
parquet shard, without running any of the training machinery.

Usage (on the pod):
    python scripts/diagnose_filtered_data.py --split cv
    python scripts/diagnose_filtered_data.py --split train
"""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd
import torchaudio

# Defaults from examples/libritts/cosyvoice3/conf/cosyvoice3.yaml's `filter` block.
DEFAULTS = {
    "max_length": 6000,
    "min_length": 100,
    "token_max_length": 200,
    "token_min_length": 1,
    "min_output_input_ratio": 0.0005,
    "max_output_input_ratio": 1.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="mayank")
    parser.add_argument("--split", default="cv")
    parser.add_argument("--dataset-root", default="/workspace/dataset")
    args = parser.parse_args()

    parquet_dir = Path(args.dataset_root) / args.speaker / args.split / "parquet"
    # Read data.list rather than globbing: it is what training itself consumes, so a
    # mismatch between what it references and what exists on disk is exactly the
    # failure worth reporting (see _verify_parquet_shards in data/remote_stages.py).
    data_list = parquet_dir / "data.list"
    if not data_list.exists():
        raise SystemExit(f"no data.list under {parquet_dir}; parquet packaging never completed")

    referenced = [Path(line.strip()) for line in data_list.read_text().splitlines() if line.strip()]
    shards = [p for p in referenced if p.exists()]
    missing = [p for p in referenced if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)}/{len(referenced)} shards in data.list do not exist on disk")
        for p in missing[:5]:
            print(f"  missing: {p}")
    if not shards:
        raise SystemExit(
            f"none of the {len(referenced)} shards referenced by {data_list} exist. Training would "
            f"silently load zero batches. Re-run `voiceclone remote extract-data`."
        )

    reasons: Counter[str] = Counter()
    kept = 0
    total = 0
    examples: dict[str, list[str]] = {}

    for shard in shards:
        df = pd.read_parquet(shard)
        print(f"{shard.name}: {len(df)} rows, columns={list(df.columns)}")
        for _, row in df.iterrows():
            total += 1
            utt = row["utt"]

            speech, sample_rate = torchaudio.load(BytesIO(row["audio_data"]))
            speech = speech.mean(dim=0, keepdim=True)
            num_frames = speech.size(1) / sample_rate * 100  # filter() counts 10ms frames

            speech_token = row.get("speech_token")
            n_speech_token = 0 if speech_token is None else len(speech_token)
            # text is tokenized later in the real pipeline; character count is a rough
            # proxy, enough to tell "plausible" from "wildly over the limit"
            n_text_chars = len(str(row["text"]))

            reason = None
            if num_frames < DEFAULTS["min_length"]:
                reason = f"too short (<{DEFAULTS['min_length']} frames / 1.0s)"
            elif num_frames > DEFAULTS["max_length"]:
                reason = f"too long (>{DEFAULTS['max_length']} frames / 60s)"
            elif n_speech_token == 0:
                reason = "empty speech_token (extractor skips audio >30s)"

            if reason:
                reasons[reason] += 1
                examples.setdefault(reason, []).append(
                    f"{utt} ({num_frames / 100:.1f}s, {n_speech_token} tokens, {n_text_chars} chars)"
                )
            else:
                kept += 1

    print(f"\n{args.split}: {total} utterances, {kept} survive the length/token filters")
    if reasons:
        print("\ndropped:")
        for reason, count in reasons.most_common():
            print(f"  {count:4d}  {reason}")
            for ex in examples[reason][:3]:
                print(f"          e.g. {ex}")
    if kept == 0:
        print(
            "\nNothing survives. An empty dataloader is what makes CosyVoice3's "
            "executor.cv() raise KeyError: 'tag'."
        )


if __name__ == "__main__":
    main()
