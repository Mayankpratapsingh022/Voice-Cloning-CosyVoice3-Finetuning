"""Thin, dependency-isolated wrapper around CosyVoice3's `AutoModel` inference class.

Kept separate from the serving/CLI layers so it can be unit-tested (with the CosyVoice
import mocked) without needing the full environment, and so the serving and batch-eval
code paths (evaluation/run_eval.py) share one implementation of "load a checkpoint,
enroll a voice, synthesize text" instead of two.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

from voiceclone.config import cosyvoice_pythonpath

if typing.TYPE_CHECKING:
    import torch

DEFAULT_SPK_ID = "voiceclone_default"


def _ensure_cosyvoice_importable() -> None:
    """Put CosyVoice3's cloned repo on sys.path for this process.

    Subprocess call sites get this via cosyvoice_subprocess_env()'s PYTHONPATH, but
    this module imports `cosyvoice` in-process, so it needs the same entries added
    directly. Same underlying reason: CosyVoice3 is a cloned repo, not an installed
    package.
    """
    for entry in reversed(cosyvoice_pythonpath()):
        if entry not in sys.path:
            sys.path.insert(0, entry)


class VoiceCloneEngine:
    """Loads one fine-tuned CosyVoice3 checkpoint and exposes enroll/synthesize.

    A single reference clip is "enrolled" once (`add_zero_shot_spk`) and then reused
    by id for every subsequent synthesis call — the pattern the upstream examples use
    to avoid re-processing the prompt audio on every request.
    """

    def __init__(self, model_dir: str):
        _ensure_cosyvoice_importable()
        from cosyvoice.cli.cosyvoice import AutoModel

        self.model = AutoModel(model_dir=model_dir)
        self._enrolled_spk_ids: set[str] = set()

    def enroll(self, prompt_text: str, prompt_wav_path: str, spk_id: str = DEFAULT_SPK_ID) -> str:
        ok = self.model.add_zero_shot_spk(prompt_text, prompt_wav_path, spk_id)
        if not ok:
            raise RuntimeError(f"failed to enroll speaker prompt as '{spk_id}'")
        self._enrolled_spk_ids.add(spk_id)
        return spk_id

    def synthesize(self, text: str, spk_id: str = DEFAULT_SPK_ID, speed: float = 1.0) -> tuple[torch.Tensor, int]:
        import torch

        if spk_id not in self._enrolled_spk_ids:
            raise ValueError(f"'{spk_id}' hasn't been enrolled — call .enroll(...) first")

        chunks = [
            output["tts_speech"]
            for output in self.model.inference_zero_shot(text, "", "", zero_shot_spk_id=spk_id, speed=speed)
        ]
        return torch.cat(chunks, dim=1), self.model.sample_rate

    def synthesize_to_file(self, text: str, out_path: Path, spk_id: str = DEFAULT_SPK_ID, speed: float = 1.0) -> Path:
        import torchaudio

        audio, sample_rate = self.synthesize(text, spk_id, speed)
        torchaudio.save(str(out_path), audio, sample_rate)
        return out_path
