"""Objective TTS quality metrics: WER (intelligibility), speaker similarity
(does it sound like the target speaker), and UTMOS (predicted naturalness MOS).

Every function takes plain paths/strings and returns a float — no CosyVoice3-specific
knowledge lives here, so this module works unchanged for comparing across the
Fish-Speech/F5-TTS/CosyVoice3 options a reader of the writeup might reasonably ask
"why not just take your word for it" about.
"""

from __future__ import annotations

import typing
from functools import lru_cache
from pathlib import Path

from voiceclone.data.transcribe import transcribe
from voiceclone.logging_utils import get_logger

if typing.TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def word_error_rate(
    reference_text: str,
    generated_audio_path: Path,
    whisper_model_size: str = "large-v3",
    device: str = "cpu",
) -> float:
    """WER between the intended text and what an independent ASR model hears in the
    generated audio. Using ASR (rather than trusting the TTS model's own alignment) is
    the point — it's an external, model-agnostic intelligibility check.
    """
    import jiwer

    hypothesis = transcribe(generated_audio_path, model_size=whisper_model_size, device=device)
    return jiwer.wer(reference_text, hypothesis)


@lru_cache(maxsize=1)
def _wavlm_sv_model(device: str):
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    logger.info("loading microsoft/wavlm-base-plus-sv on %s", device)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    return extractor, model


def _load_mono_16k(path: Path) -> torch.Tensor:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav.squeeze(0)


def speaker_embedding(audio_path: Path, device: str = "cpu") -> torch.Tensor:
    import torch

    extractor, model = _wavlm_sv_model(device)
    wav = _load_mono_16k(audio_path)
    inputs = extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).embeddings.squeeze(0)


def speaker_similarity(audio_a: Path, audio_b: Path, device: str = "cpu") -> float:
    """Cosine similarity between WavLM x-vector speaker embeddings, in [-1, 1] (in
    practice ~[0, 1] for same-vs-different speakers). This is the plan's primary
    metric — the one CosyVoice3 was picked for in the first place.
    """
    import torch

    emb_a = speaker_embedding(audio_a, device)
    emb_b = speaker_embedding(audio_b, device)
    return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()


@lru_cache(maxsize=1)
def _utmos_predictor(device: str):
    import torch

    logger.info("loading UTMOS (SpeechMOS utmos22_strong) on %s", device)
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    return predictor.to(device)


def naturalness_mos(audio_path: Path, device: str = "cpu") -> float:
    """Reference-free predicted naturalness MOS (1-5 scale). No ground truth needed —
    this is what lets us score conversational/spontaneous generations that have no
    "correct" reference recording to compare against.
    """
    import torch

    predictor = _utmos_predictor(device)
    wav = _load_mono_16k(audio_path)
    with torch.no_grad():
        score = predictor(wav.unsqueeze(0).to(device), sr=16000)
    return float(score.item() if hasattr(score, "item") else score)
