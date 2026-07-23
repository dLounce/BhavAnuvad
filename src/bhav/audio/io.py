"""Single chokepoint for all audio I/O. Nothing else in the codebase touches disk audio.

Every load declares the sample rate it expects. Mismatches either resample here
(explicitly requested) or hard-fail, so SR bugs cannot propagate silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import soxr

logger = logging.getLogger(__name__)

SR_ASR = 16_000
SR_TTS = 24_000
SR_DEMUCS = 44_100
TARGET_LUFS = -23.0


class AudioSpecError(ValueError):
    """Raised when audio does not match the sample rate / shape contract."""


def load_audio(
    path: str | Path,
    expected_sr: int,
    *,
    resample: bool = False,
    mono: bool = True,
    normalize_loudness: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio as float32 in [-1, 1].

    Fails loudly on sample-rate mismatch unless `resample=True`, which is the only
    sanctioned resampling path in the project.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.T

    if mono and wav.shape[0] > 1:
        wav = wav.mean(axis=0, keepdims=True)
    wav = wav[0] if mono else wav

    if sr != expected_sr:
        if not resample:
            raise AudioSpecError(f"{path.name}: sr={sr}, expected {expected_sr}")
        wav = soxr.resample(wav, sr, expected_sr, quality="HQ")
        sr = expected_sr

    if normalize_loudness:
        wav = normalize_lufs(wav, sr)

    return np.ascontiguousarray(wav, dtype=np.float32), sr


def save_audio(
    path: str | Path,
    wav: np.ndarray,
    sr: int,
    *,
    expected_sr: int | None = None,
    subtype: str = "PCM_16",
) -> Path:
    """Write mono float32 audio, asserting the SR contract before it hits disk."""
    if expected_sr is not None and sr != expected_sr:
        raise AudioSpecError(f"refusing to write sr={sr}, expected {expected_sr}")

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise AudioSpecError(f"expected mono 1-D array, got shape {wav.shape}")

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        logger.warning("clipping: peak %.3f, rescaling", peak)
        wav = wav / peak

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr, subtype=subtype)
    return path


def normalize_lufs(wav: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    """Loudness-normalize to `target` LUFS. Silent or too-short clips pass through."""
    if wav.size < int(0.4 * sr):
        return wav
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    if not np.isfinite(loudness):
        return wav
    return pyln.normalize.loudness(wav, loudness, target).astype(np.float32)


def duration_s(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
