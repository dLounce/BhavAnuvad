"""ASR wrapper. IndicConformer primary, Whisper fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bhav.audio.io import duration_s as _duration_s
from bhav.audio.io import load_audio

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = frozenset({"hi", "ta"})
_DECODERS = frozenset({"rnnt", "ctc"})
_WHISPER_LANG = {"hi": "hindi", "ta": "tamil"}

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


@dataclass(frozen=True)
class Transcript:
    """Output of a single ASR call.

    text is raw model output, not normalized. Callers pass it through
    bhav.text.normalize before scoring or feeding MT.
    """

    text: str
    lang: str  # "hi" or "ta"
    backend: str  # "indicconformer" or "whisper"
    audio_path: Path | None = None
    duration_s: float | None = None


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def asr_sr() -> int:
    return int(_config()["audio"]["sr_asr"])


def _device() -> str:
    import torch

    want = _config().get("device", "cpu")
    return want if (want != "cuda" or torch.cuda.is_available()) else "cpu"


@lru_cache(maxsize=1)
def _load_indicconformer() -> tuple[Any, str]:
    """Multilingual — one instance covers hi and ta. Lang is a decode-time arg."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(_config()["models"]["asr"], trust_remote_code=True)
    model.eval()
    device = _device()
    return model.to(device), device


@lru_cache(maxsize=1)
def _load_whisper() -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_id = _config()["models"]["asr_fallback"]
    device = _device()
    dtype = (
        torch.float16
        if (device == "cuda" and _config().get("dtype") == "float16")
        else torch.float32
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    return model.to(device), processor, device


def _transcribe_indicconformer(
    wav: np.ndarray,
    lang: str,
    decoder: str = "rnnt",
) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}, expected one of {sorted(SUPPORTED_LANGS)}")
    if decoder not in _DECODERS:
        raise ValueError(f"unsupported decoder {decoder!r}, expected one of {sorted(_DECODERS)}")

    import torch

    model, device = _load_indicconformer()
    tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(tensor, lang, decoder)

    if isinstance(out, list | tuple):
        out = out[0]
    return str(out).strip()


def _transcribe_whisper(wav: np.ndarray, lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}, expected one of {sorted(SUPPORTED_LANGS)}")

    import torch

    model, processor, device = _load_whisper()

    features = processor(
        wav,
        sampling_rate=asr_sr(),
        return_tensors="pt",
    ).input_features.to(device, dtype=model.dtype)

    forced = processor.get_decoder_prompt_ids(
        language=_WHISPER_LANG[lang],
        task="transcribe",
    )

    with torch.inference_mode():
        ids = model.generate(features, forced_decoder_ids=forced, max_new_tokens=440)

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe(
    audio_path: str | Path,
    lang: str,
    decoder: str = "rnnt",
    allow_fallback: bool = True,
) -> Transcript:
    """Transcribe one clip. IndicConformer primary, Whisper on failure.

    Raises if IndicConformer fails and allow_fallback is False.
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}, expected one of {sorted(SUPPORTED_LANGS)}")

    audio_path = Path(audio_path)
    wav, _ = load_audio(audio_path, expected_sr=asr_sr())
    clip_duration = _duration_s(audio_path)

    try:
        text = _transcribe_indicconformer(wav, lang, decoder)
        backend = "indicconformer"
    except Exception:
        if not allow_fallback:
            raise
        logger.warning(
            "indicconformer failed on %s, falling back to whisper", audio_path, exc_info=True
        )
        text = _transcribe_whisper(wav, lang)
        backend = "whisper"

    return Transcript(
        text=text,
        lang=lang,
        backend=backend,
        audio_path=audio_path,
        duration_s=clip_duration,
    )
