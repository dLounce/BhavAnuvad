"""Tests for bhav.asr.transcribe. Backends are mocked; no models are loaded."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from bhav.asr.transcribe import SUPPORTED_LANGS, Transcript, asr_sr, transcribe


@pytest.fixture
def clip(tmp_path: Path, tone) -> Path:
    """One second sine at the ASR rate, written through the real io path."""
    from bhav.audio.io import save_audio

    sr = asr_sr()
    path = tmp_path / "clip.wav"
    save_audio(path, tone(sr=sr, dur=1.0), sr=sr)
    return path


def test_returns_transcript_from_primary(clip: Path) -> None:
    with patch("bhav.asr.transcribe._transcribe_indicconformer", return_value="नमस्ते") as primary:
        result = transcribe(clip, "hi")

    assert isinstance(result, Transcript)
    assert result.text == "नमस्ते"
    assert result.backend == "indicconformer"
    assert result.lang == "hi"
    assert result.audio_path == clip
    assert result.duration_s == pytest.approx(1.0, abs=0.01)
    assert primary.call_args.args[1] == "hi"
    assert primary.call_args.args[2] == "rnnt"


def test_decoder_argument_is_forwarded(clip: Path) -> None:
    with patch("bhav.asr.transcribe._transcribe_indicconformer", return_value="x") as primary:
        transcribe(clip, "ta", decoder="ctc")

    assert primary.call_args.args[2] == "ctc"


def test_falls_back_to_whisper_on_primary_failure(clip: Path) -> None:
    with (
        patch("bhav.asr.transcribe._transcribe_indicconformer", side_effect=RuntimeError("boom")),
        patch("bhav.asr.transcribe._transcribe_whisper", return_value="வணக்கம்") as fallback,
    ):
        result = transcribe(clip, "ta")

    assert result.text == "வணக்கம்"
    assert result.backend == "whisper"
    fallback.assert_called_once()


def test_fallback_disabled_reraises(clip: Path) -> None:
    with (
        patch("bhav.asr.transcribe._transcribe_indicconformer", side_effect=RuntimeError("boom")),
        patch("bhav.asr.transcribe._transcribe_whisper") as fallback,
        pytest.raises(RuntimeError, match="boom"),
    ):
        transcribe(clip, "hi", allow_fallback=False)

    fallback.assert_not_called()


@pytest.mark.parametrize("lang", ["en", "HI", "", "hi-IN"])
def test_rejects_unsupported_lang(clip: Path, lang: str) -> None:
    with pytest.raises(ValueError, match="unsupported lang"):
        transcribe(clip, lang)


def test_lang_rejected_before_audio_is_touched() -> None:
    with pytest.raises(ValueError, match="unsupported lang"):
        transcribe(Path("does_not_exist.wav"), "en")


def test_transcript_is_frozen() -> None:
    t = Transcript(text="a", lang="hi", backend="indicconformer")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.text = "b"  # type: ignore[misc]


def test_supported_langs_matches_config() -> None:
    from bhav.asr.transcribe import _config

    cfg = _config()
    assert set(cfg["langs"]["source"]) == set(SUPPORTED_LANGS)
    assert set(cfg["langs"]["target"]) == set(SUPPORTED_LANGS)
