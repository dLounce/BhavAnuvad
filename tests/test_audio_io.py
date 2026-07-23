import numpy as np
import pytest
import soundfile as sf

from bhav.audio.io import AudioSpecError, duration_s, load_audio, normalize_lufs, save_audio


def test_roundtrip_preserves_shape_and_sr(tmp_path, tone):
    wav = tone(sr=16000, dur=0.5)
    p = save_audio(tmp_path / "a.wav", wav, 16000, expected_sr=16000)
    out, sr = load_audio(p, expected_sr=16000)
    assert sr == 16000
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.allclose(out, wav, atol=1e-3)


def test_sr_mismatch_raises(tmp_path, tone):
    save_audio(tmp_path / "b.wav", tone(sr=16000), 16000)
    with pytest.raises(AudioSpecError):
        load_audio(tmp_path / "b.wav", expected_sr=24000)


def test_resample_when_requested(tmp_path, tone):
    save_audio(tmp_path / "c.wav", tone(sr=16000, dur=1.0), 16000)
    out, sr = load_audio(tmp_path / "c.wav", expected_sr=24000, resample=True)
    assert sr == 24000
    assert abs(len(out) - 24000) < 100


def test_stereo_collapses_to_mono(tmp_path, tone):
    stereo = np.stack([tone(), tone(freq=440.0)], axis=1)
    sf.write(tmp_path / "d.wav", stereo, 16000)
    out, _ = load_audio(tmp_path / "d.wav", expected_sr=16000)
    assert out.ndim == 1


def test_save_rejects_non_mono(tmp_path):
    with pytest.raises(AudioSpecError):
        save_audio(tmp_path / "e.wav", np.zeros((2, 100), dtype=np.float32), 16000)


def test_save_rejects_wrong_sr(tmp_path, tone):
    with pytest.raises(AudioSpecError):
        save_audio(tmp_path / "f.wav", tone(), 16000, expected_sr=24000)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav", expected_sr=16000)


def test_lufs_normalization_changes_level(tone):
    quiet = tone(amp=0.02, dur=2.0)
    out = normalize_lufs(quiet, 16000)
    assert np.max(np.abs(out)) > np.max(np.abs(quiet))


def test_short_clip_passes_through_lufs(tone):
    short = tone(dur=0.1)
    assert np.array_equal(normalize_lufs(short, 16000), short)


def test_duration(tmp_path, tone):
    p = save_audio(tmp_path / "g.wav", tone(dur=1.5), 16000)
    assert duration_s(p) == pytest.approx(1.5, abs=0.01)
