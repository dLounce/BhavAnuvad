import numpy as np
import pytest


@pytest.fixture
def tone():
    def _make(freq=220.0, sr=16000, dur=1.0, amp=0.5):
        t = np.arange(int(sr * dur)) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return _make
