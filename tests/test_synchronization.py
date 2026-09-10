from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.synchronization import pair_synchronization, synchronization_features


def test_identical_series_have_high_synchronization():
    t = np.arange(500) / 10
    signal = np.sin(2 * np.pi * 0.5 * t)
    result = pair_synchronization(signal, signal, sample_hz=10)
    assert result["synchronization_index"] > 0.95
    assert abs(result["lag_seconds"]) < 1e-9


def test_long_format_multi_gpu_alignment():
    t = np.arange(500) / 10
    signal = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)
    frame = pd.concat(
        [
            pd.DataFrame({"timestamp": t, "gpu_id": 0, "power_w": signal}),
            pd.DataFrame({"timestamp": t + 0.01, "gpu_id": 1, "power_w": signal}),
        ],
        ignore_index=True,
    )
    result = synchronization_features(frame)
    assert result[0]["gpu_pair"] == [0, 1]
    assert result[0]["synchronization_index"] > 0.9

