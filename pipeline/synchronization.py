"""다중 GPU 시계열의 정렬·상관·coherence 피처."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import coherence


def align_gpu_series(
    df: pd.DataFrame,
    *,
    value_col: str = "power_w",
    sample_hz: float | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    """timestamp 공통 구간에 GPU별 시계열을 선형 보간한다."""
    required = {"timestamp", "gpu_id", value_col}
    if not required.issubset(df.columns):
        raise ValueError(f"동기화 분석 필수 컬럼: {sorted(required)}")
    groups = {
        int(gpu): group.sort_values("timestamp")
        for gpu, group in df.groupby("gpu_id")
    }
    if len(groups) < 2:
        raise ValueError("동기화 분석에는 GPU가 2개 이상 필요합니다.")

    starts = [pd.to_numeric(g["timestamp"], errors="coerce").min() for g in groups.values()]
    ends = [pd.to_numeric(g["timestamp"], errors="coerce").max() for g in groups.values()]
    start, stop = max(starts), min(ends)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("GPU 시계열의 공통 시간 구간이 없습니다.")

    if sample_hz is None:
        candidates = []
        for group in groups.values():
            ts = pd.to_numeric(group["timestamp"], errors="coerce").dropna().to_numpy()
            delta = np.diff(ts)
            delta = delta[delta > 0]
            if len(delta):
                candidates.append(1.0 / np.median(delta))
        sample_hz = min(candidates) if candidates else 1.0
    timeline = np.arange(start, stop, 1.0 / sample_hz)
    aligned: dict[int, np.ndarray] = {}
    for gpu, group in groups.items():
        ts = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy()
        mask = np.isfinite(ts) & np.isfinite(values)
        aligned[gpu] = np.interp(timeline, ts[mask], values[mask])
    return timeline, aligned, float(sample_hz)


def pair_synchronization(x: np.ndarray, y: np.ndarray, sample_hz: float) -> dict[str, float]:
    """두 시계열의 zero-lag 상관, 최대 상관, coherence를 계산한다."""
    if len(x) != len(y) or len(x) < 16:
        raise ValueError("길이가 같은 16개 이상의 샘플이 필요합니다.")
    xz, yz = x - np.mean(x), y - np.mean(y)
    denom = np.std(xz) * np.std(yz)
    zero_corr = float(np.mean(xz * yz) / denom) if denom > 0 else 0.0
    corr = np.correlate(xz, yz, mode="full")
    corr_denom = np.linalg.norm(xz) * np.linalg.norm(yz)
    corr = corr / corr_denom if corr_denom > 0 else corr
    best = int(np.argmax(np.abs(corr)))
    lag_samples = best - (len(x) - 1)
    max_corr = float(np.abs(corr[best]))
    _, coh = coherence(x, y, fs=sample_hz, nperseg=min(256, len(x)))
    mean_coherence = float(np.nanmean(coh)) if len(coh) else 0.0
    sync_index = float(np.clip((abs(zero_corr) + max_corr + mean_coherence) / 3.0, 0, 1))
    return {
        "zero_lag_correlation": zero_corr,
        "max_cross_correlation": max_corr,
        "lag_seconds": lag_samples / sample_hz,
        "mean_coherence": mean_coherence,
        "synchronization_index": sync_index,
    }


def synchronization_features(df: pd.DataFrame, value_col: str = "power_w") -> list[dict]:
    """세션 내부 모든 GPU 쌍의 동기화 피처를 반환한다."""
    _, aligned, hz = align_gpu_series(df, value_col=value_col)
    results = []
    for left, right in itertools.combinations(sorted(aligned), 2):
        results.append({
            "gpu_pair": [left, right],
            "sample_hz": hz,
            **pair_synchronization(aligned[left], aligned[right], hz),
        })
    return results

