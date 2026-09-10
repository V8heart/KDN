"""정상인데 전력 변동이 큰 다섯 가지 hard-negative 합성기."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.attack_profiles import PHASE1_PROFILES
from dataset.synth_common import BASELINE_POWER_W, build_frame, rng_for, smooth_noise


def _piecewise(
    rng: np.random.Generator,
    n: int,
    levels: tuple[float, ...],
    min_len: int,
    max_len: int,
) -> np.ndarray:
    values: list[float] = []
    previous = -1
    while len(values) < n:
        choices = [i for i in range(len(levels)) if i != previous]
        idx = int(rng.choice(choices))
        previous = idx
        values.extend([levels[idx]] * int(rng.integers(min_len, max_len + 1)))
    return np.asarray(values[:n], dtype=float)


def normal_distributed_training(n: int = 1200, sample_hz: float = 10, seed: int = 101) -> pd.DataFrame:
    """compute/all-reduce 위상이 불규칙하게 전환되는 분산학습 근사."""
    rng = rng_for(seed)
    power = _piecewise(rng, n, (205, 145, 185), 13, 61)
    power += smooth_noise(rng, n, 12, width=7)
    util = np.clip((power - 85) / 1.35 + rng.normal(0, 5, n), 5, 100)
    return build_frame(
        power,
        util_gpu_pct=util,
        label="normal_distributed_training",
        session_id=f"syn-normal-ddp-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        gres_req="gpu:2",
    )


def normal_hpo_search(n: int = 1200, sample_hz: float = 10, seed: int = 102) -> pd.DataFrame:
    """서로 다른 강도의 짧은 학습 trial과 준비 구간이 반복되는 HPO 근사."""
    rng = rng_for(seed)
    power = np.full(n, 35.0)
    cursor = 0
    while cursor < n:
        idle = int(rng.integers(8, 35))
        run = int(rng.integers(35, 125))
        cursor += idle
        end = min(n, cursor + run)
        power[cursor:end] = float(rng.uniform(135, 260))
        cursor = end
    power += smooth_noise(rng, n, 8)
    return build_frame(
        power,
        label="normal_hpo_search",
        session_id=f"syn-normal-hpo-{seed}",
        sample_hz=sample_hz,
        seed=seed,
    )


def normal_checkpoint(n: int = 1200, sample_hz: float = 10, seed: int = 103) -> pd.DataFrame:
    """지속 학습 중 체크포인트 저장 때 짧게 전력이 떨어지는 패턴."""
    rng = rng_for(seed)
    power = 190 + smooth_noise(rng, n, 10)
    cursor = int(rng.integers(120, 180))
    while cursor < n:
        width = int(rng.integers(8, 24))
        power[cursor : cursor + width] = 55 + rng.normal(0, 5, min(width, n - cursor))
        cursor += int(rng.integers(150, 260))
    return build_frame(
        power,
        label="normal_checkpoint",
        session_id=f"syn-normal-checkpoint-{seed}",
        sample_hz=sample_hz,
        seed=seed,
    )


def normal_dataloader_stall(n: int = 1200, sample_hz: float = 10, seed: int = 104) -> pd.DataFrame:
    """불규칙한 I/O 대기로 인한 비주기적 전력 하락."""
    rng = rng_for(seed)
    power = 175 + smooth_noise(rng, n, 15, width=5)
    for start in rng.choice(np.arange(40, n - 40), size=18, replace=False):
        width = int(rng.integers(4, 35))
        power[start : start + width] = 45 + rng.normal(0, 7, min(width, n - start))
    return build_frame(
        power,
        label="normal_dataloader_stall",
        session_id=f"syn-normal-dataloader-{seed}",
        sample_hz=sample_hz,
        seed=seed,
    )


def normal_eval_train_switch(n: int = 1200, sample_hz: float = 10, seed: int = 105) -> pd.DataFrame:
    """학습과 평가가 서로 다른 길이·전력 수준으로 교차하는 패턴."""
    rng = rng_for(seed)
    power = _piecewise(rng, n, (205, 105), 45, 180)
    power += smooth_noise(rng, n, 9)
    return build_frame(
        power,
        label="normal_eval_train_switch",
        session_id=f"syn-normal-eval-{seed}",
        sample_hz=sample_hz,
        seed=seed,
    )


def baseline_idle(n: int = 1200, sample_hz: float = 10, seed: int = 100) -> pd.DataFrame:
    """임계값 캘리브레이션을 위한 비교적 안정적인 정상 기준선."""
    rng = rng_for(seed)
    profile = PHASE1_PROFILES["normal"]
    power = BASELINE_POWER_W + smooth_noise(rng, n, 6)
    return build_frame(
        power,
        label="normal_baseline",
        session_id=f"syn-normal-baseline-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    )


NORMAL_GENERATORS = {
    "normal_baseline": baseline_idle,
    "normal_distributed_training": normal_distributed_training,
    "normal_hpo_search": normal_hpo_search,
    "normal_checkpoint": normal_checkpoint,
    "normal_dataloader_stall": normal_dataloader_stall,
    "normal_eval_train_switch": normal_eval_train_switch,
}

