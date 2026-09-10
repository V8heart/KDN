"""corpus의 정성적 특성에 대응하는 세 가지 공격 합성기."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.attack_profiles import CYBER_ATTACK_PROFILES
from dataset.synth_common import BASELINE_POWER_W, build_frame, rng_for, smooth_noise


def swma(
    n: int = 1200,
    sample_hz: float = 10,
    seed: int = 200,
    frequency_hz: float = 0.5,
    duty_cycle: float = 0.5,
) -> pd.DataFrame:
    """규칙적인 high/low 부하를 갖는 소프트웨어 관측 가능 SWMA-like 신호."""
    if not 0 < duty_cycle < 1:
        raise ValueError("duty_cycle은 0과 1 사이여야 합니다.")
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["swma"]
    phase = (np.arange(n) * frequency_hz / sample_hz) % 1.0
    active = phase < duty_cycle
    power = np.where(active, 330.0, 38.0) + rng.normal(0, 3, n)
    util = np.where(active, 98.0, 1.0) + rng.normal(0, 1, n)
    return build_frame(
        power,
        util_gpu_pct=util,
        label="swma",
        session_id=f"syn-attack-swma-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        job_type="unknown_cuda_workload",
        attack_id=(
            profile.attack_id
            if frequency_hz == profile.frequency_hz and duty_cycle == profile.duty_cycle
            else f"swma-custom-f{frequency_hz:.2f}-d{duty_cycle:.2f}-s{seed}"
        ),
        waveform_kind=profile.kind,
        waveform_frequency_hz=frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=duty_cycle,
    )


def ltma(n: int = 1200, sample_hz: float = 10, seed: int = 201) -> pd.DataFrame:
    """평균은 정상 부근이지만 학습 이벤트에 결합된 불규칙 변조 신호."""
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["ltma"]
    power = BASELINE_POWER_W + smooth_noise(rng, n, 24, width=5)
    cursor = 0
    sign = 1.0
    while cursor < n:
        width = int(rng.integers(7, 75))
        amplitude = float(rng.uniform(28, 85))
        power[cursor : cursor + width] += sign * amplitude
        sign *= -1 if rng.random() > 0.25 else 1
        cursor += width
    power -= np.mean(power) - BASELINE_POWER_W
    power = np.clip(power, 28, 260)
    util = np.clip(45 + (power - BASELINE_POWER_W) / 1.8 + rng.normal(0, 8, n), 0, 100)
    return build_frame(
        power,
        util_gpu_pct=util,
        label="ltma",
        session_id=f"syn-attack-ltma-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        job_type="llm_training",
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    )


def cryptojacking(n: int = 1200, sample_hz: float = 10, seed: int = 202) -> pd.DataFrame:
    """오랫동안 높고 평평하게 유지되는 무단 채굴성 부하 근사."""
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["cryptojacking"]
    power = 285 + smooth_noise(rng, n, 5, width=11)
    util = np.clip(96 + rng.normal(0, 1.2, n), 90, 100)
    return build_frame(
        power,
        util_gpu_pct=util,
        label="cryptojacking",
        session_id=f"syn-attack-crypto-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        job_type="unregistered_hash_benchmark",
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    )


ATTACK_GENERATORS = {
    "swma": swma,
    "ltma": ltma,
    "cryptojacking": cryptojacking,
}

