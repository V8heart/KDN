"""Shared cyber/physics waveform specifications.

The amplitude is a dimensionless fraction.  It is applied independently to a
GPU-side reference power and to a selected public test-system PQ load; it is
not a conversion from GPU watts to grid MW.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttackProfile:
    attack_id: str
    label: str
    kind: str
    frequency_hz: float | None
    amplitude_frac: float
    duty_cycle: float = 0.5
    duration_s: float = 14.0
    t_start_s: float = 1.0
    seed: int = 42
    cyber_amplitude_frac: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "normal",
            "square",
            "sine",
            "burst",
            "ramp",
            "irregular",
            "steady",
        }:
            raise ValueError(f"unsupported waveform kind: {self.kind}")
        if self.frequency_hz is not None and self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if not 0 <= self.amplitude_frac < 1:
            raise ValueError("amplitude_frac must be in [0, 1)")
        if self.cyber_amplitude_frac is not None and self.cyber_amplitude_frac < 0:
            raise ValueError("cyber_amplitude_frac must be non-negative")
        if not 0 < self.duty_cycle < 1:
            raise ValueError("duty_cycle must be in (0, 1)")
        if self.duration_s <= 0 or self.t_start_s < 0:
            raise ValueError("invalid timing")

    @property
    def t_end_s(self) -> float:
        return self.t_start_s + self.duration_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PHASE1_PROFILES: dict[str, AttackProfile] = {
    "normal": AttackProfile(
        "normal-control", "normal_baseline", "normal", 0.05, 0.0
    ),
    "periodic": AttackProfile(
        "swma-periodic-f0.60-a0.10", "swma", "square", 0.60, 0.10
    ),
    "burst": AttackProfile(
        "swma-burst-f0.60-a0.30", "swma", "burst", 0.60, 0.30
    ),
    "ramp": AttackProfile(
        "ltma-ramp-a0.15", "ltma", "ramp", None, 0.15
    ),
}

CYBER_ATTACK_PROFILES: dict[str, AttackProfile] = {
    "swma": AttackProfile(
        "swma-periodic-primary", "swma", "square", 0.50, 0.10,
        cyber_amplitude_frac=0.79,
    ),
    "ltma": AttackProfile(
        "ltma-irregular-primary", "ltma", "irregular", None, 0.10,
        seed=201, cyber_amplitude_frac=0.55,
    ),
    "cryptojacking": AttackProfile(
        "cryptojacking-steady-primary", "cryptojacking", "steady", None, 0.10,
        cyber_amplitude_frac=0.58,
    ),
}

ALL_PHYSICS_PROFILES: dict[str, AttackProfile] = {
    **PHASE1_PROFILES,
    **{f"cyber_{name}": profile for name, profile in CYBER_ATTACK_PROFILES.items()},
}


def sweep_profile(
    frequency_hz: float,
    *,
    amplitude_frac: float = 0.10,
    duration_s: float = 14.0,
    kind: str = "sine",
) -> AttackProfile:
    """Return a deterministic periodic profile for a frequency sweep."""
    return AttackProfile(
        attack_id=(
            f"swma-sweep-{kind}-f{frequency_hz:.2f}-a{amplitude_frac:.3f}"
        ),
        label="swma",
        kind=kind,
        frequency_hz=float(frequency_hz),
        amplitude_frac=float(amplitude_frac),
        duration_s=float(duration_s),
    )
