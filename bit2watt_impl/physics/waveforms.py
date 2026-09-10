"""Event schedules shared by Kundur and WECC PQ-load injections."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset.attack_profiles import AttackProfile


@dataclass(frozen=True)
class InjectionEvent:
    time_s: float
    relative_delta: float


def build_event_schedule(profile: AttackProfile, *, min_interval_s: float = 0.1) -> list[InjectionEvent]:
    """Convert a profile into sparse Ppf events.

    Event spacing is intentionally bounded to avoid high-rate discontinuities
    that are numerically fragile in ANDES.
    """
    start, end = profile.t_start_s, profile.t_end_s
    amp = profile.amplitude_frac
    events: list[InjectionEvent] = []

    if profile.kind in {"normal", "square", "burst"}:
        if profile.frequency_hz is None:
            raise ValueError("periodic profiles require frequency_hz")
        half_period = 1.0 / (2.0 * profile.frequency_hz)
        if half_period < min_interval_s:
            half_period = min_interval_s
        active_start, active_end = start, end
        if profile.kind == "burst":
            active_start = start + 0.30 * profile.duration_s
            active_end = start + 0.70 * profile.duration_s
            events.append(InjectionEvent(start, 0.0))
        sign = 1.0
        t = active_start
        while t < active_end - 1e-9:
            events.append(InjectionEvent(t, sign * amp))
            sign *= -1.0
            t += half_period
    elif profile.kind == "ramp":
        ramp_interval_s = max(0.5, min_interval_s)
        count = max(2, int(np.floor(profile.duration_s / ramp_interval_s)) + 1)
        for t, value in zip(
            np.linspace(start, end, count, endpoint=False),
            np.linspace(-amp, amp, count, endpoint=True),
        ):
            events.append(InjectionEvent(float(t), float(value)))
    elif profile.kind == "irregular":
        rng = np.random.default_rng(profile.seed)
        t = start
        while t < end - 1e-9:
            events.append(InjectionEvent(t, float(rng.uniform(-amp, amp))))
            t += float(rng.uniform(max(0.5, min_interval_s), 2.0))
    elif profile.kind == "steady":
        events.append(InjectionEvent(start, amp))
    else:
        raise ValueError(f"unsupported profile kind: {profile.kind}")

    events.append(InjectionEvent(end, 0.0))
    events.sort(key=lambda event: event.time_s)
    deduplicated: list[InjectionEvent] = []
    for event in events:
        if deduplicated and abs(deduplicated[-1].time_s - event.time_s) < 1e-9:
            deduplicated[-1] = event
        else:
            deduplicated.append(event)
    return deduplicated


def sample_relative_waveform(
    profile: AttackProfile,
    *,
    sample_hz: float = 10.0,
    post_settle_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the piecewise-constant event schedule for cyber-side features."""
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    stop = profile.t_end_s + post_settle_s
    times = np.arange(0.0, stop, 1.0 / sample_hz)
    values = np.zeros_like(times)
    current = 0.0
    cursor = 0
    events = build_event_schedule(profile)
    for i, time_s in enumerate(times):
        while cursor < len(events) and events[cursor].time_s <= time_s + 1e-12:
            current = events[cursor].relative_delta
            cursor += 1
        values[i] = current
    return times, values
