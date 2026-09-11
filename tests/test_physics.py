from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bit2watt_impl.physics.simulation import run_to
from bit2watt_impl.physics.waveforms import build_event_schedule, sample_relative_waveform
from dataset.attack_profiles import PHASE1_PROFILES, sweep_profile
from pipeline.physics_correlation import (
    correlation_report,
    join_cyber_physics,
    rollup_cyber_windows,
)
from pipeline.run_pipeline import anomaly_score, anomaly_score_components


class FakeTDS:
    def __init__(self, parent, outcomes):
        self.parent = parent
        self.outcomes = list(outcomes)
        self.config = SimpleNamespace(tf=0.0)
        self.busted = False

    def run(self):
        ok, busted = self.outcomes.pop(0) if self.outcomes else (True, False)
        self.busted = busted
        if ok:
            self.parent.dae.ts.t.append(self.config.tf)
            self.parent.dae.t = self.config.tf
        return ok


class FakeSystem:
    def __init__(self, outcomes):
        self.dae = SimpleNamespace(ts=SimpleNamespace(t=[]), t=0.0)
        self.TDS = FakeTDS(self, outcomes)


def test_waveform_schedule_is_bounded_and_resets():
    profile = sweep_profile(1.45, kind="square")
    events = build_event_schedule(profile)
    assert events[-1].relative_delta == 0.0
    assert events[-1].time_s == profile.t_end_s
    assert min(np.diff([event.time_s for event in events])) >= 0.1 - 1e-9


def test_sine_sweep_uses_twenty_steps_per_cycle_and_resets():
    profile = sweep_profile(0.5, amplitude_frac=0.012)
    events = build_event_schedule(profile)
    first_cycle = [
        event
        for event in events
        if event.time_s < profile.t_start_s + 1.0 / profile.frequency_hz
    ]
    assert profile.kind == "sine"
    assert len(first_cycle) == 20
    assert max(abs(event.relative_delta) for event in events) <= 0.012 + 1e-12
    assert events[-1].time_s == profile.t_end_s
    assert events[-1].relative_delta == 0.0


def test_high_frequency_sine_clamps_to_safe_event_spacing():
    profile = sweep_profile(1.5, amplitude_frac=0.012)
    events = build_event_schedule(profile)
    spacing = np.diff([event.time_s for event in events])
    assert spacing.min() >= 0.1 - 1e-9
    first_cycle = [
        event
        for event in events
        if event.time_s < profile.t_start_s + 1.0 / profile.frequency_hz
    ]
    assert len(first_cycle) < 20


def test_sampled_profiles_are_deterministic():
    first = sample_relative_waveform(PHASE1_PROFILES["ramp"])[1]
    second = sample_relative_waveform(PHASE1_PROFILES["ramp"])[1]
    np.testing.assert_array_equal(first, second)


def test_run_to_halves_chunk_after_recoverable_failure():
    ss = FakeSystem([(False, False), (True, False), (True, False)])
    ok, records, reason = run_to(ss, 0.5, max_chunk_s=0.5)
    assert ok and not reason
    assert records[1].chunk_s == 0.25
    assert records[-1].reached_s == 0.5


def test_run_to_stops_immediately_when_busted():
    ss = FakeSystem([(False, True)])
    ok, records, reason = run_to(ss, 1.0)
    assert not ok
    assert records[-1].busted
    assert "fresh replay" in reason


def test_cyber_rollup_and_physics_join(tmp_path):
    cyber = [
        {
            "attack_id": "a",
            "session_id": "s",
            "anomaly_score": score,
            "threat_id": "swma",
            "rag_top": [["swma", 0.8]],
            "features": {"swing_ratio": score},
        }
        for score in (0.2, 0.9)
    ]
    rolled = rollup_cyber_windows(cyber)
    assert rolled.loc[0, "cyber_anomaly_score_max"] == 0.9
    physics = pd.DataFrame(
        [
            {
                "attack_id": "a",
                "test_system": "kundur",
                "converged": True,
                "osc_std": 0.01,
                "rocof_hz_s": 0.1,
            },
            {
                "attack_id": "b",
                "test_system": "kundur",
                "converged": False,
                "osc_std": np.nan,
                "rocof_hz_s": np.nan,
            },
        ]
    )
    joined, unmatched = join_cyber_physics(rolled, physics)
    assert len(joined) == 2
    assert unmatched["attack_id"].tolist() == ["b"]
    report = correlation_report(joined)
    assert report["usable_rows"] == 1

    path = tmp_path / "cyber.json"
    path.write_text(json.dumps(cyber))
    assert path.exists()


def test_anomaly_score_is_bounded_and_transparent():
    components = anomaly_score_components(
        {"mean_w": 300, "swing_ratio": 3, "high_load_fraction": 1},
        z_score=9,
        baseline_mean_w=120,
    )
    assert all(0 <= value <= 1 for value in components.values())
    assert anomaly_score(components) == 1.0
