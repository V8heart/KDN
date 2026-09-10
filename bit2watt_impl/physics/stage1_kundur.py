"""Phase 1: cyber/physics features on the public Kundur test system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bit2watt_impl.physics.simulation import (
    build_kundur_system,
    ensure_parent,
    simulate_profile,
)
from bit2watt_impl.physics.waveforms import sample_relative_waveform
from dataset.attack_profiles import ALL_PHYSICS_PROFILES, AttackProfile
from pipeline.features import compute_window_features


def cyber_features(profile: AttackProfile, *, sample_hz: float = 10.0) -> dict[str, float]:
    """Produce comparable cyber-side features from the normalized profile."""
    _, relative = sample_relative_waveform(profile, sample_hz=sample_hz)
    power = 120.0 * (1.0 + relative)
    features = compute_window_features(power, sample_hz=sample_hz)
    return {f"profile_cyber_{key}": float(value) for key, value in features.items()}


def run_profiles(
    profiles: list[AttackProfile],
    *,
    output: Path,
    convergence_log: Path,
    pq_idx: str | None = None,
    max_chunk_s: float = 0.5,
    criteria: int = 1,
) -> pd.DataFrame:
    rows: list[dict] = []
    logs: list[dict] = []
    for profile in profiles:
        result = simulate_profile(
            build_kundur_system,
            profile,
            test_system="kundur_ieeest",
            model_name="GENROU",
            pq_idx=pq_idx,
            max_chunk_s=max_chunk_s,
            criteria=criteria,
        )
        result.row.update(cyber_features(profile))
        rows.append(result.row)
        logs.extend(result.convergence)
        ensure_parent(output)
        pd.DataFrame(rows).to_csv(output, index=False)
        ensure_parent(convergence_log)
        convergence_log.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in logs),
            encoding="utf-8",
        )
        state = "ok" if result.row["converged"] else f"failed: {result.row['failure_reason']}"
        print(f"[Kundur] {profile.attack_id}: {state}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    convergence_log.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in logs),
        encoding="utf-8",
    )
    print(f"saved: {output}")
    print(f"convergence log: {convergence_log}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/kundur_attack_features.csv"),
    )
    parser.add_argument(
        "--convergence-log",
        type=Path,
        default=Path("dataset/eval/physics_convergence.jsonl"),
    )
    parser.add_argument("--profiles", nargs="+", choices=sorted(ALL_PHYSICS_PROFILES), default=None)
    parser.add_argument("--pq-idx", default=None)
    parser.add_argument("--max-chunk-s", type=float, default=0.5)
    parser.add_argument("--criteria", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    names = args.profiles or list(ALL_PHYSICS_PROFILES)
    run_profiles(
        [ALL_PHYSICS_PROFILES[name] for name in names],
        output=args.output,
        convergence_log=args.convergence_log,
        pq_idx=args.pq_idx,
        max_chunk_s=args.max_chunk_s,
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
