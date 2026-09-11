"""Optional amplitude search for WECC+IBR attack sweeps.

Primary path after the TDS.init / current_time harness fix is a known common
amplitude (0.012 with criteria=0 for 0/30/50%). Use this module only if a
penetration still fails attack TDS at that amplitude.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bit2watt_impl.physics.simulation import (
    build_wecc_system_at_penetration,
    ensure_parent,
    simulate_profile,
)
from dataset.attack_profiles import sweep_profile


def _ok(pen: float, amp: float, *, freq_hz: float, criteria: int) -> bool:
    result = simulate_profile(
        lambda: build_wecc_system_at_penetration(pen),
        sweep_profile(freq_hz, amplitude_frac=amp, duration_s=14.0),
        test_system="wecc_179_gencls_ibr",
        model_name="GENCLS",
        pq_selection="largest_p0",
        criteria=criteria,
        max_rebuilds=1,
    )
    return bool(result.row.get("converged"))


def max_stable_amplitude(
    pen: float,
    *,
    freq_hz: float = 0.5,
    lo: float = 0.001,
    hi: float = 0.02,
    iters: int = 8,
    criteria: int = 0,
) -> dict:
    if not _ok(pen, lo, freq_hz=freq_hz, criteria=criteria):
        return {
            "target_penetration": pen,
            "max_stable_amplitude": None,
            "probe_frequency_hz": freq_hz,
            "criteria": criteria,
            "note": "fails even at lower bound",
        }
    if _ok(pen, hi, freq_hz=freq_hz, criteria=criteria):
        return {
            "target_penetration": pen,
            "max_stable_amplitude": hi,
            "probe_frequency_hz": freq_hz,
            "criteria": criteria,
            "note": "stable at upper bound",
        }
    best = lo
    low, high = lo, hi
    for _ in range(iters):
        mid = 0.5 * (low + high)
        if _ok(pen, mid, freq_hz=freq_hz, criteria=criteria):
            best = mid
            low = mid
        else:
            high = mid
    return {
        "target_penetration": pen,
        "max_stable_amplitude": best,
        "probe_frequency_hz": freq_hz,
        "criteria": criteria,
        "note": "binary_search",
    }


def run_search(
    penetrations: list[float],
    *,
    output: Path,
    freq_hz: float = 0.5,
    criteria: int = 0,
) -> pd.DataFrame:
    rows = [
        max_stable_amplitude(pen, freq_hz=freq_hz, criteria=criteria)
        for pen in penetrations
    ]
    frame = pd.DataFrame(rows)
    stables = [
        float(v)
        for v in frame["max_stable_amplitude"].tolist()
        if v is not None and pd.notna(v)
    ]
    common = min(stables) if stables else None
    frame["common_amplitude"] = common
    frame["fairness_note"] = (
        "Relative penetration comparisons remain valid only when the same "
        "common_amplitude is used for every included penetration."
    )
    ensure_parent(output)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"common_amplitude={common}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="WECC IBR max stable amplitude search")
    parser.add_argument("--penetrations", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/amplitude_search.csv"),
    )
    parser.add_argument("--freq-hz", type=float, default=0.5)
    parser.add_argument("--criteria", type=int, default=0)
    args = parser.parse_args()
    run_search(args.penetrations, output=args.output, freq_hz=args.freq_hz, criteria=args.criteria)


if __name__ == "__main__":
    main()
