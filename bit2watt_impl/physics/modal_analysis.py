"""Eigenvalue (modal) check near the WECC 1.0–1.1 Hz response peak."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bit2watt_impl.physics.simulation import (
    build_wecc_system_at_penetration,
    ensure_parent,
)


def _modes_from_eig(ss: Any) -> list[dict[str, float]]:
    """Extract oscillatory modes from ANDES EIG eigenvalues."""
    mu = np.asarray(ss.EIG.mu, dtype=complex).reshape(-1)
    modes: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for eig in mu:
        sigma = float(np.real(eig))
        omega = float(np.imag(eig))
        if abs(omega) < 1e-9:
            continue
        # Keep positive-imaginary representatives only.
        if omega < 0:
            continue
        freq_hz = omega / (2.0 * np.pi)
        damp = -sigma / np.sqrt(sigma * sigma + omega * omega)
        key = (round(freq_hz, 4), round(damp, 4))
        if key in seen:
            continue
        seen.add(key)
        modes.append(
            {
                "freq_hz": float(freq_hz),
                "damping_ratio": float(damp),
                "real": sigma,
                "imag": omega,
            }
        )
    modes.sort(key=lambda m: m["freq_hz"])
    return modes


def analyze_penetration(pen: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_penetration": float(pen),
        "scope": "public dynamic test system; not a real regional grid",
        "eig_ok": False,
        "modes": [],
        "modes_near_1p0_1p1_hz": [],
        "failure_reason": "",
    }
    try:
        ss = build_wecc_system_at_penetration(pen, require_pflow=True)
        if hasattr(ss.TDS, "init"):
            ss.TDS.init()
        ok = bool(ss.EIG.run())
        row["eig_ok"] = ok
        if not ok:
            row["failure_reason"] = "EIG.run returned false"
            return row
        modes = _modes_from_eig(ss)
        row["modes"] = modes
        near = [m for m in modes if 0.95 <= m["freq_hz"] <= 1.15]
        row["modes_near_1p0_1p1_hz"] = near
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return row


def run_modal(
    penetrations: list[float],
    *,
    output: Path,
) -> dict[str, Any]:
    results = [analyze_penetration(pen) for pen in penetrations]
    any_near = any(bool(r.get("modes_near_1p0_1p1_hz")) for r in results)
    report = {
        "scope": "public dynamic test system; not a real regional grid",
        "penetrations": penetrations,
        "results": results,
        "peak_language": (
            "resonance"
            if any_near
            else "response peak"
        ),
        "interpretation": (
            "ANDES EIG found oscillatory mode(s) near 1.0–1.1 Hz; "
            "the WECC osc_std peak may be described as resonance."
            if any_near
            else "ANDES EIG did not confirm a mode near 1.0–1.1 Hz "
            "(or EIG failed); describe the WECC osc_std feature as a "
            "response peak only, not resonance."
        ),
    }
    ensure_parent(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WECC IBR modal analysis near 1.0–1.1 Hz")
    parser.add_argument("--penetrations", type=float, nargs="+", default=[0.0, 0.3])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/modal_analysis.json"),
    )
    args = parser.parse_args()
    run_modal(args.penetrations, output=args.output)


if __name__ == "__main__":
    main()
