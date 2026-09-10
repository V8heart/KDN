"""Phase 3: attack-frequency × IBR-penetration sweep on public WECC 179.

Uses only penetrations that pass Phase-2 PFlow+TDS baselines. Metrics reuse
GENCLS omega features from remaining synchronous machines after SG→REGCA1
replacement. Results are reported for the public test system only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gridpulse-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bit2watt_impl.physics.simulation import (
    build_wecc_system_at_penetration,
    ensure_parent,
    simulate_profile,
)
from dataset.attack_profiles import sweep_profile


def _builder(penetration: float) -> Callable[[], Any]:
    def _build():
        return build_wecc_system_at_penetration(penetration, require_pflow=True)

    return _build


def load_viable_penetrations(baseline_csv: Path) -> list[float]:
    frame = pd.read_csv(baseline_csv)
    viable = frame[frame["pflow_ok"].astype(bool) & frame["tds_ok"].astype(bool)]
    if viable.empty:
        raise RuntimeError(f"no viable penetrations in {baseline_csv}")
    return [float(x) for x in viable["target_penetration"].tolist()]


def run_2d_sweep(
    *,
    penetrations: list[float],
    frequencies: np.ndarray,
    output: Path,
    heatmap: Path,
    overlay: Path,
    summary: Path,
    amplitude_frac: float = 0.02,
    duration_s: float = 14.0,
    max_chunk_s: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    if output.exists():
        prior = pd.read_csv(output)
        rows = prior.to_dict(orient="records")
        done = {
            (round(float(r["target_penetration"]), 6), round(float(r["frequency_hz"]), 6))
            for r in rows
        }
    else:
        done = set()

    for pen in penetrations:
        # Capture IBR meta once per penetration from a fresh build.
        probe = build_wecc_system_at_penetration(pen, require_pflow=True)
        meta = dict(getattr(probe, "_gridpulse_ibr_meta", {}) or {})
        del probe

        for frequency in frequencies:
            key = (round(float(pen), 6), round(float(frequency), 6))
            if key in done:
                print(f"[IBR 2D] skip existing {pen:.0%} @ {frequency:.2f} Hz")
                continue
            profile = sweep_profile(
                float(frequency),
                amplitude_frac=amplitude_frac,
                duration_s=duration_s,
            )
            result = simulate_profile(
                _builder(pen),
                profile,
                test_system="wecc_179_gencls_ibr",
                model_name="GENCLS",
                max_chunk_s=max_chunk_s,
                pq_selection="largest_p0",
            )
            row = dict(result.row)
            row["target_penetration"] = float(pen)
            row["achieved_penetration"] = meta.get("achieved_penetration")
            row["replaced_count"] = meta.get("replaced_count")
            row["inertia_before"] = meta.get("inertia_before")
            row["inertia_after"] = meta.get("inertia_after")
            row["inertia_reduction_frac"] = meta.get("inertia_reduction_frac")
            rows.append(row)
            ensure_parent(output)
            pd.DataFrame(rows).to_csv(output, index=False)
            status = "ok" if row.get("converged") else row.get("failure_reason")
            print(
                f"[IBR 2D] {pen:.0%} @ {frequency:.2f} Hz: {status} "
                f"osc_std={row.get('osc_std')} rocof={row.get('rocof_hz_s')}"
            )

    frame = pd.DataFrame(rows)
    ensure_parent(output)
    frame.to_csv(output, index=False)
    _write_plots(frame, heatmap=heatmap, overlay=overlay)
    report = _summary_report(frame, penetrations=penetrations)
    ensure_parent(summary)
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return frame


def _write_plots(frame: pd.DataFrame, *, heatmap: Path, overlay: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["converged"] = work["converged"].fillna(False).astype(bool)

    # Heatmap of osc_std
    pivot = work.pivot_table(
        index="target_penetration",
        columns="frequency_hz",
        values="osc_std",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not pivot.empty:
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{float(v):.0%}" for v in pivot.index])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{float(c):.2f}" for c in pivot.columns], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, label="osc_std (omega p.u.)")
    ax.set_xlabel("Attack frequency (Hz)")
    ax.set_ylabel("Target IBR penetration")
    ax.set_title("Public WECC 179: osc_std vs frequency × IBR penetration")
    fig.tight_layout()
    ensure_parent(heatmap)
    fig.savefig(heatmap, dpi=160)
    plt.close(fig)

    # Overlay curves
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for pen, group in work.groupby("target_penetration"):
        ok = group[group["converged"]].sort_values("frequency_hz")
        if ok.empty:
            continue
        ax.plot(
            ok["frequency_hz"],
            ok["osc_std"],
            marker="o",
            ms=3,
            label=f"{float(pen):.0%} IBR",
        )
        bad = group[~group["converged"]]
        if not bad.empty:
            ax.scatter(
                bad["frequency_hz"],
                np.zeros(len(bad)),
                marker="x",
                color="gray",
                s=20,
            )
    ax.set_xlabel("Attack frequency (Hz)")
    ax.set_ylabel("Max remaining-GENCLS omega std (p.u.)")
    ax.set_title("Public WECC 179: response vs attack frequency by IBR penetration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_parent(overlay)
    fig.savefig(overlay, dpi=160)
    plt.close(fig)


def _summary_report(frame: pd.DataFrame, *, penetrations: list[float]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "test_system": "wecc_179_gencls",
        "scope": "public dynamic test system; not a real regional grid",
        "method": "capacity-ranked GENCLS→REGCA1+REECA1 replacement; PQ.Ppf square-wave injection",
        "amplitude_frac_note": (
            "Default attack amplitude is 0.02 (not 0.10): on this public WECC+REGCA1 case, "
            "±10% PQ.Ppf square waves routinely bust TDS after SG→IBR replacement."
        ),
        "penetrations_requested": penetrations,
        "runs": int(len(frame)),
        "converged": int(frame["converged"].fillna(False).sum()) if len(frame) else 0,
        "by_penetration": {},
        "penetration_vs_mean_osc_std": {},
        "spearman_penetration_osc_std": None,
        "response_increases_with_penetration": None,
        "interpretation": "insufficient data",
    }
    if frame.empty:
        return report

    ok = frame[frame["converged"].fillna(False)].copy()
    report["convergence_rate"] = float(len(ok) / len(frame))

    for pen, group in frame.groupby("target_penetration"):
        g_ok = group[group["converged"].fillna(False)]
        report["by_penetration"][f"{float(pen):.2f}"] = {
            "runs": int(len(group)),
            "converged": int(len(g_ok)),
            "mean_osc_std": float(g_ok["osc_std"].mean()) if len(g_ok) else None,
            "mean_rocof_hz_s": float(g_ok["rocof_hz_s"].mean()) if len(g_ok) else None,
            "inertia_after": (
                float(g_ok["inertia_after"].iloc[0])
                if len(g_ok) and "inertia_after" in g_ok
                else None
            ),
        }
        if len(g_ok):
            report["penetration_vs_mean_osc_std"][f"{float(pen):.2f}"] = float(
                g_ok["osc_std"].mean()
            )

    # Per-frequency paired trend: mean slope of osc_std vs penetration
    if not ok.empty and ok["target_penetration"].nunique() >= 2:
        slopes: list[float] = []
        for _, grp in ok.groupby("frequency_hz"):
            if grp["target_penetration"].nunique() < 2:
                continue
            x = grp["target_penetration"].to_numpy(dtype=float)
            y = grp["osc_std"].to_numpy(dtype=float)
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and len(x) >= 2:
                slopes.append(float(np.polyfit(x, y, 1)[0]))
        n_levels = int(ok["target_penetration"].nunique())
        report["comparable_penetration_levels"] = n_levels
        failed_pens = sorted(
            {
                float(p)
                for p, g in frame.groupby("target_penetration")
                if not g["converged"].fillna(False).any()
            }
        )
        report["attack_tds_failed_penetrations"] = failed_pens
        if slopes:
            mean_slope = float(np.mean(slopes))
            report["mean_osc_std_vs_penetration_slope"] = mean_slope
            report["response_increases_with_penetration"] = bool(mean_slope > 0)
            try:
                from scipy.stats import spearmanr

                agg = (
                    ok.groupby("target_penetration")["osc_std"]
                    .mean()
                    .reset_index()
                )
                if len(agg) >= 3:
                    rho, pvalue = spearmanr(agg["target_penetration"], agg["osc_std"])
                    report["spearman_penetration_osc_std"] = {
                        "n": int(len(agg)),
                        "rho": float(rho) if np.isfinite(rho) else None,
                        "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                    }
            except Exception:
                pass

            fail_note = ""
            if failed_pens:
                fail_note = (
                    f" Attack-TDS failed for target penetrations {failed_pens} "
                    "(idle baseline may still have passed); those levels are excluded "
                    "from the response trend."
                )
            if n_levels < 3:
                report["interpretation"] = (
                    f"Only {n_levels} IBR levels produced converged attack runs on this "
                    "public WECC case, so the penetration trend is descriptive only "
                    f"(mean osc_std slope={'positive' if mean_slope > 0 else 'non-positive'})."
                    + fail_note
                )
            elif report["response_increases_with_penetration"]:
                report["interpretation"] = (
                    "On this public WECC case, mean remaining-GENCLS osc_std tended to "
                    "increase with IBR penetration for the tested square-wave PQ injections."
                    + fail_note
                )
            else:
                report["interpretation"] = (
                    "On this public WECC case, mean remaining-GENCLS osc_std did not "
                    "increase with IBR penetration for the tested square-wave PQ injections; "
                    "report the measured trend without claiming inverter-dominated risk."
                    + fail_note
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WECC IBR penetration × attack-frequency 2D sweep (public test system)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("dataset/eval/ibr_penetration_baseline.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_sweep.csv"),
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_heatmap.png"),
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_overlay.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("dataset/eval/ibr_frequency_2d_summary.json"),
    )
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--stop", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--amplitude-frac",
        type=float,
        default=0.02,
        help="PQ relative amplitude (0.10 is often unstable after IBR replacement on WECC)",
    )
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument(
        "--penetrations",
        type=float,
        nargs="*",
        default=None,
        help="Override baseline-viable list",
    )
    args = parser.parse_args()
    if args.penetrations:
        pens = [float(x) for x in args.penetrations]
    else:
        pens = load_viable_penetrations(args.baseline)
    freqs = np.arange(args.start, args.stop + 1e-9, args.step)
    run_2d_sweep(
        penetrations=pens,
        frequencies=freqs,
        output=args.output,
        heatmap=args.heatmap,
        overlay=args.overlay,
        summary=args.summary,
        amplitude_frac=args.amplitude_frac,
        duration_s=args.duration_s,
    )


if __name__ == "__main__":
    main()
