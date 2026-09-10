"""Join Cyber Stage-1/RAG results with Physics Tier B by attack_id."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_cyber_results(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("pipeline output must be a JSON list")
    return data


def rollup_cyber_windows(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate candidate windows to the scenario grain used by simulation."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        attack_id = item.get("attack_id") or item.get("session_id")
        if not attack_id:
            raise ValueError("cyber result lacks attack_id and session_id")
        groups.setdefault(str(attack_id), []).append(item)

    rows: list[dict[str, Any]] = []
    for attack_id, items in groups.items():
        selected = max(items, key=lambda value: float(value.get("anomaly_score", 0.0)))
        threats = [
            str(item.get("threat_id") or item.get("verdict", {}).get("closest_match"))
            for item in items
        ]
        threat_id = Counter(threats).most_common(1)[0][0] if threats else None
        scores = [float(item.get("anomaly_score", 0.0)) for item in items]
        row: dict[str, Any] = {
            "attack_id": attack_id,
            "session_id": selected.get("session_id"),
            "threat_id": threat_id,
            "cyber_anomaly_score_max": max(scores),
            "cyber_anomaly_score_median": float(pd.Series(scores).median()),
            "cyber_candidate_windows": len(items),
            "cyber_rag_top_score": (
                float(selected["rag_top"][0][1]) if selected.get("rag_top") else None
            ),
        }
        for key, value in selected.get("features", {}).items():
            row[f"cyber_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_physics_results(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("at least one physics CSV is required")
    frame = pd.concat(frames, ignore_index=True)
    required = {"attack_id", "test_system", "converged"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"physics CSV missing columns: {sorted(missing)}")
    duplicate = frame.duplicated(["attack_id", "test_system"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["attack_id", "test_system"]].to_dict("records")
        raise ValueError(f"duplicate physics scenario/system keys: {keys}")
    return frame


def join_cyber_physics(
    cyber: pd.DataFrame,
    physics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = physics.merge(cyber, on="attack_id", how="left", validate="many_to_one")
    unmatched = joined[joined["cyber_anomaly_score_max"].isna()].copy()
    return joined, unmatched


def correlation_report(joined: pd.DataFrame) -> dict[str, Any]:
    usable = joined[
        joined["converged"].fillna(False)
        & joined["cyber_anomaly_score_max"].notna()
        & joined["osc_std"].notna()
    ].copy()
    report: dict[str, Any] = {
        "scope": "public dynamic test systems only; not a real regional grid",
        "joined_rows": int(len(joined)),
        "usable_rows": int(len(usable)),
        "converged_rows": int(joined["converged"].fillna(False).sum()),
        "minimum_rows_for_inference": 8,
        "inference_status": (
            "descriptive_only" if len(usable) < 8 else "candidate_for_validation"
        ),
        "warning": (
            "Very small samples can produce perfect rank correlations; "
            "do not treat rho or p-values as calibrated evidence."
        ),
        "spearman": {},
    }
    for target in ("osc_std", "rocof_hz_s"):
        subset = usable[["cyber_anomaly_score_max", target]].dropna()
        if len(subset) < 3:
            report["spearman"][target] = {"n": int(len(subset)), "rho": None, "pvalue": None}
            continue
        rho, pvalue = spearmanr(
            subset["cyber_anomaly_score_max"],
            subset[target],
        )
        report["spearman"][target] = {
            "n": int(len(subset)),
            "rho": float(rho) if np.isfinite(rho) else None,
            "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyber-json", type=Path, required=True)
    parser.add_argument("--physics-csv", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/eval/cyber_physics_joined.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset/eval/cyber_physics_correlation.json"),
    )
    parser.add_argument(
        "--rejects",
        type=Path,
        default=Path("dataset/eval/cyber_physics_unmatched.csv"),
    )
    args = parser.parse_args()
    cyber = rollup_cyber_windows(load_cyber_results(args.cyber_json))
    physics = load_physics_results(args.physics_csv)
    joined, unmatched = join_cyber_physics(cyber, physics)
    for target in (args.out, args.report, args.rejects):
        target.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out, index=False)
    unmatched.to_csv(args.rejects, index=False)
    report = correlation_report(joined)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
