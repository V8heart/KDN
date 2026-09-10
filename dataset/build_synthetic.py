"""D0~D4 합성 데이터셋을 결정론적으로 생성한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.schema import SessionManifest, validate_frame, write_manifest
from dataset.synth_attacks import ATTACK_GENERATORS
from dataset.synth_common import write_csv
from dataset.synth_normal_patterns import NORMAL_GENERATORS
from pipeline.features import compute_window_features


def _manifest(df: pd.DataFrame, source: str, seed: int) -> SessionManifest:
    return SessionManifest(
        session_id=str(df["session_id"].iloc[0]),
        label=str(df["label"].iloc[0]),
        source=source,
        sample_hz=float(df["sample_hz"].iloc[0]),
        gpu_ids=sorted(int(x) for x in df["gpu_id"].unique()),
        rows=len(df),
        seed=seed,
        workload=str(df["job_type"].iloc[0]),
        attack_id=str(df["attack_id"].iloc[0]) if "attack_id" in df else None,
    )


def build(output_dir: Path, *, rows: int = 1200, sample_hz: float = 10) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    manifests: list[SessionManifest] = []
    stats: dict[str, dict[str, float]] = {}

    for index, (label, generator) in enumerate(NORMAL_GENERATORS.items(), start=100):
        frame = generator(n=rows, sample_hz=sample_hz, seed=index)
        write_csv(frame, output_dir / "normal" / f"{label}.csv")
        frames.append(frame)
        manifests.append(_manifest(frame, "synthetic", index))

    for index, (label, generator) in enumerate(ATTACK_GENERATORS.items(), start=200):
        frame = generator(n=rows, sample_hz=sample_hz, seed=index)
        write_csv(frame, output_dir / "attacks" / f"{label}.csv")
        frames.append(frame)
        manifests.append(_manifest(frame, "synthetic", index))

    combined = pd.concat(frames, ignore_index=True)
    validate_frame(combined)
    write_csv(combined, output_dir / "all_v2.csv")
    write_manifest(manifests, output_dir / "manifest.json")

    for frame in frames:
        label = str(frame["label"].iloc[0])
        feats = compute_window_features(
            frame["power_w"].to_numpy(),
            frame["util_gpu_pct"].to_numpy(),
            sample_hz=sample_hz,
        )
        stats[label] = {
            key: round(float(feats[key]), 5)
            for key in ("mean_w", "swing_ratio", "duty_regularity", "periodicity_strength")
        }
    (output_dir / "feature_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset" / "synthetic")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    args = parser.parse_args()
    frame = build(args.output_dir, rows=args.rows, sample_hz=args.sample_hz)
    print(f"생성 완료: {args.output_dir / 'all_v2.csv'} ({len(frame):,}행)")


if __name__ == "__main__":
    main()

