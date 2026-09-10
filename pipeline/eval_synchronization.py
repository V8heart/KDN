"""D5 multi-GPU CSV에서 세션별 동기화 피처를 산출한다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.synchronization import synchronization_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dataset/eval/synchronization_report.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.telemetry)
    reports = []
    for session_id, group in frame.groupby("session_id", sort=False):
        if group["gpu_id"].nunique() < 2:
            continue
        reports.append({
            "session_id": str(session_id),
            "label": str(group["label"].iloc[0]) if "label" in group else None,
            "pairs": synchronization_features(group),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

