"""D6 수집 로그에서 요청 주기와 실효 값 갱신률을 요약한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "power_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "temp_c",
    "fb_used_mb",
]


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    actual = pd.to_numeric(df.get("actual_interval_ms"), errors="coerce").dropna()
    result = {
        "file": str(path),
        "rows": len(df),
        "sessions": int(df["session_id"].nunique()) if "session_id" in df else 0,
        "requested_interval_ms": float(df["requested_interval_ms"].dropna().iloc[0]),
        "actual_interval_ms": {
            "mean": float(actual.mean()) if len(actual) else None,
            "p50": float(actual.quantile(0.5)) if len(actual) else None,
            "p95": float(actual.quantile(0.95)) if len(actual) else None,
            "max": float(actual.max()) if len(actual) else None,
        },
        "row_refresh_ratio": float(df["value_changed"].astype(str).str.lower().eq("true").mean()),
        "fields": {},
    }
    for field in FIELDS:
        if field not in df:
            continue
        values = pd.to_numeric(df[field], errors="coerce")
        valid = values.dropna()
        changes = values.ne(values.shift()) & values.notna() & values.shift().notna()
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
        change_times = timestamps[changes].to_numpy()
        intervals = np.diff(change_times) if len(change_times) > 1 else np.array([])
        result["fields"][field] = {
            "missing_ratio": float(values.isna().mean()),
            "unique_ratio": float(valid.nunique() / len(valid)) if len(valid) else 0.0,
            "refresh_ratio": float(changes.mean()),
            "change_interval_s_p50": float(np.median(intervals)) if len(intervals) else None,
        }
    return result


def write_summary(inputs: list[Path], output: Path) -> list[dict]:
    summaries = [summarize_file(path) for path in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Telemetry observability summary", ""]
    for item in summaries:
        cadence = item["actual_interval_ms"]
        lines.extend(
            [
                f"## {item['file']}",
                f"- rows: {item['rows']}",
                f"- requested interval: {item['requested_interval_ms']:.3f} ms",
                f"- actual interval mean/p95: {cadence['mean']:.3f} / {cadence['p95']:.3f} ms",
                f"- row refresh ratio: {item['row_refresh_ratio']:.4f}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset/real/observability/summary.md"))
    args = parser.parse_args()
    write_summary(args.inputs, args.output)
    print(f"요약 저장: {args.output}")


if __name__ == "__main__":
    main()

