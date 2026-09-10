"""RTX 4090에서 지원되는 NVML 필드를 long-format CSV로 수집한다."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pynvml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.schema import normalize_frame, validate_frame


def _safe(call: Callable[[], Any], errors: dict[str, int], field: str) -> Any:
    try:
        return call()
    except pynvml.NVMLError:
        errors[field] = errors.get(field, 0) + 1
        return None


def _process_name(pid: int) -> str | None:
    for suffix in ("comm", "cmdline"):
        try:
            text = Path(f"/proc/{pid}/{suffix}").read_text(errors="replace").replace("\0", " ").strip()
            if text:
                return text
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return None


def _processes(handle, errors: dict[str, int]) -> list[dict[str, Any]]:
    procs = _safe(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(handle), errors, "processes") or []
    result = []
    for proc in procs:
        used = getattr(proc, "usedGpuMemory", None)
        result.append(
            {
                "pid": int(proc.pid),
                "process_name": _process_name(int(proc.pid)),
                "used_gpu_memory_mb": None if used is None else float(used / 1024**2),
            }
        )
    return sorted(result, key=lambda item: item["used_gpu_memory_mb"] or -1, reverse=True)


def read_gpu(handle, gpu_id: int, errors: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """한 GPU의 계획서 Tier 1 필드를 읽는다."""
    util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle), errors, "utilization")
    memory = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle), errors, "memory")
    processes = _processes(handle, errors)
    primary = processes[0] if processes else {}
    row = {
        "gpu_id": gpu_id,
        "power_w": _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, errors, "power_w"),
        "util_gpu_pct": None if util is None else float(util.gpu),
        "mem_copy_util_pct": None if util is None else float(util.memory),
        "sm_clock_mhz": _safe(
            lambda: float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            errors,
            "sm_clock_mhz",
        ),
        "temp_c": _safe(
            lambda: float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
            errors,
            "temp_c",
        ),
        "fb_used_mb": None if memory is None else float(memory.used / 1024**2),
        "pid": primary.get("pid"),
        "process_name": primary.get("process_name"),
    }
    return row, processes


def collect(
    output: Path,
    *,
    duration_s: float,
    interval_ms: float,
    gpu_ids: list[int],
    session_id: str,
    label: str,
    id_user: str,
    job_type: str,
    gres_req: str,
    dcgm_crosscheck: bool = False,
) -> pd.DataFrame:
    if duration_s <= 0 or interval_ms <= 0:
        raise ValueError("duration과 interval은 양수여야 합니다.")

    pynvml.nvmlInit()
    errors: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    seen_processes: dict[int, dict[str, Any]] = {}
    previous_values: dict[int, tuple[Any, ...]] = {}
    previous_times: dict[int, float] = {}
    start = time.monotonic()
    deadline = start + duration_s
    next_tick = start
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if any(gpu < 0 or gpu >= device_count for gpu in gpu_ids):
            raise ValueError(f"GPU ID 범위 오류: 장치 수={device_count}, 요청={gpu_ids}")
        handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpu_ids}

        while time.monotonic() < deadline:
            now_mono = time.monotonic()
            for gpu_id, handle in handles.items():
                values, processes = read_gpu(handle, gpu_id, errors)
                sampled = time.monotonic()
                for process in processes:
                    seen_processes[process["pid"]] = process
                fingerprint = tuple(values.get(k) for k in (
                    "power_w", "util_gpu_pct", "mem_copy_util_pct",
                    "sm_clock_mhz", "temp_c", "fb_used_mb",
                ))
                rows.append(
                    {
                        "timestamp": sampled - start,
                        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                        "raw_timestamp": None,
                        "session_id": session_id,
                        "sample_hz": 1000.0 / interval_ms,
                        "requested_interval_ms": interval_ms,
                        "actual_interval_ms": (
                            None if gpu_id not in previous_times
                            else (sampled - previous_times[gpu_id]) * 1000.0
                        ),
                        "value_changed": (
                            True if gpu_id not in previous_values
                            else fingerprint != previous_values[gpu_id]
                        ),
                        "id_user": id_user,
                        "job_type": job_type,
                        "gres_req": gres_req,
                        "label": label,
                        **values,
                    }
                )
                previous_values[gpu_id] = fingerprint
                previous_times[gpu_id] = sampled
            next_tick += interval_ms / 1000.0
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        pynvml.nvmlShutdown()

    frame = normalize_frame(pd.DataFrame(rows))
    validate_frame(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output)

    metadata = {
        "session_id": session_id,
        "label": label,
        "duration_s": duration_s,
        "requested_interval_ms": interval_ms,
        "gpu_ids": gpu_ids,
        "rows": len(frame),
        "field_errors": errors,
        "processes": list(seen_processes.values()),
        "collector": "pynvml",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if dcgm_crosscheck:
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", "100,101,150,155,203,204,250,251", "-c", "2"],
            text=True,
            capture_output=True,
            check=False,
        )
        output.with_suffix(".dcgm.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--interval-ms", type=float, default=1000.0)
    parser.add_argument("--gpu-ids", default="0", help="쉼표로 구분한 GPU ID")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--id-user", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--job-type", default="unknown")
    parser.add_argument("--gres-req", default="gpu:1")
    parser.add_argument("--dcgm-crosscheck", action="store_true")
    args = parser.parse_args()
    session_id = args.session_id or f"real-{uuid.uuid4().hex[:12]}"
    frame = collect(
        args.output,
        duration_s=args.duration,
        interval_ms=args.interval_ms,
        gpu_ids=[int(item) for item in args.gpu_ids.split(",")],
        session_id=session_id,
        label=args.label,
        id_user=args.id_user,
        job_type=args.job_type,
        gres_req=args.gres_req,
        dcgm_crosscheck=args.dcgm_crosscheck,
    )
    print(f"수집 완료: {args.output} ({len(frame):,}행, session={session_id})")


if __name__ == "__main__":
    main()

