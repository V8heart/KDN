"""합성·실측 텔레메트리의 공통 스키마와 검증 함수."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE_COLUMNS = [
    "timestamp",
    "collection_timestamp",
    "raw_timestamp",
    "session_id",
    "attack_id",
    "waveform_kind",
    "waveform_frequency_hz",
    "waveform_amplitude_frac",
    "waveform_duty_cycle",
    "gpu_id",
    "sample_hz",
    "power_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "requested_interval_ms",
    "actual_interval_ms",
    "value_changed",
    "pid",
    "process_name",
    "id_user",
    "job_type",
    "gres_req",
    "label",
]

REQUIRED_COLUMNS = {
    "timestamp",
    "session_id",
    "gpu_id",
    "sample_hz",
    "power_w",
    "label",
}

NUMERIC_COLUMNS = {
    "timestamp",
    "raw_timestamp",
    "waveform_frequency_hz",
    "waveform_amplitude_frac",
    "waveform_duty_cycle",
    "gpu_id",
    "sample_hz",
    "power_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "requested_interval_ms",
    "actual_interval_ms",
    "pid",
}


@dataclass(frozen=True)
class SessionManifest:
    """한 캡처/합성 세션의 재현에 필요한 최소 메타데이터."""

    session_id: str
    label: str
    source: str
    sample_hz: float
    gpu_ids: list[int]
    rows: int
    seed: int | None = None
    workload: str | None = None
    notes: str | None = None
    attack_id: str | None = None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """누락된 선택 컬럼을 추가하고 공통 컬럼 순서로 정렬한다."""
    out = df.copy()
    for column in CORE_COLUMNS:
        if column not in out:
            out[column] = np.nan if column in NUMERIC_COLUMNS else pd.NA
    ordered = CORE_COLUMNS + [c for c in out.columns if c not in CORE_COLUMNS]
    return out.loc[:, ordered]


def validate_frame(df: pd.DataFrame, *, require_nonempty: bool = True) -> None:
    """스키마, 범위, 세션 내부 일관성을 검사한다."""
    if require_nonempty and df.empty:
        raise ValueError("텔레메트리 데이터가 비어 있습니다.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    for column in NUMERIC_COLUMNS & set(df.columns):
        values = pd.to_numeric(df[column], errors="coerce")
        if column in REQUIRED_COLUMNS and values.isna().any():
            raise ValueError(f"{column}에 숫자가 아닌 값 또는 결측값이 있습니다.")

    for column in ("util_gpu_pct", "mem_copy_util_pct"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if ((values < 0) | (values > 100)).any():
                raise ValueError(f"{column} 값은 0~100 범위여야 합니다.")
    if "power_w" in df and (pd.to_numeric(df["power_w"], errors="coerce") < 0).any():
        raise ValueError("power_w는 음수일 수 없습니다.")
    if "sample_hz" in df and (pd.to_numeric(df["sample_hz"], errors="coerce") <= 0).any():
        raise ValueError("sample_hz는 양수여야 합니다.")

    grouped = df.groupby(["session_id", "gpu_id"], dropna=False)
    for key, group in grouped:
        if group["label"].nunique(dropna=False) != 1:
            raise ValueError(f"세션 {key} 안에 여러 label이 섞였습니다.")
        hz = pd.to_numeric(group["sample_hz"], errors="coerce").dropna()
        if hz.nunique() != 1:
            raise ValueError(f"세션 {key} 안에 여러 sample_hz가 섞였습니다.")
        timestamps = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError(f"세션 {key} timestamp가 단조 증가하지 않습니다.")


def write_manifest(manifests: list[SessionManifest], path: str | Path) -> None:
    """세션 manifest를 사람이 읽을 수 있는 JSON으로 저장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(item) for item in manifests], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def manifest_from_dict(data: dict[str, Any]) -> SessionManifest:
    """JSON 등에서 읽은 사전을 검증 가능한 manifest로 변환한다."""
    return SessionManifest(**data)

