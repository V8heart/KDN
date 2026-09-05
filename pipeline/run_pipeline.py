"""
GridPulse 전체 파이프라인 오케스트레이션.

입력: DCGM/NVML로 수집한 텔레메트리 CSV (timestamp, power_w, util_gpu_pct, ... , label)
흐름:
  [1단계] 상시 저비용 스크리닝: 슬라이딩 윈도우별 변동폭 z-score로 의심 후보 선별
  [2단계] 의심 후보만 고해상도 특성 분석 (주기성/규칙성/평균유지)
  [3단계] RAG로 알려진 공격 시그니처와 대조 + LLM 판정·설명

논문 대비 우리 기여:
  - kHz 원신호를 탐지하려 하지 않음 (그건 물리계측 영역, 우선순위에서 제외)
  - 대신 "의심 활동만 선별 → 알려진 공격과 대조" 하는 실용 파이프라인 (검증된 2-tier IDS 패턴)
  - 새 공격은 corpus/에 문서 추가만으로 대응 (재학습 불필요)

사용:
  python run_pipeline.py --telemetry data.csv --baseline-mean 120 \
      --rag-backend tfidf --llm-backend stub
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import compute_window_features, features_to_description
from rag_analyzer import SignatureRetriever, analyze_with_llm

WINDOW = 200
STRIDE = 100


def stage1_screen(df: pd.DataFrame, z_threshold: float = 2.5):
    """1단계: 윈도우별 변동폭 계산, 전체 대비 z-score가 높은 윈도우만 후보로."""
    windows = []
    for start in range(0, len(df) - WINDOW, STRIDE):
        w = df.iloc[start:start + WINDOW]
        power = w["power_w"].values
        mean = np.mean(power)
        swing = (np.max(power) - np.min(power)) / mean if mean > 1e-6 else 0.0
        windows.append({"start": start, "swing_ratio": swing})

    wdf = pd.DataFrame(windows)
    if len(wdf) < 3:
        return wdf.assign(is_candidate=False)

    mu, sigma = wdf["swing_ratio"].mean(), wdf["swing_ratio"].std()
    wdf["z_score"] = (wdf["swing_ratio"] - mu) / (sigma + 1e-9)
    wdf["is_candidate"] = wdf["z_score"].abs() > z_threshold
    return wdf


def run(args):
    df = pd.read_csv(args.telemetry)
    print(f"[입력] {args.telemetry}: {len(df)}행")

    # 1단계
    wdf = stage1_screen(df, z_threshold=args.z_threshold)
    candidates = wdf[wdf["is_candidate"]]
    print(f"[1단계] 전체 {len(wdf)}개 윈도우 → 후보 {len(candidates)}개 "
          f"(z>{args.z_threshold})")

    if len(candidates) == 0:
        print("의심 후보 없음. 정상으로 판단.")
        return

    # RAG 검색기 준비 (1회)
    retriever = SignatureRetriever(backend=args.rag_backend)

    results = []
    util_col = "util_gpu_pct" if "util_gpu_pct" in df.columns else None

    for _, cand in candidates.iterrows():
        start = int(cand["start"])
        w = df.iloc[start:start + WINDOW]
        power = w["power_w"].values
        util = w[util_col].values if util_col else None

        # 2단계: 특성 분석
        feats = compute_window_features(power, util, sample_hz=args.sample_hz)
        desc = features_to_description(feats, baseline_mean_w=args.baseline_mean)

        # 3단계: RAG + LLM
        retrieved = retriever.search(desc, top_k=3)
        context = {}
        for c in ("id_user", "job_type", "gres_req"):
            if c in w.columns:
                context[c] = str(w[c].iloc[0])
        verdict = analyze_with_llm(desc, retrieved, context=context,
                                    backend=args.llm_backend, model=args.llm_model)

        results.append({
            "window_start": start,
            "swing_ratio": round(float(cand["swing_ratio"]), 3),
            "z_score": round(float(cand["z_score"]), 3),
            "description": desc,
            "rag_top": [(n, round(s, 3)) for n, s, _ in retrieved],
            "verdict": verdict,
        })

    print(f"\n[결과] 의심 후보 {len(results)}개 분석 완료:\n")
    for r in results:
        v = r["verdict"]
        print(f"  윈도우@{r['window_start']} (swing {r['swing_ratio']}, z {r['z_score']})")
        print(f"    → 위험도: {v.get('risk')} / 최근접: {v.get('closest_match')}")
        print(f"    → 근거: {v.get('reason')}")
        print(f"    → RAG: {r['rag_top']}\n")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"저장: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True, help="DCGM/NVML 수집 CSV")
    ap.add_argument("--baseline-mean", type=float, default=None,
                     help="정상 평균 전력(W). 정상 베이스라인 수집에서 산출")
    ap.add_argument("--sample-hz", type=float, default=1000.0,
                     help="텔레메트리 실효 샘플링레이트")
    ap.add_argument("--z-threshold", type=float, default=2.5)
    ap.add_argument("--rag-backend", choices=["sbert", "tfidf"], default="sbert")
    ap.add_argument("--llm-backend", choices=["ollama", "stub"], default="stub")
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
