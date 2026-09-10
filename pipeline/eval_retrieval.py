"""D7 RAG positive/open-set 평가셋 생성과 retrieval 지표 계산."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import compute_window_features, features_to_description
from pipeline.rag_analyzer import SignatureRetriever
from pipeline.run_pipeline import context_to_query, infer_sample_hz


def expected_document(label: str) -> str:
    return "normal_workloads" if label.startswith("normal") else label


def build_cases(telemetry: Path) -> list[dict]:
    df = pd.read_csv(telemetry)
    cases = []
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        hz = infer_sample_hz(group)
        feats = compute_window_features(
            group["power_w"].to_numpy(),
            group["util_gpu_pct"].to_numpy() if "util_gpu_pct" in group else None,
            sample_hz=hz,
        )
        description = features_to_description(feats, baseline_mean_w=120)
        context = {
            key: str(group[key].iloc[0])
            for key in ("id_user", "job_type", "gres_req")
            if key in group
        }
        label = str(group["label"].iloc[0])
        cases.append({
            "case_id": f"{session_id}-gpu{gpu_id}",
            "kind": "positive",
            "label": label,
            "expected_document": expected_document(label),
            "description": description,
            "query": f"{description} {context_to_query(context)}".strip(),
        })

    rng = np.random.default_rng(999)
    unknown_signals = {
        "unknown_single_impulse": np.r_[np.full(499, 120.0), 350.0, np.full(500, 120.0)],
        "unknown_slow_drift": np.linspace(70, 180, 1000),
        "unknown_random_walk": 120 + np.cumsum(rng.normal(0, 1.5, 1000)),
    }
    for name, signal in unknown_signals.items():
        feats = compute_window_features(signal, sample_hz=10)
        description = features_to_description(feats, baseline_mean_w=120)
        cases.append({
            "case_id": name,
            "kind": "unknown",
            "label": "unknown",
            "expected_document": None,
            "description": description,
            "query": description,
        })
    return cases


def evaluate(cases: list[dict], backend: str = "tfidf", unknown_threshold: float = 0.28) -> dict:
    retriever = SignatureRetriever(backend=backend)
    positive = correct = 0
    unknown_scores = []
    details = []
    for case in cases:
        result = retriever.search(case["query"], top_k=3)
        top_name, top_score, _ = result[0]
        if case["kind"] == "positive":
            positive += 1
            correct += int(top_name == case["expected_document"])
        else:
            unknown_scores.append(top_score)
        details.append({
            "case_id": case["case_id"],
            "expected": case["expected_document"],
            "top1": top_name,
            "top1_score": top_score,
        })
    return {
        "backend": backend,
        "positive_cases": positive,
        "top1_accuracy": correct / positive if positive else 0.0,
        "unknown_cases": len(unknown_scores),
        "unknown_max_similarity": max(unknown_scores, default=None),
        "unknown_threshold": unknown_threshold,
        "unknown_rejection_rate": (
            sum(score < unknown_threshold for score in unknown_scores) / len(unknown_scores)
            if unknown_scores else 0.0
        ),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v2.csv")
    parser.add_argument("--cases-out", type=Path, default=ROOT / "dataset/eval/retrieval_eval.jsonl")
    parser.add_argument("--report-out", type=Path, default=ROOT / "dataset/eval/retrieval_report.json")
    parser.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--unknown-threshold", type=float, default=0.28)
    args = parser.parse_args()
    cases = build_cases(args.telemetry)
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    report = evaluate(cases, args.backend, args.unknown_threshold)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

