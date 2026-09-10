from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

from dataset.synth_attacks import cryptojacking, swma
from pipeline.features import compute_window_features
from pipeline.run_pipeline import run, stage1_screen


def test_feature_masks_keep_util_aligned():
    power = np.array([100, 101, np.nan, 103, 104, 105, 106, 107, 108], dtype=float)
    util = np.arange(len(power), dtype=float)
    feats = compute_window_features(power, util, sample_hz=10)
    assert feats["util_mean"] == np.mean(util[np.isfinite(power)])


def test_stage1_detects_flat_high_load():
    frame = cryptojacking(n=400)
    screened = stage1_screen(
        frame,
        baseline_mean_w=120,
        sample_hz=10,
        window=200,
        stride=100,
    )
    assert screened["is_candidate"].all()
    assert screened["candidate_reasons"].str.contains("persistent_high_load").all()


def test_stage1_detects_mechanical_swma():
    frame = swma(n=400)
    screened = stage1_screen(
        frame,
        baseline_mean_w=120,
        sample_hz=10,
        window=200,
        stride=100,
    )
    assert screened["candidate_reasons"].str.contains("mechanical_periodicity").all()


def test_schema_groups_prevent_cross_session_windows():
    first = pd.DataFrame({"power_w": np.full(200, 100.0)})
    second = pd.DataFrame({"power_w": np.full(200, 300.0)})
    assert len(stage1_screen(first, window=200, stride=100)) == 1
    assert len(stage1_screen(second, window=200, stride=100)) == 1


def test_pipeline_emits_tier_b_join_fields(tmp_path, capsys):
    telemetry = tmp_path / "swma.csv"
    swma(n=400).to_csv(telemetry, index=False)
    results = run(
        Namespace(
            telemetry=str(telemetry),
            baseline_mean=120.0,
            sample_hz=None,
            window=200,
            stride=100,
            z_threshold=2.5,
            rag_backend="tfidf",
            llm_backend="stub",
            llm_model="unused",
            out=None,
        )
    )
    capsys.readouterr()
    assert results
    result = results[0]
    assert result["attack_id"]
    assert result["window_id"]
    assert result["threat_id"] == result["verdict"]["closest_match"]
    assert 0 <= result["anomaly_score"] <= 1
    assert result["score_version"] == "cyber-stage1-or-v1"

