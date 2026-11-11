"""
Model Validation & Unit Test Suite — Credit Risk Scorecard
===========================================================
Validates core mathematical transformations, leakage-free cleaning,
Bayes probability calibration, PDO score boundaries, and artifact integrity.

Usage:
    python test_pipeline.py
    # or
    pytest test_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from train_model import (
    BASE_ODDS,
    BASE_SCORE,
    FACTOR,
    OFFSET,
    PAST_DUE_COLS,
    PDO,
    apply_cleaning,
    calculate_psi,
    calculate_woe_iv,
    calibrate_probabilities,
    compute_cleaning_params,
    pd_to_score,
)

MODEL_DIR = Path("models")


# ──────────────────────────────────────────────────────────────────────
# 1. ARTIFACT INTEGRITY TESTS
# ──────────────────────────────────────────────────────────────────────
def test_artifacts_exist():
    """Verify that all 7 mandatory artifacts exist in models/."""
    expected_files = [
        "iv_table.csv",
        "lgbm_model.pkl",
        "lr_pipeline.pkl",
        "metadata.pkl",
        "metrics.json",
        "shap_background.csv",
        "validation_curves.json",
    ]
    for filename in expected_files:
        p = MODEL_DIR / filename
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"


def test_metadata_keys():
    """Verify metadata dictionary contract."""
    meta = joblib.load(MODEL_DIR / "metadata.pkl")
    assert "selected_features" in meta
    assert "cleaning_params" in meta
    assert "pdo_params" in meta
    assert len(meta["selected_features"]) == 9

    pdo = meta["pdo_params"]
    assert pdo["base_score"] == 600
    assert pdo["pdo"] == 50
    assert np.isclose(pdo["factor"], 50 / np.log(2), atol=1e-4)


def test_metrics_governance_pass():
    """Verify that GINI gap and PSI meet Basel / BDDK regulatory thresholds."""
    with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # GINI gap <= 0.05 (anti-overfitting)
    lgbm_gap = abs(metrics["LightGBM"]["Train"]["GINI"] - metrics["LightGBM"]["Test"]["GINI"])
    assert lgbm_gap <= 0.05, f"LightGBM GINI gap ({lgbm_gap:.4f}) exceeds 0.05 threshold!"

    # PSI < 0.10 (population stability)
    assert metrics["PSI"]["LightGBM"] < 0.10, "LightGBM PSI indicates significant population shift!"
    assert metrics["PSI"]["LogisticRegression"] < 0.10, "Logistic Regression PSI exceeds stable threshold!"


# ──────────────────────────────────────────────────────────────────────
# 2. MATHEMATICAL & SCORING TESTS
# ──────────────────────────────────────────────────────────────────────
def test_bayes_calibration_monotonicity():
    """Higher raw probability must strictly yield higher calibrated probability."""
    raw_probs = np.array([0.05, 0.20, 0.50, 0.80, 0.95])
    cal_probs = calibrate_probabilities(raw_probs, train_base_rate=0.0668, balanced_rate=0.5)

    assert np.all(np.diff(cal_probs) > 0), "Calibration broke monotonicity!"
    # When balanced rate is 0.5 and true base is ~6.68%, calibrated must be lower than raw
    assert np.all(cal_probs < raw_probs), "Calibrated probabilities must reflect the lower population base rate!"
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0)), "Probabilities must stay in [0, 1]!"


def test_pdo_score_bounds_and_direction():
    """Credit score must be strictly in [300, 850] and monotonically decreasing with risk."""
    pd_values = np.array([0.001, 0.01, 0.05, 0.10, 0.30, 0.70, 0.999])
    scores = pd_to_score(pd_values)

    # Range constraint
    assert np.all(scores >= 300), f"Scores violated lower bound: {scores.min()}"
    assert np.all(scores <= 850), f"Scores violated upper bound: {scores.max()}"

    # Risk inverse direction: higher PD -> lower score
    assert np.all(np.diff(scores) <= 0), "Credit scores must decrease as PD increases!"

    # Known anchor check: Base Odds (1/19 ~ 5% PD) should map close to Base Score (600)
    anchor_score = pd_to_score(np.array([1 / 20]))[0]  # Odds = 19
    assert abs(anchor_score - 600) <= 2, f"Base odds did not map to ~600 score (got {anchor_score})!"


def test_psi_identical_distributions():
    """PSI of identical distributions must be virtually zero."""
    rng = np.random.RandomState(42)
    sample = rng.normal(loc=650, scale=80, size=5000)
    psi = calculate_psi(sample, sample)
    assert psi < 0.001, f"PSI of identical distributions should be ~0, got {psi}"


def test_woe_iv_zero_division_safety():
    """calculate_woe_iv must never raise ZeroDivisionError or produce NaN/inf."""
    df_test = pd.DataFrame({
        "util": [0.0, 0.0, 0.0, 0.5, 0.9, 1.2, 0.0, 0.1],
        "default": [0, 0, 0, 0, 1, 1, 0, 0],
    })
    woe_table, iv = calculate_woe_iv(df_test, feature="util", target="default", n_bins=4)

    assert not np.isnan(iv), "IV resulted in NaN!"
    assert not np.isinf(iv), "IV resulted in Inf!"
    assert iv >= 0, "Information Value cannot be negative!"


# ──────────────────────────────────────────────────────────────────────
# 3. LEAKAGE-FREE CLEANING TESTS
# ──────────────────────────────────────────────────────────────────────
def test_apply_cleaning_deterministic():
    """Cleaning rules must replace out-of-range age, systemic codes, and clip outliers."""
    params = {
        "age_median": 52.0,
        "NumberOfTime30-59DaysPastDueNotWorse_median": 0.0,
        "NumberOfTimes90DaysLate_median": 0.0,
        "NumberOfTime60-89DaysPastDueNotWorse_median": 0.0,
        "revolving_clip_upper": 1.09,
        "MonthlyIncome_median": 5400.0,
        "NumberOfDependents_median": 0.0,
    }

    raw_test = pd.DataFrame([{
        "age": 14,                                     # Out of range (<18)
        "NumberOfTime30-59DaysPastDueNotWorse": 98,   # Systemic code
        "NumberOfTimes90DaysLate": 96,                 # Systemic code
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "RevolvingUtilizationOfUnsecuredLines": 50.0, # Outlier (>1.09)
        "MonthlyIncome": np.nan,                       # Missing
        "NumberOfDependents": np.nan,                  # Missing
        "DebtRatio": 0.35,
        "NumberOfOpenCreditLinesAndLoans": 5,
    }])

    cleaned = apply_cleaning(raw_test, params)

    assert cleaned.loc[0, "age"] == 52.0
    assert cleaned.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"] == 0.0
    assert cleaned.loc[0, "NumberOfTimes90DaysLate"] == 0.0
    assert cleaned.loc[0, "RevolvingUtilizationOfUnsecuredLines"] == 1.09
    assert cleaned.loc[0, "MonthlyIncome"] == 5400.0
    assert cleaned.loc[0, "NumberOfDependents"] == 0.0
    assert not cleaned.isna().any().any(), "Cleaned dataset contains lingering NaNs!"


# ──────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  RUNNING CREDIT RISK SCORECARD TEST SUITE")
    print("=" * 64)
    tests = [
        test_artifacts_exist,
        test_metadata_keys,
        test_metrics_governance_pass,
        test_bayes_calibration_monotonicity,
        test_pdo_score_bounds_and_direction,
        test_psi_identical_distributions,
        test_woe_iv_zero_division_safety,
        test_apply_cleaning_deterministic,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__:<42} [PASSED]")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__:<42} [FAILED]: {e}")

    print("=" * 64)
    print(f"  Results: {passed}/{len(tests)} tests passed.")
    print("=" * 64)
