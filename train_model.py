"""
Credit Risk Scorecard Engine — Production-Grade Training Pipeline
=================================================================
Dataset : Kaggle "Give Me Some Credit" (cs-training.csv)
Author  : Credit Risk & Data Science
Purpose : Build a robust, bank-ready credit scorecard with full
          leakage-safe preprocessing, WoE/IV feature selection,
          LightGBM + Logistic Regression modelling, and banking-grade
          evaluation (GINI, KS, ROC-AUC).

Usage
-----
    python train_model.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/cs-training.csv")
MODEL_DIR = Path("models")
TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
SYSTEMIC_ERROR_CODES = {96, 98}

# PDO Scorecard Parameters (industry standard)
BASE_SCORE = 600
BASE_ODDS = 1 / 19  # ~5% PD
PDO = 50
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(1 / BASE_ODDS)  # = 600 - Factor*ln(19)

# ──────────────────────────────────────────────────────────────────────
# 1.  ROBUST DATA CLEANING & ANOMALY MANAGEMENT
# ──────────────────────────────────────────────────────────────────────


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV and drop the unnamed index column."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print(f"[DATA] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {path}")
    return df


def compute_cleaning_params(df_train: pd.DataFrame) -> dict:
    """
    Derive all cleaning parameters **exclusively** from the training set
    so that no information from the test set leaks into the pipeline.
    """
    params: dict = {}

    # Age — median for out-of-range replacement
    valid_ages = df_train["age"][(df_train["age"] > 18) & (df_train["age"] <= 95)]
    params["age_median"] = float(valid_ages.median())

    # Past-due columns — median after removing systemic codes
    for col in PAST_DUE_COLS:
        clean = df_train[col][~df_train[col].isin(SYSTEMIC_ERROR_CODES)]
        params[f"{col}_median"] = float(clean.median())

    # RevolvingUtilizationOfUnsecuredLines — 99th percentile clip boundary
    params["revolving_clip_upper"] = float(
        df_train["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
    )

    # MonthlyIncome & NumberOfDependents — imputation medians
    params["MonthlyIncome_median"] = float(df_train["MonthlyIncome"].median())
    params["NumberOfDependents_median"] = float(df_train["NumberOfDependents"].median())

    return params


def apply_cleaning(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules using pre-computed train parameters.
    Safe to call on both train and test splits without data leakage.
    """
    df = df.copy()

    # 1. Age anomalies → train median
    mask_age = (df["age"] <= 18) | (df["age"] > 95)
    df.loc[mask_age, "age"] = params["age_median"]

    # 2. Systemic error codes (96, 98) in past-due columns → train median
    for col in PAST_DUE_COLS:
        mask = df[col].isin(SYSTEMIC_ERROR_CODES)
        df.loc[mask, col] = params[f"{col}_median"]

    # 3. Revolving utilisation — clip extreme outliers at 99th pctl
    upper = params["revolving_clip_upper"]
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=upper)

    # 4. Missing value imputation
    df["MonthlyIncome"].fillna(params["MonthlyIncome_median"], inplace=True)
    df["NumberOfDependents"].fillna(params["NumberOfDependents_median"], inplace=True)

    # 5. Safety net — catch any remaining NaN / inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    medians = {col: params.get(f"{col}_median", 0) for col in df.columns}
    df = df.fillna(medians).fillna(0)

    return df


# ──────────────────────────────────────────────────────────────────────
# 2.  WoE / IV CALCULATION  (pure pandas + numpy, no external package)
# ──────────────────────────────────────────────────────────────────────


def _bin_feature(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Quantile-bin a continuous feature into at most `n_bins` groups.
    For zero-inflated / low-cardinality columns where qcut collapses
    to fewer than 3 bins, fall back to fixed discrete breakpoints.
    """
    try:
        binned = pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(series.rank(method="first"), bins=n_bins, duplicates="drop")

    # Fallback for zero-inflated columns (e.g., past-due counts)
    if binned.nunique() < 3:
        binned = pd.cut(
            series,
            bins=[-np.inf, 0, 1, 2, 5, np.inf],
            duplicates="drop",
        )
    return binned


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
    laplace: float = 0.5,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence and Information Value for a single
    feature using only pandas/numpy.

    Parameters
    ----------
    df       : Training DataFrame.
    feature  : Column name.
    target   : Binary target column.
    n_bins   : Max quantile bins (≤ 10).
    laplace  : Smoothing constant to prevent log(0).

    Returns
    -------
    (woe_table, iv_value)
    """
    tmp = df[[feature, target]].copy()
    tmp["bin"] = _bin_feature(tmp[feature], n_bins=n_bins)

    grouped = tmp.groupby("bin", observed=False)[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    # Laplace smoothing — prevents division by zero / log(0)
    grouped["dist_events"] = (grouped["events"] + laplace) / (
        total_events + laplace * len(grouped)
    )
    grouped["dist_non_events"] = (grouped["non_events"] + laplace) / (
        total_non_events + laplace * len(grouped)
    )

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv_component"] = (
        grouped["dist_non_events"] - grouped["dist_events"]
    ) * grouped["woe"]

    iv_value = float(grouped["iv_component"].sum())

    return grouped.reset_index(), iv_value


def run_iv_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    iv_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute IV for every feature on the training set, drop those below
    the threshold, and return a summary table plus the surviving feature
    list.
    """
    records = []
    for feat in features:
        _, iv = calculate_woe_iv(
            pd.concat([X_train[feat], y_train], axis=1), feat, TARGET
        )
        records.append({"Feature": feat, "IV": round(iv, 6)})

    iv_df = (
        pd.DataFrame(records)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )

    # Strength labels (banking convention)
    def _strength(v: float) -> str:
        if v < 0.02:
            return "Not Predictive"
        elif v < 0.1:
            return "Weak"
        elif v < 0.3:
            return "Medium"
        elif v < 0.5:
            return "Strong"
        return "Very Strong"

    iv_df["Strength"] = iv_df["IV"].apply(_strength)

    selected = iv_df.loc[iv_df["IV"] >= iv_threshold, "Feature"].tolist()
    dropped = iv_df.loc[iv_df["IV"] < iv_threshold, "Feature"].tolist()

    # ── Pretty-print ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  INFORMATION VALUE (IV) TABLE — Feature Selection")
    print("=" * 68)
    print(f"{'#':<4} {'Feature':<46} {'IV':>8}  {'Strength'}")
    print("-" * 68)
    for i, row in iv_df.iterrows():
        marker = "✗" if row["IV"] < iv_threshold else "✓"
        print(
            f" {marker}  {row['Feature']:<46} {row['IV']:>8.4f}  {row['Strength']}"
        )
    print("-" * 68)
    print(f"  Threshold: IV ≥ {iv_threshold}")
    print(f"  Selected : {len(selected)} features")
    if dropped:
        print(f"  Dropped  : {dropped}")
    print("=" * 68 + "\n")

    return iv_df, selected


# ──────────────────────────────────────────────────────────────────────
# 3.  MODELLING
# ──────────────────────────────────────────────────────────────────────


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with strict regularisation
    to prevent overfitting — suitable for credit-risk scorecards.
    """
    model = lgb.LGBMClassifier(
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        n_estimators=300,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    StandardScaler + Logistic Regression baseline pipeline.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# ──────────────────────────────────────────────────────────────────────
# 3b. PROBABILITY CALIBRATION (Bayes Prior Adjustment)
# ──────────────────────────────────────────────────────────────────────


def calibrate_probabilities(
    probs: np.ndarray,
    train_base_rate: float,
    balanced_rate: float = 0.5,
) -> np.ndarray:
    """
    Correct inflated probabilities from class_weight='balanced' back to
    the true population default rate using Bayes' prior adjustment.

    P_cal = (P_bal × r) / (P_bal × r + (1 − P_bal) × (1 − r))
    where r = train_base_rate / balanced_rate
    """
    r = train_base_rate / balanced_rate
    p_cal = (probs * r) / (probs * r + (1 - probs) * (1 - r))
    return p_cal


# ──────────────────────────────────────────────────────────────────────
# 3c. PDO CREDIT SCORE
# ──────────────────────────────────────────────────────────────────────


def pd_to_score(pd_calibrated: np.ndarray) -> np.ndarray:
    """
    Convert calibrated PD to a FICO/Findeks-style credit score
    using the Points-to-Double-the-Odds (PDO) methodology.
    Higher score = lower risk.
    """
    odds = (1 - pd_calibrated) / np.clip(pd_calibrated, 1e-15, None)
    score = OFFSET + FACTOR * np.log(odds)
    return np.clip(score, 300, 850).astype(int)


# ──────────────────────────────────────────────────────────────────────
# 3d. POPULATION STABILITY INDEX (PSI)
# ──────────────────────────────────────────────────────────────────────


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate the Population Stability Index between train (expected)
    and test (actual) score distributions.

    PSI < 0.10 → Stable
    PSI 0.10–0.25 → Slight shift
    PSI > 0.25 → Significant shift
    """
    # Use expected (train) quantiles as bin edges
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicate edges
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions with Laplace smoothing
    exp_pct = (exp_counts + 0.5) / (exp_counts.sum() + 0.5 * len(exp_counts))
    act_pct = (act_counts + 0.5) / (act_counts.sum() + 0.5 * len(act_counts))

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 6)


# ──────────────────────────────────────────────────────────────────────
# 4.  BANKING PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────


def evaluate_bank_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute banking-grade model evaluation metrics:
      • ROC-AUC
      • GINI coefficient  (2 × AUC − 1)
      • KS statistic      (max separation of cumulative distributions)
      • Brier Score        (calibration quality — lower is better)
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    # KS — Kolmogorov-Smirnov
    prob_event = y_prob[y_true == 1]
    prob_non_event = y_prob[y_true == 0]
    ks_stat, _ = stats.ks_2samp(prob_event, prob_non_event)

    # Brier Score
    brier = brier_score_loss(y_true, y_prob)

    return {
        "AUC": round(auc, 4),
        "GINI": round(gini, 4),
        "KS": round(ks_stat, 4),
        "BRIER": round(brier, 6),
    }


def print_metrics_table(metrics: dict) -> None:
    """Pretty-print a comparative Train vs Test table for both models."""
    print("\n" + "=" * 86)
    print("  MODEL PERFORMANCE — Banking Metrics (Train vs Test)")
    print("=" * 86)
    header = f"{'Model':<25} {'Set':<7} {'AUC':>8} {'GINI':>8} {'KS':>8} {'BRIER':>10}"
    print(header)
    print("-" * 86)

    for model_name, splits in metrics.items():
        for split_name, m in splits.items():
            print(
                f"  {model_name:<23} {split_name:<7} {m['AUC']:>8.4f} "
                f"{m['GINI']:>8.4f} {m['KS']:>8.4f} {m['BRIER']:>10.6f}"
            )
        print("-" * 86)

    # Overfit check — flag if Train GINI exceeds Test GINI by > 0.05
    for model_name, splits in metrics.items():
        gap = splits["Train"]["GINI"] - splits["Test"]["GINI"]
        status = "✓ PASS" if abs(gap) <= 0.05 else "⚠ WARNING"
        print(f"  {model_name:<23} GINI Gap (Train−Test): {gap:+.4f}  [{status}]")

    print("=" * 86 + "\n")


# ──────────────────────────────────────────────────────────────────────
# 5.  ARTIFACT SAVING
# ──────────────────────────────────────────────────────────────────────


def save_artifacts(
    lgbm_model: lgb.LGBMClassifier,
    lr_pipeline: Pipeline,
    selected_features: List[str],
    cleaning_params: dict,
    iv_df: pd.DataFrame,
    all_metrics: dict,
    X_train: pd.DataFrame,
    validation_curves: dict | None = None,
) -> None:
    """Persist every model artifact needed for deployment / Streamlit UI."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LightGBM model
    joblib.dump(lgbm_model, MODEL_DIR / "lgbm_model.pkl")

    # 2. Logistic Regression pipeline
    joblib.dump(lr_pipeline, MODEL_DIR / "lr_pipeline.pkl")

    # 3. Metadata (features, medians, clip limits, PDO params)
    metadata = {
        "selected_features": selected_features,
        "cleaning_params": cleaning_params,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "pdo_params": {
            "base_score": BASE_SCORE,
            "pdo": PDO,
            "factor": FACTOR,
            "offset": OFFSET,
            "base_odds": BASE_ODDS,
        },
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.pkl")

    # 4. IV table
    iv_df.to_csv(MODEL_DIR / "iv_table.csv", index=False)

    # 5. Metrics JSON
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # 6. SHAP background sample (300 rows, stratified if possible)
    n_sample = min(300, len(X_train))
    shap_bg = X_train.sample(n=n_sample, random_state=RANDOM_STATE)
    shap_bg.to_csv(MODEL_DIR / "shap_background.csv", index=False)

    # 7. Validation curves (calibration + cut-off analysis)
    if validation_curves is not None:
        with open(MODEL_DIR / "validation_curves.json", "w", encoding="utf-8") as f:
            json.dump(validation_curves, f, indent=2, ensure_ascii=False)

    print("[SAVE] All artifacts written to", MODEL_DIR.resolve())
    for f in sorted(MODEL_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"       └── {f.name:<28} ({size_kb:>8.1f} KB)")


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "▓" * 68)
    print("  CREDIT RISK SCORECARD — Training Pipeline")
    print("▓" * 68 + "\n")

    # ── 1. Load & Split (stratified) ──────────────────────────────
    df = load_raw_data(DATA_PATH)
    features = [c for c in df.columns if c != TARGET]

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(
        f"[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | "
        f"Default rate — Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}"
    )

    # ── 2. Compute cleaning params on train ONLY, then apply ─────
    cleaning_params = compute_cleaning_params(X_train)

    X_train = apply_cleaning(X_train, cleaning_params)
    X_test = apply_cleaning(X_test, cleaning_params)

    print("[CLEAN] Anomalies fixed, outliers clipped, missings imputed.")

    # ── 3. WoE / IV Feature Selection (train only) ───────────────
    iv_df, selected_features = run_iv_selection(
        X_train, y_train, features, iv_threshold=0.02
    )

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # ── 4. Model Training ────────────────────────────────────────
    print("[MODEL] Training LightGBM (regularised) …")
    lgbm_model = train_lgbm(X_train_sel, y_train)

    print("[MODEL] Training Logistic Regression baseline …")
    lr_pipeline = train_logistic_regression(X_train_sel, y_train)

    # ── 5. Evaluation ────────────────────────────────────────────
    all_metrics: dict = {}
    train_base_rate = float(y_train.mean())

    # LightGBM — raw probs → calibrated
    lgbm_train_prob_raw = lgbm_model.predict_proba(X_train_sel)[:, 1]
    lgbm_test_prob_raw = lgbm_model.predict_proba(X_test_sel)[:, 1]
    lgbm_train_prob = calibrate_probabilities(lgbm_train_prob_raw, train_base_rate)
    lgbm_test_prob = calibrate_probabilities(lgbm_test_prob_raw, train_base_rate)

    all_metrics["LightGBM"] = {
        "Train": evaluate_bank_metrics(y_train.values, lgbm_train_prob),
        "Test": evaluate_bank_metrics(y_test.values, lgbm_test_prob),
    }

    # Logistic Regression — raw probs → calibrated
    lr_train_prob_raw = lr_pipeline.predict_proba(X_train_sel)[:, 1]
    lr_test_prob_raw = lr_pipeline.predict_proba(X_test_sel)[:, 1]
    lr_train_prob = calibrate_probabilities(lr_train_prob_raw, train_base_rate)
    lr_test_prob = calibrate_probabilities(lr_test_prob_raw, train_base_rate)

    all_metrics["LogisticRegression"] = {
        "Train": evaluate_bank_metrics(y_train.values, lr_train_prob),
        "Test": evaluate_bank_metrics(y_test.values, lr_test_prob),
    }

    print_metrics_table(all_metrics)

    # ── 5b. PDO Credit Scores ────────────────────────────────────
    lgbm_test_scores = pd_to_score(lgbm_test_prob)
    lr_test_scores = pd_to_score(lr_test_prob)

    print("=" * 60)
    print("  PDO CREDIT SCORE DISTRIBUTION (Test Set)")
    print("=" * 60)
    for name, scores in [("LightGBM", lgbm_test_scores), ("LogReg", lr_test_scores)]:
        print(
            f"  {name:<12}  Mean: {np.mean(scores):>6.0f}  "
            f"Median: {np.median(scores):>6.0f}  "
            f"Min: {np.min(scores):>4d}  Max: {np.max(scores):>4d}"
        )
    print("=" * 60 + "\n")

    # ── 5c. PSI (Population Stability Index) ─────────────────────
    lgbm_train_scores = pd_to_score(lgbm_train_prob)
    lr_train_scores = pd_to_score(lr_train_prob)

    psi_lgbm = calculate_psi(lgbm_train_scores.astype(float), lgbm_test_scores.astype(float))
    psi_lr = calculate_psi(lr_train_scores.astype(float), lr_test_scores.astype(float))

    def _psi_label(v: float) -> str:
        if v < 0.10:
            return "✓ Stable"
        elif v < 0.25:
            return "~ Slight Shift"
        return "⚠ Significant Shift"

    print("=" * 60)
    print("  POPULATION STABILITY INDEX (PSI)")
    print("=" * 60)
    print(f"  LightGBM          PSI = {psi_lgbm:.6f}  [{_psi_label(psi_lgbm)}]")
    print(f"  LogisticRegression PSI = {psi_lr:.6f}  [{_psi_label(psi_lr)}]")
    print("=" * 60 + "\n")

    all_metrics["PSI"] = {"LightGBM": psi_lgbm, "LogisticRegression": psi_lr}

    # ── 5d. Calibration & Optimal Cut-off Curves ─────────────────
    validation_curves: dict = {}

    # Calibration table (10 bins on test set, LightGBM)
    cal_records = []
    test_df_cal = pd.DataFrame({"prob": lgbm_test_prob, "actual": y_test.values})
    test_df_cal["bin"] = pd.qcut(test_df_cal["prob"], q=10, duplicates="drop")
    for bin_label, grp in test_df_cal.groupby("bin", observed=False):
        cal_records.append({
            "bin": str(bin_label),
            "count": int(len(grp)),
            "mean_predicted_pd": round(float(grp["prob"].mean()), 6),
            "actual_default_rate": round(float(grp["actual"].mean()), 6),
        })
    validation_curves["calibration_table"] = cal_records

    # Cut-off analysis (PD thresholds from 0.02 to 0.50)
    cutoff_records = []
    total_n = len(y_test)
    total_bads = int(y_test.sum())
    for threshold in np.arange(0.02, 0.52, 0.02):
        approved = lgbm_test_prob <= threshold
        n_approved = int(approved.sum())
        approval_rate = round(n_approved / total_n, 4)
        bads_in_approved = int(y_test.values[approved].sum())
        bad_capture = round(1 - bads_in_approved / max(total_bads, 1), 4)
        # Simplified P&L: each good loan earns +1 unit, each bad costs -10 units
        expected_pl = (n_approved - bads_in_approved) * 1 + bads_in_approved * (-10)
        cutoff_records.append({
            "pd_threshold": round(float(threshold), 2),
            "n_approved": n_approved,
            "approval_rate": approval_rate,
            "bad_capture_rate": bad_capture,
            "expected_pl_units": int(expected_pl),
        })
    validation_curves["cutoff_analysis"] = cutoff_records

    # ── 6. Save Artifacts ────────────────────────────────────────
    save_artifacts(
        lgbm_model=lgbm_model,
        lr_pipeline=lr_pipeline,
        selected_features=selected_features,
        cleaning_params=cleaning_params,
        iv_df=iv_df,
        all_metrics=all_metrics,
        X_train=X_train_sel,
        validation_curves=validation_curves,
    )

    print("\n✅  Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
