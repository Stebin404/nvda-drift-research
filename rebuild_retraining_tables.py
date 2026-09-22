"""
rebuild_retraining_tables.py

Reproduces Table IV (single 400-row evaluation window, expanding vs.
sliding training window) and Table V (3 independent 300-row windows)
from the paper, extended with the forecast-model baselines from
models/baseline_models.py: zero-return, historical-mean, naive
persistence, and linear/AR.

This does not modify retraining_engine.py or
evaluation/rolling_retraining_evaluation.py -- every baseline plugs
into the existing walk_forward_predict / compare_retraining_strategies
/ run_rolling_retraining_evaluation machinery via build_model_fn.

Run from the repo root:  python rebuild_retraining_tables.py
"""

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from models.baseline_models import BASELINE_REGISTRY
from retraining.retraining_engine import (
    compare_retraining_strategies,
    walk_forward_predict,
)
from evaluation.rolling_retraining_evaluation import generate_retraining_windows


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# Table IV -- single 400-row window
# ---------------------------------------------------------------------

def table_iv(df):
    print_header("TABLE IV -- Single-Window Retraining Comparison (400-row eval window)")

    EVAL_WINDOW = 400
    initial_train_size = len(df) - EVAL_WINDOW
    TRAIN_WINDOW_SIZE = 600  # matches test_retraining_sliding.py

    print(f"Total rows: {len(df)}, initial_train_size: {initial_train_size}, "
          f"evaluating last {EVAL_WINDOW} rows\n")

    # --- LightGBM, expanding and sliding (unchanged from the paper) ---
    exp_results = compare_retraining_strategies(
        df, FEATURES, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
    )
    slid_results = compare_retraining_strategies(
        df, FEATURES, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
        train_window_size=TRAIN_WINDOW_SIZE,
    )

    rows = []
    for strategy in ("never", "fixed", "drift_triggered"):
        rows.append((
            f"LightGBM ({strategy})",
            exp_results[strategy]["mae"],
            slid_results[strategy]["mae"],
            exp_results[strategy]["n_retrains"],
        ))

    # --- Baselines, each under its recommended strategy ---
    for name, (build_fn, strategy, retrain_interval) in BASELINE_REGISTRY.items():
        kwargs = dict(
            df=df, feature_columns=FEATURES, target_column="Target",
            build_model_fn=build_fn, initial_train_size=initial_train_size,
            strategy=strategy,
        )
        if retrain_interval is not None:
            kwargs["retrain_interval"] = retrain_interval

        exp_r = walk_forward_predict(**kwargs)
        slid_r = walk_forward_predict(**kwargs, train_window_size=TRAIN_WINDOW_SIZE)

        rows.append((
            f"{name} ({strategy})",
            exp_r["mae"], slid_r["mae"], exp_r["n_retrains"],
        ))

    print(f"{'Strategy':32s} {'MAE (exp.)':>12s} {'MAE (slid.)':>12s} {'Retrains':>10s}")
    print("-" * 68)
    for label, mae_exp, mae_slid, n_retrain in rows:
        print(f"{label:32s} {mae_exp:12.4f} {mae_slid:12.4f} {n_retrain:10d}")

    return rows


# ---------------------------------------------------------------------
# Table V -- 3 independent 300-row windows
# ---------------------------------------------------------------------

def run_rolling_baseline(full_df, build_fn, strategy, retrain_interval,
                          eval_window_size=300, min_initial_train=900, n_windows=3):
    """
    Same window generation as
    evaluation.rolling_retraining_evaluation.run_rolling_retraining_evaluation,
    parameterized over an arbitrary build_model_fn / strategy instead
    of being hardcoded to LightGBM's 3-way comparison.
    """
    windows = generate_retraining_windows(
        n_rows=len(full_df), eval_window_size=eval_window_size,
        min_initial_train=min_initial_train, n_windows=n_windows,
    )

    per_window = []
    for window in windows:
        window_df = full_df.iloc[:window["eval_end"]].reset_index(drop=True)
        kwargs = dict(
            df=window_df, feature_columns=FEATURES, target_column="Target",
            build_model_fn=build_fn, initial_train_size=window["initial_train_size"],
            strategy=strategy,
        )
        if retrain_interval is not None:
            kwargs["retrain_interval"] = retrain_interval
        result = walk_forward_predict(**kwargs)
        per_window.append({
            "window_id": window["window_id"],
            "mae": result["mae"],
            "n_retrains": result["n_retrains"],
        })
    return per_window


def table_v(full_df):
    print_header("TABLE V -- Multi-Window Retraining Comparison (3 independent 300-row windows)")

    from evaluation.rolling_retraining_evaluation import run_rolling_retraining_evaluation
    lgbm_results = run_rolling_retraining_evaluation(
        eval_window_size=300, min_initial_train=900, n_windows=3, fixed_interval=30,
    )

    all_series = {}
    for strategy in ("never", "fixed", "drift_triggered"):
        all_series[f"LightGBM ({strategy})"] = [w["mae"] for w in lgbm_results[strategy]]

    for name, (build_fn, strategy, retrain_interval) in BASELINE_REGISTRY.items():
        per_window = run_rolling_baseline(full_df, build_fn, strategy, retrain_interval)
        all_series[f"{name} ({strategy})"] = [w["mae"] for w in per_window]

    n_windows = len(next(iter(all_series.values())))
    header = f"{'Model':32s} " + " ".join(f"W{i:>8d}" for i in range(n_windows)) + f" {'Mean':>10s} {'Std':>8s}"
    print(header)
    print("-" * len(header))
    for label, maes in all_series.items():
        row = f"{label:32s} " + " ".join(f"{m:9.4f}" for m in maes)
        row += f" {np.mean(maes):10.4f} {np.std(maes):8.4f}"
        print(row)

    print("\nWINS PER WINDOW (lowest MAE across all models, including baselines):")
    for w_id in range(n_windows):
        winner = min(all_series, key=lambda k: all_series[k][w_id])
        print(f"  window {w_id}: {winner} (MAE={all_series[winner][w_id]:.4f})")

    return all_series


if __name__ == "__main__":
    df = load_stock_data("NVDA")
    df = create_features(df)
    df = create_target(df)
    df = df.reset_index(drop=True)

    table_iv(df)
    table_v(df)