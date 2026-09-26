"""
evaluation/kswin_warmstart_experiment.py

TODO item #4: compares KSWIN's fold-based tournament performance under
three regimes on the SAME folds and SAME earnings ground truth:

  1. "reinit"    -- a fresh KSWIN instance per fold, seeing only the
                     test-fold stream (the existing behaviour of
                     evaluation/run_rolling_evaluation.py and
                     evaluation/bridge_experiment.py).
  2. "warmstart" -- a fresh KSWIN instance per fold, but first warmed
                     up (update-only, no alarms recorded) on that
                     fold's own training-window stream -- data the
                     fold's model was already allowed to see -- before
                     scoring alarms on the test window
                     (drift.kswin_detector.run_kswin_seeded).
  3. "continuous"-- one persistent KSWIN instance run across the ENTIRE
                     price history with no fold boundaries at all
                     (Section IV-E's existing full-history result,
                     reproduced here for a side-by-side comparison
                     rather than only quoted from a separate script).

Run for both the LightGBM residual stream and the raw Volatility_20
stream, at tolerance=30 (causal), matching Tables I/II.
"""

import json

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES

from drift.kswin_detector import run_kswin, run_kswin_seeded

from evaluation.rolling_origin import generate_rolling_folds, HORIZON
from evaluation.ground_truth_external import (
    get_earnings_ground_truth_for_test_fold,
    get_earnings_ground_truth,
)
from evaluation.detector_metrics import evaluate_detector

TICKERS = {
    "NVDA": "data/earnings_dates_FINAL.csv",
    "AMD": "data/earnings_dates_FINAL_amd.csv",
    "TSLA": "data/earnings_dates_FINAL_tsla.csv",
}

N_FOLDS = 5
MIN_TRAIN_FRACTION = 0.5
TEST_FRACTION = 0.08
TOLERANCE = 30
ONE_SIDED = True


def _rebuild_dataset(ticker):
    df = load_stock_data(ticker)
    df = create_features(df)
    df = create_target(df)
    return df.reset_index(drop=True)


def _residual_train_test_streams(df, fold):
    """In-sample (training-window) and out-of-sample (test-window)
    |error| streams from the SAME fold model, using the SAME purge-gap
    boundary as evaluation/rolling_origin.py's run_rolling_fold."""
    fold_df = df.iloc[:fold["test_end"]].reset_index(drop=True)
    X = fold_df[FEATURES]
    y = fold_df["Target"]

    purge_end = fold["train_end"] - HORIZON
    X_train = X.iloc[:purge_end]
    y_train = y.iloc[:purge_end]
    X_test = X.iloc[fold["test_start"]:fold["test_end"]]
    y_test = y.iloc[fold["test_start"]:fold["test_end"]]

    model = build_lightgbm_model()
    model.fit(X_train, y_train)

    train_stream = np.abs(y_train.values - model.predict(X_train))
    test_stream = np.abs(y_test.values - model.predict(X_test))
    return train_stream, test_stream


def _volatility_train_test_streams(df, fold):
    """Raw Volatility_20: everything before this fold's test window is
    'training window' data (no purge gap needed -- it is not a target
    that leaks future prices)."""
    train_stream = df["Volatility_20"].iloc[:fold["test_start"]].reset_index(drop=True).values
    test_stream = df["Volatility_20"].iloc[fold["test_start"]:fold["test_end"]].reset_index(drop=True).values
    return train_stream, test_stream


def _aggregate(per_fold):
    precisions = [f["precision"] for f in per_fold]
    folds_with_gt = [f for f in per_fold if f["n_gt"] > 0]
    recalls = [f["recall"] for f in folds_with_gt]
    delays = [f["avg_delay"] for f in per_fold if f["avg_delay"] is not None]
    total_alarms = sum(f["n_alarms"] for f in per_fold)
    return {
        "total_alarms": total_alarms,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)) if recalls else None,
        "delay_mean": float(np.mean(delays)) if delays else None,
    }


def run_comparison(ticker, csv_path, stream_kind):
    """stream_kind in {'residual', 'volatility'}."""
    df = _rebuild_dataset(ticker)
    folds = generate_rolling_folds(
        len(df), n_folds=N_FOLDS,
        min_train_fraction=MIN_TRAIN_FRACTION, test_fraction=TEST_FRACTION,
    )

    reinit_per_fold, warmstart_per_fold = [], []

    for fold in folds:
        gt = get_earnings_ground_truth_for_test_fold(
            df, fold["test_start"], fold["test_end"], csv_path=csv_path,
        )

        if stream_kind == "residual":
            train_stream, test_stream = _residual_train_test_streams(df, fold)
        else:
            train_stream, test_stream = _volatility_train_test_streams(df, fold)

        reinit_alarms = run_kswin(test_stream)
        warmstart_alarms = run_kswin_seeded(train_stream, test_stream)

        reinit_metrics = evaluate_detector(reinit_alarms, gt, tolerance=TOLERANCE, one_sided=ONE_SIDED)
        warmstart_metrics = evaluate_detector(warmstart_alarms, gt, tolerance=TOLERANCE, one_sided=ONE_SIDED)

        reinit_per_fold.append({"n_alarms": len(reinit_alarms), "n_gt": len(gt), **reinit_metrics})
        warmstart_per_fold.append({"n_alarms": len(warmstart_alarms), "n_gt": len(gt), **warmstart_metrics})

    # Continuous, full-history reference (only meaningful/available for
    # the raw-volatility stream -- there is no single "continuous
    # LightGBM residual stream" since the model itself is refit per
    # fold; Section IV-E's continuous result was always Volatility_20-only).
    continuous = None
    if stream_kind == "volatility":
        full_volatility_stream = df["Volatility_20"].values
        continuous_alarms = run_kswin(full_volatility_stream)
        continuous_gt = get_earnings_ground_truth(df, csv_path=csv_path)
        continuous_metrics = evaluate_detector(continuous_alarms, continuous_gt, tolerance=TOLERANCE, one_sided=ONE_SIDED)
        continuous = {
            "total_alarms": len(continuous_alarms),
            "precision_macro": continuous_metrics["precision"],
            "recall_macro": continuous_metrics["recall"],
            "delay_mean": continuous_metrics["avg_delay"],
        }

    return {
        "reinit": _aggregate(reinit_per_fold),
        "warmstart": _aggregate(warmstart_per_fold),
        "continuous": continuous,
    }


def print_comparison(ticker, stream_kind, result):
    print(f"\n=== {ticker} / {stream_kind}: KSWIN reinit vs warmstart vs continuous (tol={TOLERANCE}, causal) ===")
    for regime in ("reinit", "warmstart", "continuous"):
        r = result[regime]
        if r is None:
            print(f"  {regime:12s} n/a")
            continue
        recall = r["recall_macro"] if r["recall_macro"] is not None else float("nan")
        delay = r["delay_mean"] if r["delay_mean"] is not None else float("nan")
        print(f"  {regime:12s} alarms={r['total_alarms']:3d}  precision={r['precision_macro']:.3f}  recall={recall:.3f}  delay={delay:.1f}")


if __name__ == "__main__":
    all_out = {}
    for ticker, csv_path in TICKERS.items():
        all_out[ticker] = {}
        for stream_kind in ("residual", "volatility"):
            result = run_comparison(ticker, csv_path, stream_kind)
            print_comparison(ticker, stream_kind, result)
            all_out[ticker][stream_kind] = result

    with open("kswin_warmstart_results.json", "w") as f:
        json.dump(all_out, f, indent=2, default=str)

    print("\nFull results written to kswin_warmstart_results.json")
