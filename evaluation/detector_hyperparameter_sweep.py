"""
evaluation/detector_hyperparameter_sweep.py

REVIEWER GAP ADDRESSED: ADWIN and Page-Hinkley are run with a single
library-default configuration on the real NVDA/AMD/TSLA streams
(evaluation/run_rolling_evaluation.py, evaluation/bridge_experiment.py).
The paper originally interpreted "zero alarms, ever" as evidence about
detector *class* behavior on this kind of data -- but that is not
distinguishable from "the default threshold happens to be insensitive
for this particular stream's scale" without actually sweeping the
hyperparameters. benchmarks/run_benchmark.py already does exactly this
kind of check for Page-Hinkley's threshold on SYNTHETIC data (finding
the default threshold is miscalibrated for high-magnitude streams).
This module runs the equivalent sweep on the REAL residual and raw
raw-volatility streams, across all three tickers, so the "zero alarms"
finding is something we checked, not something we assumed.

Grids:
  ADWIN: delta over two orders of magnitude around the library default
         (0.002).
  Page-Hinkley: threshold, min_instances, and delta each swept around
         their library defaults (50, 30, 0.005) one at a time, holding
         the others fixed, plus a magnitude-scaled threshold variant
         (threshold = 50 * stream std) mirroring the fix already used
         in the synthetic benchmark.

This module does not change any existing detector default or any
already-reported table -- it is a new, additional experiment.
"""

import json

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES

from drift.adwin_detector import run_adwin
from drift.page_hinkley_detector import run_page_hinkley

from evaluation.rolling_origin import generate_rolling_folds, run_rolling_fold
from evaluation.ground_truth_external import get_earnings_ground_truth_for_test_fold
from evaluation.detector_metrics import evaluate_detector

TICKERS = {
    "NVDA": "data/earnings_dates_FINAL.csv",
    "AMD": "data/earnings_dates_FINAL_amd.csv",
    "TSLA": "data/earnings_dates_FINAL_tsla.csv",
}

ADWIN_DELTA_GRID = [0.0002, 0.001, 0.002, 0.01, 0.05]
PH_THRESHOLD_GRID = [5, 20, 50, 150, 400]
PH_MIN_INSTANCES_GRID = [10, 30, 60]
PH_DELTA_GRID = [0.001, 0.005, 0.02]

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


def _folds_and_ground_truth(df, earnings_csv_path):
    folds = generate_rolling_folds(
        len(df), n_folds=N_FOLDS,
        min_train_fraction=MIN_TRAIN_FRACTION, test_fraction=TEST_FRACTION,
    )
    fold_data = []
    for fold in folds:
        gt = get_earnings_ground_truth_for_test_fold(
            df, fold["test_start"], fold["test_end"], csv_path=earnings_csv_path,
        )
        fold_data.append((fold, gt))
    return fold_data


def sweep_on_streams(streams_by_fold, ground_truth_by_fold, detector_fn):
    per_fold = []
    for stream, gt in zip(streams_by_fold, ground_truth_by_fold):
        alarms = detector_fn(stream)
        metrics = evaluate_detector(alarms, gt, tolerance=TOLERANCE, one_sided=ONE_SIDED)
        per_fold.append({"n_alarms": len(alarms), "n_gt": len(gt), **metrics})

    total_alarms = sum(f["n_alarms"] for f in per_fold)
    precisions = [f["precision"] for f in per_fold]
    folds_with_gt = [f for f in per_fold if f["n_gt"] > 0]
    recalls = [f["recall"] for f in folds_with_gt]

    return {
        "total_alarms": total_alarms,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)) if recalls else None,
        "per_fold": per_fold,
    }


def run_sweep(ticker, earnings_csv_path):
    df = _rebuild_dataset(ticker)
    fold_data = _folds_and_ground_truth(df, earnings_csv_path)
    folds = [f for f, _ in fold_data]
    ground_truth_by_fold = [gt for _, gt in fold_data]

    residual_by_fold = []
    for fold in folds:
        fold_result = run_rolling_fold(df, fold, build_lightgbm_model, FEATURES)
        residual_by_fold.append(fold_result["error_stream"])

    volatility_by_fold = [
        df["Volatility_20"].iloc[fold["test_start"]:fold["test_end"]].reset_index(drop=True).values
        for fold in folds
    ]

    results = {
        "ADWIN": {"residual": [], "volatility": []},
        "Page-Hinkley": {"residual": [], "volatility": []},
    }

    for delta in ADWIN_DELTA_GRID:
        fn = lambda s, d=delta: run_adwin(s, delta=d)
        results["ADWIN"]["residual"].append(
            [f"delta={delta}", sweep_on_streams(residual_by_fold, ground_truth_by_fold, fn)])
        results["ADWIN"]["volatility"].append(
            [f"delta={delta}", sweep_on_streams(volatility_by_fold, ground_truth_by_fold, fn)])

    for threshold in PH_THRESHOLD_GRID:
        fn = lambda s, t=threshold: run_page_hinkley(s, threshold=t)
        results["Page-Hinkley"]["residual"].append(
            [f"threshold={threshold}", sweep_on_streams(residual_by_fold, ground_truth_by_fold, fn)])
        results["Page-Hinkley"]["volatility"].append(
            [f"threshold={threshold}", sweep_on_streams(volatility_by_fold, ground_truth_by_fold, fn)])

    for min_inst in PH_MIN_INSTANCES_GRID:
        fn = lambda s, m=min_inst: run_page_hinkley(s, min_instances=m)
        results["Page-Hinkley"]["residual"].append(
            [f"min_instances={min_inst}", sweep_on_streams(residual_by_fold, ground_truth_by_fold, fn)])
        results["Page-Hinkley"]["volatility"].append(
            [f"min_instances={min_inst}", sweep_on_streams(volatility_by_fold, ground_truth_by_fold, fn)])

    for delta in PH_DELTA_GRID:
        fn = lambda s, d=delta: run_page_hinkley(s, delta=d)
        results["Page-Hinkley"]["residual"].append(
            [f"delta={delta}", sweep_on_streams(residual_by_fold, ground_truth_by_fold, fn)])
        results["Page-Hinkley"]["volatility"].append(
            [f"delta={delta}", sweep_on_streams(volatility_by_fold, ground_truth_by_fold, fn)])

    def ph_scaled(stream):
        std = np.std(stream)
        thr = 50 * std if std > 0 else 50
        return run_page_hinkley(stream, threshold=thr)

    results["Page-Hinkley"]["residual"].append(
        ["scaled=50*std", sweep_on_streams(residual_by_fold, ground_truth_by_fold, ph_scaled)])
    results["Page-Hinkley"]["volatility"].append(
        ["scaled=50*std", sweep_on_streams(volatility_by_fold, ground_truth_by_fold, ph_scaled)])

    return results


def print_sweep(ticker, results):
    print(f"\n=== {ticker}: hyperparameter sensitivity sweep (tolerance={TOLERANCE}, causal) ===")
    for detector_name, streams in results.items():
        for stream_name, entries in streams.items():
            print(f"\n-- {detector_name} on {stream_name} --")
            for param, res in entries:
                recall = res["recall_macro"]
                recall_str = f"{recall:.3f}" if recall is not None else "undef"
                print(
                    f"  {param:20s} total_alarms={res['total_alarms']:3d}  "
                    f"precision_macro={res['precision_macro']:.3f}  "
                    f"recall_macro={recall_str}"
                )


if __name__ == "__main__":
    all_results = {}
    for ticker, csv_path in TICKERS.items():
        results = run_sweep(ticker, csv_path)
        print_sweep(ticker, results)
        all_results[ticker] = results

    with open("sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nFull results written to sweep_results.json")
