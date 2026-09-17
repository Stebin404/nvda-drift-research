"""
evaluation/bridge_experiment.py

Runs ADWIN, Page-Hinkley, and KSWIN directly on the raw feature
streams (Volatility_20, Log_Return) rather than on the LightGBM
residual stream, using the SAME rolling-origin folds and the SAME
earnings ground truth as evaluation/run_rolling_evaluation.py.

This isolates whether the detection pattern found on model residuals
(KSWIN modestly successful with ~14-day delay, ADWIN/PH silent) is a
genuine property of NVDA's market behavior, or an artifact of how
LightGBM's prediction errors happen to accumulate after an earnings
event. No model training is needed for this experiment -- detectors
run directly on the feature values for each fold's test window.
"""

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target

from drift.adwin_detector import run_adwin
from drift.page_hinkley_detector import run_page_hinkley
from drift.kswin_detector import run_kswin

from evaluation.rolling_origin import generate_rolling_folds
from evaluation.ground_truth_external import get_earnings_ground_truth_for_test_fold
from evaluation.detector_metrics import evaluate_detector


FEATURE_STREAMS_TO_TEST = ["Volatility_20", "Log_Return"]

DETECTORS = {
    "ADWIN": lambda stream: run_adwin(stream),
    "Page-Hinkley": lambda stream: run_page_hinkley(stream),
    "KSWIN": lambda stream: run_kswin(stream),
}


def run_bridge_experiment(
    ticker="NVDA",
    n_folds=5,
    min_train_fraction=0.5,
    test_fraction=0.08,
    tolerance=30,
    one_sided=True,
    earnings_csv_path="data/earnings_dates_FINAL.csv",
):
    full_df = load_stock_data(ticker)
    full_df = create_features(full_df)
    full_df = create_target(full_df)
    full_df = full_df.reset_index(drop=True)

    folds = generate_rolling_folds(
        n_rows=len(full_df),
        n_folds=n_folds,
        min_train_fraction=min_train_fraction,
        test_fraction=test_fraction,
    )

    results = {
        feature_name: {detector_name: {"per_fold": []} for detector_name in DETECTORS}
        for feature_name in FEATURE_STREAMS_TO_TEST
    }

    for fold in folds:
        ground_truth = get_earnings_ground_truth_for_test_fold(
            full_df,
            test_fold_start_index=fold["test_start"],
            test_fold_end_index=fold["test_end"],
            csv_path=earnings_csv_path,
        )

        for feature_name in FEATURE_STREAMS_TO_TEST:
            stream = (
                full_df[feature_name]
                .iloc[fold["test_start"]:fold["test_end"]]
                .reset_index(drop=True)
                .values
            )

            for detector_name, detector_fn in DETECTORS.items():
                alarms = detector_fn(stream)
                metrics = evaluate_detector(alarms, ground_truth, tolerance=tolerance, one_sided=one_sided)

                results[feature_name][detector_name]["per_fold"].append({
                    "fold_id": fold["fold_id"],
                    "n_alarms": len(alarms),
                    "n_ground_truth": len(ground_truth),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "avg_delay": metrics["avg_delay"],
                    "matched_alarms": metrics["matched_alarms"],
                    "matched_truth": metrics["matched_truth"],
                })

    for feature_name in FEATURE_STREAMS_TO_TEST:
        for detector_name in DETECTORS:
            per_fold = results[feature_name][detector_name]["per_fold"]
            folds_with_gt = [f for f in per_fold if f["n_ground_truth"] > 0]

            precisions = [f["precision"] for f in per_fold]
            recalls = [f["recall"] for f in folds_with_gt]
            delays = [f["avg_delay"] for f in per_fold if f["avg_delay"] is not None]

            results[feature_name][detector_name]["aggregate"] = {
                "precision_mean": float(np.mean(precisions)),
                "precision_std": float(np.std(precisions)),
                "recall_mean": float(np.mean(recalls)) if recalls else None,
                "recall_std": float(np.std(recalls)) if recalls else None,
                "recall_n_folds_used": len(recalls),
                "avg_delay_mean": float(np.mean(delays)) if delays else None,
            }

    return results


def print_bridge_results(results):
    print("\nBRIDGE EXPERIMENT: detectors on RAW FEATURE STREAMS")
    print("=" * 70)

    for feature_name, detector_results in results.items():
        print(f"\n--- Feature: {feature_name} ---")

        for detector_name, data in detector_results.items():
            agg = data["aggregate"]
            recall_str = (
                f"{agg['recall_mean']:.3f} (n={agg['recall_n_folds_used']} folds)"
                if agg["recall_mean"] is not None
                else "undefined"
            )
            print(
                f"  {detector_name:14s} precision={agg['precision_mean']:.3f}  "
                f"recall={recall_str}  delay={agg['avg_delay_mean']}"
            )


if __name__ == "__main__":
    results = run_bridge_experiment()
    print_bridge_results(results)