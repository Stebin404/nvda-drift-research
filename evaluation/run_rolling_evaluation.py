"""
evaluation/run_rolling_evaluation.py

The main entry point tying together rolling-origin folds, the
SEC-verified earnings ground truth, and the three detectors. Produces
aggregated (mean +/- std) precision/recall/delay across folds instead
of a single-run point estimate.
"""

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target

from models.lightgbm_model import build_lightgbm_model, FEATURES

from drift.adwin_detector import run_adwin
from drift.page_hinkley_detector import run_page_hinkley
from drift.kswin_detector import run_kswin

from evaluation.rolling_origin import generate_rolling_folds, run_rolling_fold
from evaluation.ground_truth_external import get_earnings_ground_truth_for_test_fold
from evaluation.detector_metrics import evaluate_detector


DETECTORS = {
    "ADWIN": lambda stream: run_adwin(stream),
    "Page-Hinkley": lambda stream: run_page_hinkley(stream),
    "KSWIN": lambda stream: run_kswin(stream),
}


def run_full_rolling_evaluation(
    ticker="NVDA",
    n_folds=5,
    min_train_fraction=0.5,
    test_fraction=0.08,
    tolerance=10,
    earnings_csv_path="data/earnings_dates_FINAL.csv",
):
    """
    tolerance default = 10 trading days (~2 weeks). With earnings-based
    ground truth (4 events/year) and test windows of ~150 rows, a
    looser tolerance (e.g. 30) risks "matching" an alarm to an earnings
    event from an unrelated price move weeks away. Report results
    across a couple of tolerance values as a sensitivity check rather
    than trusting one silently.
    """
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

    results = {name: {"per_fold": []} for name in DETECTORS}

    for fold in folds:
        fold_result = run_rolling_fold(full_df, fold, build_lightgbm_model, FEATURES)
        error_stream = fold_result["error_stream"]

        ground_truth = get_earnings_ground_truth_for_test_fold(
            full_df,
            test_fold_start_index=fold["test_start"],
            test_fold_end_index=fold["test_end"],
            csv_path=earnings_csv_path,
        )

        for detector_name, detector_fn in DETECTORS.items():
            alarms = detector_fn(error_stream)
            metrics = evaluate_detector(alarms, ground_truth, tolerance=tolerance)

            results[detector_name]["per_fold"].append({
                "fold_id": fold["fold_id"],
                "n_alarms": len(alarms),
                "n_ground_truth": len(ground_truth),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "avg_delay": metrics["avg_delay"],
            })

    for detector_name in DETECTORS:
        per_fold = results[detector_name]["per_fold"]

        # Folds with zero ground-truth events are excluded from the
        # recall aggregate. evaluate_detector() returns recall=0.0 for
        # an empty ground truth, but recall is actually UNDEFINED there
        # (nothing to recall) -- including it would silently drag the
        # mean down for a detector that did nothing wrong.
        folds_with_gt = [f for f in per_fold if f["n_ground_truth"] > 0]

        precisions = [f["precision"] for f in per_fold]
        recalls = [f["recall"] for f in folds_with_gt]
        delays = [f["avg_delay"] for f in per_fold if f["avg_delay"] is not None]

        results[detector_name]["aggregate"] = {
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)) if recalls else None,
            "recall_std": float(np.std(recalls)) if recalls else None,
            "recall_n_folds_used": len(recalls),
            "avg_delay_mean": float(np.mean(delays)) if delays else None,
            "avg_delay_std": float(np.std(delays)) if delays else None,
            "folds_with_no_alarms": sum(1 for f in per_fold if f["n_alarms"] == 0),
            "folds_with_no_ground_truth": sum(1 for f in per_fold if f["n_ground_truth"] == 0),
        }

    return results


def print_results(results):
    print("\nROLLING-ORIGIN DETECTOR EVALUATION (earnings ground truth)")
    print("=" * 70)

    for detector_name, data in results.items():
        print(f"\n{detector_name}")
        print("-" * 40)

        for fold in data["per_fold"]:
            print(
                f"  fold {fold['fold_id']}: alarms={fold['n_alarms']:3d}  "
                f"gt={fold['n_ground_truth']}  "
                f"precision={fold['precision']:.3f}  "
                f"recall={fold['recall']:.3f}  "
                f"delay={fold['avg_delay']}"
            )

        agg = data["aggregate"]
        recall_str = (
            f"{agg['recall_mean']:.3f} (+/-{agg['recall_std']:.3f}, n={agg['recall_n_folds_used']} folds)"
            if agg["recall_mean"] is not None
            else "undefined (no folds had ground truth)"
        )
        print(
            f"  AGGREGATE: precision={agg['precision_mean']:.3f} (+/-{agg['precision_std']:.3f})  "
            f"recall={recall_str}  delay={agg['avg_delay_mean']}"
        )


if __name__ == "__main__":
    results = run_full_rolling_evaluation()
    print_results(results)