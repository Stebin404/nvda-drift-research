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
from evaluation.detector_metrics import evaluate_detector, permutation_null_precision


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
    one_sided=True,
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
            metrics = evaluate_detector(alarms, ground_truth, tolerance=tolerance, one_sided=one_sided)

            # Permutation null: how well would THIS SAME alarm list
            # "detect" a random pseudo-event calendar of the same size,
            # drawn from the same test-fold span? Answers Reviewer 3's
            # point 4/17 ask directly. Only defined when both alarms
            # and ground truth are non-empty and there's a nontrivial
            # span to draw from -- an empty alarm list gives a
            # degenerate (undefined) null.
            null_result = None
            if alarms and ground_truth:
                fold_span = fold["test_end"] - fold["test_start"]
                null_result = permutation_null_precision(
                    alarms,
                    n_ground_truth=len(ground_truth),
                    fold_start=0,
                    fold_end=fold_span,
                    tolerance=tolerance,
                    one_sided=one_sided,
                    n_trials=1000,
                    seed=42,
                )

            results[detector_name]["per_fold"].append({
                "fold_id": fold["fold_id"],
                "n_alarms": len(alarms),
                "n_ground_truth": len(ground_truth),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "avg_delay": metrics["avg_delay"],
                "matched_alarms": metrics["matched_alarms"],
                "matched_truth": metrics["matched_truth"],
                "null_precision_mean": null_result["precision_mean"] if null_result else None,
                "null_precision_p95": null_result["precision_p95"] if null_result else None,
                "null_recall_mean": null_result["recall_mean"] if null_result else None,
                "null_recall_p95": null_result["recall_p95"] if null_result else None,
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

        null_precisions = [f["null_precision_mean"] for f in per_fold if f["null_precision_mean"] is not None]
        null_precision_p95s = [f["null_precision_p95"] for f in per_fold if f["null_precision_p95"] is not None]
        null_recalls = [f["null_recall_mean"] for f in per_fold if f["null_recall_mean"] is not None]

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
            # Chance baseline: mean-of-per-fold-null-means is a coarser
            # summary than pooling all 1000*n_folds trials, but keeps
            # each fold's own alarm count/span fixed rather than mixing
            # them, matching how precision/recall above are aggregated.
            "null_precision_mean": float(np.mean(null_precisions)) if null_precisions else None,
            "null_precision_p95_mean": float(np.mean(null_precision_p95s)) if null_precision_p95s else None,
            "null_recall_mean": float(np.mean(null_recalls)) if null_recalls else None,
            "null_n_folds_used": len(null_precisions),
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
        if agg["null_precision_mean"] is not None:
            print(
                f"  CHANCE BASELINE (permuted event calendars, n=1000/fold, "
                f"{agg['null_n_folds_used']} folds): "
                f"precision_null_mean={agg['null_precision_mean']:.3f} "
                f"(p95={agg['null_precision_p95_mean']:.3f})  "
                f"recall_null_mean={agg['null_recall_mean']:.3f}"
            )


if __name__ == "__main__":
    results = run_full_rolling_evaluation()
    print_results(results)