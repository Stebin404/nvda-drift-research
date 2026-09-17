"""
rebuild_tables.py

Runs the full one-sided-vs-symmetric sweep and prints paper-ready
table text for Table I (residual stream), Table II (bridge/raw
features), and Table VI (continuous full-history). Run this from the
repo root AFTER dropping in the patched files.

Usage:
    python rebuild_tables.py
"""

import io
import contextlib

from data.loader import load_stock_data
from features.engineer import create_features
from drift.kswin_detector import run_kswin
from evaluation.ground_truth_external import get_earnings_ground_truth
from evaluation.detector_metrics import evaluate_detector
from evaluation.run_rolling_evaluation import run_full_rolling_evaluation
from evaluation.bridge_experiment import run_bridge_experiment
from evaluation.confidence_intervals import pooled_across_folds, format_rate_with_ci


def silent(fn, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(**kwargs)
    return result


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def table_i():
    print_header("TABLE I -- KSWIN on LightGBM residual stream, by tolerance")
    print(f"{'tol':>4} {'mode':>10} | {'precision':38} | {'recall':38}")
    for tol in [10, 15, 20, 25, 30]:
        for one_sided in [True, False]:
            results = silent(run_full_rolling_evaluation, tolerance=tol, one_sided=one_sided)
            pooled = pooled_across_folds(results["KSWIN"]["per_fold"])
            mode = "one-sided" if one_sided else "symmetric"
            print(f"{tol:>4} {mode:>10} | {pooled['pooled_precision_str']:38} | {pooled['pooled_recall_str']:38}")


def table_ii():
    print_header("TABLE II -- KSWIN by signal type, tolerance=30")
    for one_sided in [True, False]:
        results = silent(run_bridge_experiment, tolerance=30, one_sided=one_sided)
        mode = "one-sided" if one_sided else "symmetric"
        print(f"\n-- {mode} --")
        for feature in ["Volatility_20", "Log_Return"]:
            pooled = pooled_across_folds(results[feature]["KSWIN"]["per_fold"])
            print(f"  {feature:14} precision={pooled['pooled_precision_str']:38} recall={pooled['pooled_recall_str']:38}")

        # Residual stream for the same comparison, reusing Table I's tol=30 run
        residual_results = silent(run_full_rolling_evaluation, tolerance=30, one_sided=one_sided)
        pooled_resid = pooled_across_folds(residual_results["KSWIN"]["per_fold"])
        print(f"  {'LightGBM residual':14} precision={pooled_resid['pooled_precision_str']:38} recall={pooled_resid['pooled_recall_str']:38}")


def table_vi():
    print_header("TABLE VI -- Continuous, full-history KSWIN (raw volatility)")
    df = load_stock_data("NVDA")
    df = create_features(df)
    df = df.reset_index(drop=True)

    volatility_stream = df["Volatility_20"].values
    kswin_alarms = run_kswin(volatility_stream)
    earnings_ground_truth = get_earnings_ground_truth(df)

    for one_sided in [True, False]:
        for tolerance in [15, 30]:
            metrics = evaluate_detector(kswin_alarms, earnings_ground_truth, tolerance=tolerance, one_sided=one_sided)
            mode = "one-sided" if one_sided else "symmetric"
            prec_str = format_rate_with_ci(metrics["matched_alarms"], len(kswin_alarms))
            rec_str = format_rate_with_ci(metrics["matched_truth"], len(earnings_ground_truth))
            print(f"tol={tolerance:>2} {mode:>10} | precision={prec_str:38} recall={rec_str:38} delay={metrics['avg_delay']}")


if __name__ == "__main__":
    table_i()
    table_ii()
    table_vi()
    print("\nDone. Copy the one-sided rows into your revised Tables I, II, VI.")
    print("Report the symmetric rows too, as a sensitivity comparison (Reviewer 3 point 4).")