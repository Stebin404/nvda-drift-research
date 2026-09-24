"""
run_multiticker_core_tournament.py

TODO item #3: AMD and TSLA were previously only run through the
chance-baseline / forecast-baseline check (run_amd_evaluation.py,
run_tsla_evaluation.py). The CORE error-stream detector tournament
(Table I) and the raw-feature bridge experiment (Table II) -- the
experiments that establish the paper's central mean-shift-vs-
distribution-shift finding -- were NVDA-only. This script runs both
on AMD and TSLA with the existing, unmodified pipeline
(evaluation/run_rolling_evaluation.py, evaluation/bridge_experiment.py),
so the pattern can be checked for replication rather than only
disclosed as a limitation.

Run from the repo root: python run_multiticker_core_tournament.py
"""

import json

from evaluation.run_rolling_evaluation import run_full_rolling_evaluation
from evaluation.bridge_experiment import run_bridge_experiment

TICKERS = {
    "NVDA": "data/earnings_dates_FINAL.csv",
    "AMD": "data/earnings_dates_FINAL_amd.csv",
    "TSLA": "data/earnings_dates_FINAL_tsla.csv",
}

TOLERANCES = [10, 15, 20, 25, 30]


def run_tournament_sweep(ticker, csv_path):
    """Table-I-style: KSWIN precision/recall/delay (+ ADWIN/PH alarm
    counts, which are zero at every tolerance on NVDA) across the
    tolerance sweep, on the LightGBM residual stream."""
    rows = []
    for tol in TOLERANCES:
        results = run_full_rolling_evaluation(
            ticker=ticker, tolerance=tol, one_sided=True, earnings_csv_path=csv_path,
        )
        row = {"tolerance": tol}
        for detector in ("ADWIN", "Page-Hinkley", "KSWIN"):
            agg = results[detector]["aggregate"]
            per_fold = results[detector]["per_fold"]
            total_alarms = sum(f["n_alarms"] for f in per_fold)
            row[detector] = {
                "total_alarms": total_alarms,
                "precision_macro": agg["precision_mean"],
                "recall_macro": agg["recall_mean"],
                "delay_mean": agg["avg_delay_mean"],
            }
        rows.append(row)
    return rows


def run_bridge_sweep(ticker, csv_path, tolerance=30):
    """Table-II-style: all three detectors on raw Volatility_20 and
    raw Log_Return, at a fixed tolerance, same folds/ground truth as
    the residual-stream tournament."""
    return run_bridge_experiment(ticker=ticker, tolerance=tolerance, one_sided=True, earnings_csv_path=csv_path)


def print_tournament(ticker, rows):
    print(f"\n=== {ticker}: error-stream tournament (Table I style, causal matching) ===")
    print(f"{'tol':>4} | {'KSWIN prec':>10} {'KSWIN rec':>10} {'KSWIN delay':>11} | {'ADWIN alarms':>12} {'PH alarms':>9}")
    for row in rows:
        k = row["KSWIN"]
        a = row["ADWIN"]
        p = row["Page-Hinkley"]
        rec = k["recall_macro"] if k["recall_macro"] is not None else float("nan")
        delay = k["delay_mean"] if k["delay_mean"] is not None else float("nan")
        print(f"{row['tolerance']:>4} | {k['precision_macro']:>10.3f} {rec:>10.3f} {delay:>11.1f} | {a['total_alarms']:>12} {p['total_alarms']:>9}")


def print_bridge(ticker, results):
    print(f"\n=== {ticker}: bridge experiment (Table II style, tolerance=30, causal) ===")
    for feature_name, detector_results in results.items():
        print(f"-- {feature_name} --")
        for detector_name, data in detector_results.items():
            agg = data["aggregate"]
            recall = agg["recall_mean"] if agg["recall_mean"] is not None else float("nan")
            delay = agg["avg_delay_mean"] if agg["avg_delay_mean"] is not None else float("nan")
            n_alarms = sum(f["n_alarms"] for f in data["per_fold"])
            print(f"  {detector_name:14s} alarms={n_alarms:3d}  precision={agg['precision_mean']:.3f}  recall={recall:.3f}  delay={delay:.1f}")


if __name__ == "__main__":
    all_out = {}
    for ticker, csv_path in TICKERS.items():
        tournament_rows = run_tournament_sweep(ticker, csv_path)
        print_tournament(ticker, tournament_rows)

        bridge_results = run_bridge_sweep(ticker, csv_path)
        print_bridge(ticker, bridge_results)

        all_out[ticker] = {"tournament": tournament_rows, "bridge": bridge_results}

    with open("multiticker_tournament_results.json", "w") as f:
        json.dump(all_out, f, indent=2, default=str)

    print("\nFull results written to multiticker_tournament_results.json")
