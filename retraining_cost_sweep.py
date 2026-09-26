"""
retraining_cost_sweep.py

Applies retraining/cost_function.py's J = MAE + lambda*n_retrains to
the SAME single-window (Table IV) and multi-window (Table V) retraining
comparisons already in the paper, for NVDA. lambda values are chosen so
the sweep spans "retrains are free" (lambda=0, i.e. MAE alone decides)
through "a retrain costs as much as a ~10% relative MAE degradation"
given this dataset's typical retrain counts and MAE scale.

Run from the repo root: python retraining_cost_sweep.py
"""

import json

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from retraining.retraining_engine import compare_retraining_strategies
from evaluation.rolling_retraining_evaluation import generate_retraining_windows
from retraining.cost_function import cost_table, print_cost_table

LAMBDAS = [0.0, 0.0005, 0.001, 0.002, 0.005]


def single_window_results(df):
    EVAL_WINDOW = 400
    initial_train_size = len(df) - EVAL_WINDOW
    return compare_retraining_strategies(
        df, FEATURES, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
    )


def multi_window_mean_results(df):
    """Mean MAE and mean n_retrains across the 3 independent 300-row
    windows used for Table V, per strategy -- the same aggregate the
    paper already reports (Section IV-D), just fed through J."""
    windows = generate_retraining_windows(
        n_rows=len(df), eval_window_size=300, min_initial_train=900, n_windows=3,
    )
    per_strategy = {"never": [], "fixed": [], "drift_triggered": []}
    for w in windows:
        results = compare_retraining_strategies(
            df, FEATURES, "Target", build_lightgbm_model,
            initial_train_size=w["initial_train_size"], fixed_interval=30,
        )
        for strategy in per_strategy:
            per_strategy[strategy].append(results[strategy])

    mean_results = {}
    for strategy, runs in per_strategy.items():
        mean_results[strategy] = {
            "mae": sum(r["mae"] for r in runs) / len(runs),
            "n_retrains": sum(r["n_retrains"] for r in runs) / len(runs),
        }
    return mean_results


if __name__ == "__main__":
    df = load_stock_data("NVDA")
    df = create_features(df)
    df = create_target(df)
    df = df.reset_index(drop=True)

    single = single_window_results(df)
    single_table = cost_table(single, LAMBDAS)
    print_cost_table(single_table, title="NVDA -- single 400-row window (Table IV strategies)")

    multi = multi_window_mean_results(df)
    multi_table = cost_table(multi, LAMBDAS)
    print_cost_table(multi_table, title="NVDA -- mean over 3 independent 300-row windows (Table V strategies)")

    with open("retraining_cost_results.json", "w") as f:
        json.dump({"single_window": {"strategies": single, "cost_table": single_table},
                    "multi_window_mean": {"strategies": multi, "cost_table": multi_table}},
                   f, indent=2, default=str)
    print("\nFull results written to retraining_cost_results.json")
