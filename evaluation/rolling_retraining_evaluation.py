"""
evaluation/rolling_retraining_evaluation.py

Runs the never / fixed / drift_triggered retraining comparison
independently across multiple non-overlapping evaluation windows
spanning NVDA's history, instead of a single 400-row window at the
end of the dataset.

This directly tests whether the earlier finding ("never beats both
retraining strategies") is a stable property of NVDA's data, or
specific to the one window previously tested. Each window gets its
own initial_train_size and its own 400-row evaluation period,
non-overlapping with the others, so this produces several independent
realizations of the same three-way comparison.
"""

import numpy as np

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from retraining.retraining_engine import compare_retraining_strategies


def generate_retraining_windows(n_rows, eval_window_size=400, min_initial_train=600, n_windows=4):
    """
    Generates n_windows non-overlapping (initial_train_size,
    eval_end) pairs, walking forward through the dataset.

    Each window's initial_train_size grows by eval_window_size from
    the previous one, so windows tile the back portion of the dataset
    without overlapping -- e.g. with eval_window_size=400 and 4
    windows, the last 1600 rows get split into four independent
    400-row evaluation stretches, each with its own training cutoff.

    Raises ValueError if the requested configuration doesn't fit,
    rather than silently running fewer windows than asked for.
    """
    total_needed = min_initial_train + n_windows * eval_window_size

    if total_needed > n_rows:
        raise ValueError(
            f"Requested {n_windows} windows of {eval_window_size} rows "
            f"each, after a minimum initial training size of "
            f"{min_initial_train}, require {total_needed} rows but "
            f"only {n_rows} are available. Reduce n_windows or "
            f"eval_window_size."
        )

    first_window_start = n_rows - n_windows * eval_window_size

    windows = []
    for w in range(n_windows):
        initial_train_size = first_window_start + w * eval_window_size
        windows.append({
            "window_id": w,
            "initial_train_size": initial_train_size,
            "eval_end": initial_train_size + eval_window_size,
        })

    return windows


def run_rolling_retraining_evaluation(
    ticker="NVDA",
    eval_window_size=300,
    min_initial_train=900,
    n_windows=3,
    fixed_interval=30,
):
    full_df = load_stock_data(ticker)
    full_df = create_features(full_df)
    full_df = create_target(full_df)
    full_df = full_df.reset_index(drop=True)

    windows = generate_retraining_windows(
        n_rows=len(full_df),
        eval_window_size=eval_window_size,
        min_initial_train=min_initial_train,
        n_windows=n_windows,
    )

    results = {"never": [], "fixed": [], "drift_triggered": []}

    for window in windows:
        # Slice df down to exactly this window's visible data (training
        # history + this window's own evaluation period), so later
        # windows' evaluation rows can never leak into an earlier
        # window's comparison.
        window_df = full_df.iloc[:window["eval_end"]].reset_index(drop=True)

        comparison = compare_retraining_strategies(
            window_df, FEATURES, "Target", build_lightgbm_model,
            initial_train_size=window["initial_train_size"],
            fixed_interval=fixed_interval,
        )

        for strategy_name, data in comparison.items():
            results[strategy_name].append({
                "window_id": window["window_id"],
                "mae": data["mae"],
                "n_retrains": data["n_retrains"],
            })

    return results


def print_rolling_retraining_results(results):
    print("\nROLLING RETRAINING COMPARISON (multiple independent windows)")
    print("=" * 70)

    for strategy_name, window_results in results.items():
        print(f"\n{strategy_name}")
        print("-" * 40)
        for w in window_results:
            print(f"  window {w['window_id']}: MAE={w['mae']:.6f}  n_retrains={w['n_retrains']}")

        maes = [w["mae"] for w in window_results]
        retrains = [w["n_retrains"] for w in window_results]
        print(f"  MEAN MAE: {np.mean(maes):.6f}  (+/-{np.std(maes):.6f})")
        print(f"  MEAN RETRAINS: {np.mean(retrains):.1f}")

    print("\n" + "=" * 70)
    print("WINS PER WINDOW (lowest MAE among the three strategies)")
    n_windows = len(results["never"])
    for w_id in range(n_windows):
        maes_this_window = {
            strategy: results[strategy][w_id]["mae"]
            for strategy in results
        }
        winner = min(maes_this_window, key=maes_this_window.get)
        print(f"  window {w_id}: {winner} wins ({maes_this_window})")


if __name__ == "__main__":
    results = run_rolling_retraining_evaluation()
    print_rolling_retraining_results(results)