"""
run_amd_evaluation.py

Runs the same detection and forecasting evaluations already validated
for NVDA, on AMD, using its just-finalized earnings ground truth
(data/earnings_dates_FINAL_amd.csv).

Run from the repo root:  python run_amd_evaluation.py
"""

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from models.baseline_models import BASELINE_REGISTRY
from retraining.retraining_engine import compare_retraining_strategies, walk_forward_predict
from evaluation.run_rolling_evaluation import run_full_rolling_evaluation

AMD_EARNINGS_CSV = "data/earnings_dates_FINAL_amd.csv"


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# Part 1: Detection (Table I/VI-equivalent) + chance baseline
# ---------------------------------------------------------------------

def run_detection_sweep():
    print_header("AMD -- Detection tolerance sweep (KSWIN, one-sided) + chance baseline")
    print(f"{'tol':>4} {'obs.prec':>10} {'null.mean':>10} {'null.p95':>10} {'obs.recall':>11} {'null.recall':>12} {'n_alarms(mean)':>15}")
    for tol in [10, 15, 20, 25, 30]:
        results = run_full_rolling_evaluation(
            ticker="AMD", tolerance=tol, one_sided=True,
            earnings_csv_path=AMD_EARNINGS_CSV,
        )
        agg = results["KSWIN"]["aggregate"]
        per_fold = results["KSWIN"]["per_fold"]
        mean_alarms = sum(f["n_alarms"] for f in per_fold) / len(per_fold)
        nm = agg["null_precision_mean"] if agg["null_precision_mean"] is not None else float("nan")
        np95 = agg["null_precision_p95_mean"] if agg["null_precision_p95_mean"] is not None else float("nan")
        nr = agg["null_recall_mean"] if agg["null_recall_mean"] is not None else float("nan")
        rm = agg["recall_mean"] if agg["recall_mean"] is not None else float("nan")
        print(f"{tol:>4} {agg['precision_mean']:>10.3f} {nm:>10.3f} {np95:>10.3f} {rm:>11.3f} {nr:>12.3f} {mean_alarms:>15.2f}")

    print("\nADWIN / Page-Hinkley alarm counts (sanity check -- were NVDA's zero-alarm results ticker-specific or systemic?):")
    for detector in ["ADWIN", "Page-Hinkley"]:
        results = run_full_rolling_evaluation(
            ticker="AMD", tolerance=20, one_sided=True,
            earnings_csv_path=AMD_EARNINGS_CSV,
        )
        per_fold = results[detector]["per_fold"]
        mean_alarms = sum(f["n_alarms"] for f in per_fold) / len(per_fold)
        print(f"  {detector}: mean alarms/fold = {mean_alarms:.2f}")


# ---------------------------------------------------------------------
# Part 2: Forecasting (Table IV/V-equivalent) + baselines
# ---------------------------------------------------------------------

def run_forecasting_comparison(df):
    print_header("AMD -- Single-window retraining comparison (400-row eval window) + baselines")

    EVAL_WINDOW = 400
    initial_train_size = len(df) - EVAL_WINDOW

    results = compare_retraining_strategies(
        df, FEATURES, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
    )

    print(f"{'Strategy':32s} {'MAE':>10s} {'Retrains':>10s}")
    print("-" * 54)
    for strategy in ("never", "fixed", "drift_triggered"):
        print(f"LightGBM ({strategy}){'':<12s} {results[strategy]['mae']:10.4f} {results[strategy]['n_retrains']:10d}")

    for name, (build_fn, strategy, retrain_interval) in BASELINE_REGISTRY.items():
        kwargs = dict(
            df=df, feature_columns=FEATURES, target_column="Target",
            build_model_fn=build_fn, initial_train_size=initial_train_size,
            strategy=strategy,
        )
        if retrain_interval is not None:
            kwargs["retrain_interval"] = retrain_interval
        r = walk_forward_predict(**kwargs)
        print(f"{name} ({strategy}){'':<8s} {r['mae']:10.4f} {r['n_retrains']:10d}")


if __name__ == "__main__":
    run_detection_sweep()

    df = load_stock_data("AMD")
    df = create_features(df)
    df = create_target(df)
    df = df.reset_index(drop=True)
    run_forecasting_comparison(df)