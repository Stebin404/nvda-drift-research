"""
test_scale_free_features.py

Compares LightGBM trained on the original raw-price-level FEATURES
against FEATURES_SCALE_FREE (Price_to_MA20/Price_to_MA50/Momentum_10_Pct
instead of MA_20/MA_50/Momentum_10), alongside the baseline forecasters,
to check whether feature scaling was suppressing LightGBM's signal
relative to zero-return / historical-mean.

Run from the repo root:  python test_scale_free_features.py
"""

from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES, FEATURES_SCALE_FREE
from models.baseline_models import BASELINE_REGISTRY
from retraining.retraining_engine import compare_retraining_strategies


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def table_iv_feature_comparison(df):
    print_header("Table IV setup -- raw-price vs. scale-free LightGBM features")

    EVAL_WINDOW = 400
    initial_train_size = len(df) - EVAL_WINDOW

    raw_results = compare_retraining_strategies(
        df, FEATURES, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
    )
    scale_free_results = compare_retraining_strategies(
        df, FEATURES_SCALE_FREE, "Target", build_lightgbm_model,
        initial_train_size=initial_train_size, fixed_interval=30,
    )

    print(f"{'Strategy':32s} {'MAE (raw)':>12s} {'MAE (scale-free)':>18s} {'Delta':>10s}")
    print("-" * 76)
    for strategy in ("never", "fixed", "drift_triggered"):
        raw_mae = raw_results[strategy]["mae"]
        sf_mae = scale_free_results[strategy]["mae"]
        print(f"LightGBM ({strategy}){'':<12s} {raw_mae:12.4f} {sf_mae:18.4f} {sf_mae - raw_mae:+10.4f}")

    # Reference lines: where the baselines sit, for context
    print("\nBaselines (never/fixed as in BASELINE_REGISTRY), for reference:")
    for name, (build_fn, strategy, retrain_interval) in BASELINE_REGISTRY.items():
        kwargs = dict(
            df=df, feature_columns=FEATURES, target_column="Target",
            build_model_fn=build_fn, initial_train_size=initial_train_size,
            strategy=strategy,
        )
        if retrain_interval is not None:
            kwargs["retrain_interval"] = retrain_interval
        from retraining.retraining_engine import walk_forward_predict
        r = walk_forward_predict(**kwargs)
        print(f"  {name:20s} MAE={r['mae']:.4f}")

    return raw_results, scale_free_results


if __name__ == "__main__":
    df = load_stock_data("NVDA")
    df = create_features(df)
    df = create_target(df)
    df = df.reset_index(drop=True)

    table_iv_feature_comparison(df)