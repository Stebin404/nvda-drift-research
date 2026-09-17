"""
retraining/retraining_engine.py

Compares three retraining strategies using forecasting MAE:

1. NEVER  -- train once, forecast the rest with a frozen model.
2. FIXED  -- retrain every retrain_interval rows.
3. DRIFT_TRIGGERED -- retrain when a PERSISTENT KSWIN detector,
   updated one Volatility_20 value at a time across the whole
   evaluation period, signals drift_detected.

FIX over the previous version: that version rebuilt a fresh KSWIN
detector from scratch on every row using a sliding window -- this
caused KSWIN to fire on 100% of rows, since comparing two halves of
the same short overlapping window picks up ordinary day-to-day
variation, not genuine drift. A persistent detector that accumulates
state across the whole walk-forward period (exactly like
drift/kswin_detector.py's run_kswin function already does in the
rolling-origin and bridge experiments) only fires when something
shifts relative to real accumulated history.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from river.drift import KSWIN

HORIZON = 5
# Target column (features/target.py) is y_t = ln(Close[t+5]/Close[t]),
# computed on the full continuous df before this loop ever runs. At real
# time i, the target for any row j > i - HORIZON is not yet observable
# (it needs Close[j+5], which is in the future relative to i). Every
# .fit() call below is therefore capped at i + 1 - HORIZON, not i + 1,
# so training never uses a label that wouldn't actually exist yet at
# that point in a live deployment.


def walk_forward_predict(df, feature_columns, target_column,
                          build_model_fn, initial_train_size,
                          strategy="never", retrain_interval=60,
                          train_window_size=None):
    """
    train_window_size: if None (default), every retrain uses an
    EXPANDING window (all data from row 0 up to the current point) --
    this was the original behavior. If set to an integer (e.g. 600),
    every retrain uses only the most recent train_window_size rows
    (a SLIDING window), so a retrain actually shifts what the model
    has seen, rather than adding a small amount of new data to an
    already-large, mostly-unchanged training set.
    """
    """
    strategy:
      "never"           -- train once, never retrain.
      "fixed"           -- retrain every `retrain_interval` rows.
      "drift_triggered" -- retrain whenever a persistent KSWIN
                           detector (fed Volatility_20 one value at a
                           time, starting from row 0 of df, not just
                           the evaluation period) signals drift.

    Returns: predictions, actuals, mae, retrain_points, n_retrains
    """
    X_full = df[feature_columns]
    y_full = df[target_column]
    volatility = df["Volatility_20"].values

    model = build_model_fn()
    init_fit_end = initial_train_size - HORIZON
    model.fit(X_full.iloc[:init_fit_end], y_full.iloc[:init_fit_end])

    # Persistent KSWIN detector, seeded on all volatility history up
    # to the start of evaluation, then updated one value at a time as
    # we walk forward -- this is the same usage pattern as
    # drift/kswin_detector.py's run_kswin, just kept alive across the
    # whole loop instead of rebuilt every row.
    #
    # seed=42 fixed to match drift/kswin_detector.py's DEFAULT_KSWIN_SEED
    # -- river's KSWIN does internal random sub-sampling and is
    # non-deterministic with seed=None, which previously made every
    # "drift_triggered" retraining run (and therefore Tables IV/V)
    # non-reproducible across repeated executions on the same data.
    kswin_detector = KSWIN(seed=42)
    for v in volatility[:initial_train_size]:
        kswin_detector.update(float(v))

    predictions = []
    actuals = []
    retrain_points = []
    rows_since_last_retrain = 0

    for i in range(initial_train_size, len(df)):
        x_row = X_full.iloc[[i]]
        y_row = y_full.iloc[i]

        pred = model.predict(x_row)[0]
        predictions.append(pred)
        actuals.append(y_row)

        kswin_detector.update(float(volatility[i]))

        rows_since_last_retrain += 1
        eval_relative_idx = i - initial_train_size
        should_retrain = False

        if strategy == "fixed":
            should_retrain = rows_since_last_retrain >= retrain_interval

        elif strategy == "drift_triggered":
            should_retrain = kswin_detector.drift_detected

        if should_retrain:
            model = build_model_fn()
            if train_window_size is None:
                # Expanding window: train on everything seen so far.
                train_start = 0
            else:
                # Sliding window: train on only the most recent
                # train_window_size rows, so a retrain actually shifts
                # the training distribution meaningfully rather than
                # being diluted by a huge, mostly-unchanged history.
                train_start = max(0, i + 1 - train_window_size)
            fit_end = i + 1 - HORIZON
            model.fit(X_full.iloc[train_start:fit_end], y_full.iloc[train_start:fit_end])
            retrain_points.append(eval_relative_idx)
            rows_since_last_retrain = 0

    mae = mean_absolute_error(actuals, predictions)

    return {
        "strategy": strategy,
        "predictions": predictions,
        "actuals": actuals,
        "mae": mae,
        "retrain_points": retrain_points,
        "n_retrains": len(retrain_points),
    }


def compare_retraining_strategies(df, feature_columns, target_column,
                                   build_model_fn, initial_train_size,
                                   fixed_interval=60, train_window_size=None):
    results = {}

    results["never"] = walk_forward_predict(
        df, feature_columns, target_column, build_model_fn,
        initial_train_size, strategy="never",
        train_window_size=train_window_size
    )

    results["fixed"] = walk_forward_predict(
        df, feature_columns, target_column, build_model_fn,
        initial_train_size, strategy="fixed", retrain_interval=fixed_interval,
        train_window_size=train_window_size
    )

    results["drift_triggered"] = walk_forward_predict(
        df, feature_columns, target_column, build_model_fn,
        initial_train_size, strategy="drift_triggered",
        train_window_size=train_window_size
    )

    return results


def print_comparison(results):
    print("\nRETRAINING STRATEGY COMPARISON")
    print("=" * 60)
    for strategy_name, data in results.items():
        print(
            f"{strategy_name:18s} MAE={data['mae']:.6f}  "
            f"n_retrains={data['n_retrains']}"
        )