"""
evaluation/rolling_origin.py

Replaces a single static 80/20 train/test split with a rolling-origin
(walk-forward) evaluation scheme.

Why this matters, not just as a nicety:

1. STATISTICAL: A single 80/20 split gives exactly one realization of
   "did the detector fire near a regime change." With only a handful
   of earnings events landing in any single test fold, precision/
   recall computed from one fold are small-integer ratios, not stable
   estimates. Rolling-origin produces multiple independent folds, so
   results can be reported as mean +/- std across folds.

2. COVERAGE: different folds expose different earnings dates to the
   detectors, so aggregating across folds uses far more of the
   available ground truth (30 confirmed earnings events) than any
   single split does.

3. REALISM: a deployed drift-monitoring system trains on history,
   monitors a window, eventually retrains -- rolling-origin evaluation
   matches this deployment scenario more closely than one static split.

This module does not retrain with different hyperparameters per fold
-- it reuses the same model constructor (e.g. build_lightgbm_model)
across folds, just on different data slices.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error


def generate_rolling_folds(n_rows, n_folds=5, min_train_fraction=0.5, test_fraction=0.1):
    """
    Computes (train_end, test_start, test_end) tuples for n_folds
    sequential, non-overlapping test windows, each preceded by an
    expanding training window (train on everything up to that point).

    Parameters
    ----------
    n_rows : int
        Total number of rows in the full feature+target dataframe.
    n_folds : int
        Number of sequential test windows to generate.
    min_train_fraction : float
        The first fold's training set is at least this fraction of
        n_rows, so early folds aren't trained on too little history.
    test_fraction : float
        Size of each test window, as a fraction of n_rows.

    Returns
    -------
    list[dict] with keys: fold_id, train_end, test_start, test_end.

    Raises
    ------
    ValueError if the requested folds don't fit inside n_rows. A
    silent truncation to fewer folds than requested would make a
    reported "5-fold rolling evaluation" actually be a 3-fold one
    without anyone noticing -- this must fail loudly instead.
    """
    test_size = int(n_rows * test_fraction)
    min_train_size = int(n_rows * min_train_fraction)

    last_test_end = min_train_size + n_folds * test_size

    if last_test_end > n_rows:
        raise ValueError(
            f"Requested {n_folds} folds of test_fraction={test_fraction} "
            f"after an initial training fraction of {min_train_fraction} "
            f"require {last_test_end} rows but only {n_rows} are "
            f"available. Reduce n_folds, reduce test_fraction, or reduce "
            f"min_train_fraction."
        )

    folds = []
    for fold_id in range(n_folds):
        train_end = min_train_size + fold_id * test_size
        test_start = train_end
        test_end = test_start + test_size

        folds.append({
            "fold_id": fold_id,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    return folds


HORIZON = 5
# Target (features/target.py) is y_t = ln(Close[t+5]/Close[t]), computed on
# the full continuous price series before fold slicing. A training row at
# position train_end-1 therefore has a target that depends on prices at
# train_end .. train_end+4 -- i.e. the first HORIZON rows of that fold's
# test window. Purging the last HORIZON rows from every fold's training
# set removes this leakage; see PATCH NOTE in run_rolling_fold below.


def run_rolling_fold(df, fold, build_model_fn, feature_columns, target_column="Target"):
    """
    Trains a fresh model on df[:fold['train_end'] - HORIZON] and evaluates
    on df[fold['test_start']:fold['test_end']].

    build_model_fn must be a zero-argument constructor returning an
    UNFIT estimator with .fit(X, y) and .predict(X) -- pass the
    function itself (e.g. build_lightgbm_model), not its result. A
    fresh model is built per fold rather than reusing one object,
    since not every estimator type is guaranteed to fully reset
    internal state on a second .fit() call.

    df must already have feature_columns and target_column present
    (i.e. create_features() and create_target() already applied).

    PATCH NOTE (purge gap): the last HORIZON rows before train_end are
    dropped from training. Their Target values are computed from prices
    that fall inside this fold's test window (see HORIZON comment above),
    so including them would leak test-period prices into training.
    """
    fold_df = df.iloc[:fold["test_end"]].reset_index(drop=True)

    X = fold_df[feature_columns]
    y = fold_df[target_column]

    purge_end = fold["train_end"] - HORIZON

    X_train = X.iloc[:purge_end]
    X_test = X.iloc[fold["test_start"]:fold["test_end"]]

    y_train = y.iloc[:purge_end]
    y_test = y.iloc[fold["test_start"]:fold["test_end"]]

    model = build_model_fn()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    error_stream = np.abs(y_test.values - preds)
    signed_error_stream = y_test.values - preds

    return {
        "fold_id": fold["fold_id"],
        "model": model,
        "mae": mae,
        "predictions": preds,
        "actuals": y_test,
        "error_stream": error_stream,
        "signed_error_stream": signed_error_stream,
        "test_start_full_df_index": fold["test_start"],
        "test_end_full_df_index": fold["test_end"],
    }


def run_all_folds(df, build_model_fn, feature_columns, target_column="Target",
                   n_folds=5, min_train_fraction=0.5, test_fraction=0.1):
    """
    Convenience wrapper: generates folds and runs all of them,
    returning a list of per-fold result dicts.
    """
    folds = generate_rolling_folds(
        n_rows=len(df),
        n_folds=n_folds,
        min_train_fraction=min_train_fraction,
        test_fraction=test_fraction,
    )

    return [
        run_rolling_fold(df, fold, build_model_fn, feature_columns, target_column)
        for fold in folds
    ]