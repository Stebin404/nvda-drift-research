"""
models/baseline_models.py

Baseline forecasters for the 5-day-forward log-return target, to be
compared against LightGBM (models/lightgbm_model.py) under the exact
same walk-forward protocol (retraining_engine.walk_forward_predict /
compare_retraining_strategies) and the exact same rolling-window setup
(evaluation/rolling_retraining_evaluation.py).

Every build_*_model() function returns an object with a scikit-learn-
compatible .fit(X, y) / .predict(X) interface, which is all
retraining_engine.py requires of build_model_fn -- so these plug into
the existing harness with no changes to retraining_engine.py or
evaluation/rolling_retraining_evaluation.py.

Three baselines, in increasing sophistication:

1. Zero-return baseline (build_zero_model) -- always predicts 0.0.
   This is the efficient-market-style null: "the best forecast of a
   log return is no change." sklearn.dummy.DummyRegressor already
   implements this exactly; no custom code needed.

2. Historical-mean baseline (build_mean_model) -- always predicts the
   mean of the target seen in training so far. Also
   DummyRegressor, refit at every retrain point exactly like
   LightGBM is, so the comparison stays apples-to-apples under the
   same protocol.

3. Naive persistence (build_persistence_model, NaivePersistenceModel)
   -- predicts the last realized target value. DummyRegressor can't
   do this (it has no "last value" strategy), so this is a small
   custom class. Intended to be run with strategy="fixed",
   retrain_interval=1 in walk_forward_predict, so it actually updates
   every row rather than freezing at whatever interval LightGBM
   retrains on.

4. Linear/AR baseline (build_linear_model) -- ordinary least squares
   on the same FEATURES used by LightGBM (models/lightgbm_model.py).
   sklearn.linear_model.LinearRegression is already build_model_fn-
   compatible; no wrapper needed.
"""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


def build_zero_model():
    """Always predicts 0.0 -- the 'no change' / efficient-market null."""
    return DummyRegressor(strategy="constant", constant=0.0)


def build_mean_model():
    """Always predicts the training-target mean, refit at every retrain."""
    return DummyRegressor(strategy="mean")


class NaivePersistenceModel:
    """
    Predicts the most recently observed target value.

    Not sklearn-provided (DummyRegressor has no "last value" strategy),
    but matches its .fit(X, y) / .predict(X) interface so it drops
    into walk_forward_predict unchanged. Meant to be run with
    strategy="fixed", retrain_interval=1, since "fit once, predict
    many" is meaningless for a persistence forecaster -- it needs to
    see the newest target value before every single prediction.
    """

    def fit(self, X, y):
        self.last_value = float(y.iloc[-1]) if hasattr(y, "iloc") else float(y[-1])
        return self

    def predict(self, X):
        n = len(X)
        return np.full(n, self.last_value)


def build_persistence_model():
    return NaivePersistenceModel()


def build_linear_model():
    """OLS on the same FEATURES as LightGBM -- the AR-style baseline."""
    return LinearRegression()


# name -> (build_fn, recommended strategy, recommended retrain_interval)
# Recommended settings reflect what each baseline is actually meant to
# test: zero-return and historical-mean are trivial, non-adaptive
# references (strategy="never"); persistence needs to update every
# row to mean anything (strategy="fixed", retrain_interval=1); linear
# regression is cheap to refit, so it is reported under "never" here
# to match the other baselines, but can be re-run under "fixed" /
# "drift_triggered" exactly like LightGBM if that comparison is wanted.
BASELINE_REGISTRY = {
    "zero_return": (build_zero_model, "never", None),
    "historical_mean": (build_mean_model, "never", None),
    "naive_persistence": (build_persistence_model, "fixed", 1),
    "linear_ar": (build_linear_model, "never", None),
}