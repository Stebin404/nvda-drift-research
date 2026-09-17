"""
Bootstrap confidence intervals on the mean MAE difference between
retraining strategies, using the 3 independent rolling-window results
already produced by evaluation/rolling_retraining_evaluation.py.

Why bootstrap and not a t-test: with only 3 independent windows, a
t-test's normality assumption cannot be justified, and a p-value from
n=3 would overstate precision. Bootstrap resampling makes no
distributional assumption and correctly reports a wide interval when
the underlying sample is small, rather than a misleadingly narrow one.
"""

import numpy as np

# Your actual per-window MAE values from Table V (frozen dataset,
# data/nvda_ohlcv_frozen.csv, post purge-gap fix -- see
# evaluation/rolling_origin.py and retraining/retraining_engine.py).
never = np.array([0.064293, 0.072373, 0.070024])
fixed = np.array([0.073753, 0.070878, 0.050098])
drift_triggered = np.array([0.071582, 0.077924, 0.053830])  # updated post KSWIN-seed fix

N_BOOTSTRAP = 10000
rng = np.random.default_rng(42)


def bootstrap_mean_diff(a, b, n_boot=N_BOOTSTRAP):
    """
    Resamples paired window indices with replacement (paired, since
    all three strategies were evaluated on the SAME 3 windows), and
    returns the distribution of mean(a) - mean(b) under resampling.
    """
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return diffs


def report(name_a, a, name_b, b):
    diffs = bootstrap_mean_diff(a, b)
    point_estimate = a.mean() - b.mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    includes_zero = ci_low <= 0 <= ci_high
    print(f"{name_a} - {name_b}:")
    print(f"  point estimate of mean difference: {point_estimate:.5f}")
    print(f"  95% bootstrap CI: [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"  CI includes zero: {includes_zero}")
    print()


report("drift_triggered", drift_triggered, "fixed", fixed)
report("drift_triggered", drift_triggered, "never", never)
report("fixed", fixed, "never", never)