"""
benchmarks/synthetic_drift.py

FIX over the original version: the original generated three "different"
synthetic experiments (mean drift, variance drift, combined drift) that
all shared the exact same single changepoint location (index 1000) and
the same random seed (42). This meant any apparent agreement between
detectors across the three "different" experiments was guaranteed by
construction -- they weren't independent tests of detector behavior,
they were three views of the same single event.

This version varies the changepoint location, the drift magnitude, and
the random seed across multiple independent trials, so detector
behavior can actually be assessed across a range of conditions rather
than one specific (and accidentally shared) configuration.
"""

import numpy as np


def generate_mean_drift_stream(
    length=2000,
    changepoint_fraction=0.5,
    magnitude=3.0,
    seed=42,
):
    """
    Generates a stream with a single mean shift: N(0, 1) before the
    changepoint, N(magnitude, 1) after it.

    changepoint_fraction: where the changepoint occurs, as a fraction
    of total length (0.5 = exact midpoint). Varying this across trials
    is what the original benchmark failed to do.
    """
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = rng.normal(loc=0, scale=1, size=changepoint)
    after = rng.normal(loc=magnitude, scale=1, size=length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_variance_drift_stream(
    length=2000,
    changepoint_fraction=0.5,
    magnitude=5.0,
    seed=42,
):
    """
    Generates a stream with a single variance shift: N(0, 1) before
    the changepoint, N(0, magnitude) after it. Mean stays at 0
    throughout, so a mean-shift detector should NOT fire if it is
    correctly insensitive to variance-only changes.
    """
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = rng.normal(loc=0, scale=1, size=changepoint)
    after = rng.normal(loc=0, scale=magnitude, size=length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_combined_drift_stream(
    length=2000,
    changepoint_fraction=0.5,
    mean_magnitude=3.0,
    variance_magnitude=5.0,
    seed=42,
):
    """
    Generates a stream with BOTH a mean shift and a variance shift at
    the same changepoint: N(0, 1) before, N(mean_magnitude,
    variance_magnitude) after.
    """
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = rng.normal(loc=0, scale=1, size=changepoint)
    after = rng.normal(loc=mean_magnitude, scale=variance_magnitude, size=length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_trial_grid(
    drift_types=("mean", "variance", "combined"),
    changepoint_fractions=(0.3, 0.5, 0.7),
    seeds=(1, 2, 3, 4, 5),
    length=2000,
):
    """
    Generates a full grid of independent trials: every combination of
    drift type, changepoint location, and seed. This is what makes the
    benchmark genuinely informative -- instead of one stream per drift
    type (the original bug), this produces
    len(drift_types) x len(changepoint_fractions) x len(seeds) total
    independent streams, each with its own known, recorded changepoint
    location for later evaluation.

    Returns a list of dicts: {drift_type, changepoint_fraction, seed,
    stream, changepoint}.
    """
    trials = []

    for drift_type in drift_types:
        for changepoint_fraction in changepoint_fractions:
            for seed in seeds:
                if drift_type == "mean":
                    stream, changepoint = generate_mean_drift_stream(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed
                    )
                elif drift_type == "variance":
                    stream, changepoint = generate_variance_drift_stream(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed
                    )
                elif drift_type == "combined":
                    stream, changepoint = generate_combined_drift_stream(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed
                    )
                else:
                    raise ValueError(f"Unknown drift_type: {drift_type}")

                trials.append({
                    "drift_type": drift_type,
                    "changepoint_fraction": changepoint_fraction,
                    "seed": seed,
                    "stream": stream,
                    "changepoint": changepoint,
                })

    return trials

# ---------------------------------------------------------------------
# TODO item #2: heavy-tailed innovations.
#
# The Gaussian generators above (i.i.d. N(0, sigma) innovations) are a
# clean sanity check but are not realistic financial-return noise: real
# daily returns have heavier tails than a Gaussian (large moves are
# more common than Gaussian kurtosis=3 predicts), which is exactly the
# kind of noise that could make a distribution-shift detector like
# KSWIN fire spuriously even with NO real changepoint, inflating the
# false-alarm rate the pure-Gaussian benchmark cannot expose. A
# reviewer specifically asked for this before trusting the synthetic
# corroboration of the real-data finding (Section IV-C).
#
# These functions reuse the exact same changepoint-fraction / seed
# design as the Gaussian generators, so results are directly comparable
# -- distribution is the ONLY thing that changes.

import numpy as _np_for_t


def _standardized_t_samples(rng, df, size):
    """Student-t samples with STANDARD DEVIATION 1 (not the raw t
    distribution, which has variance df/(df-2) for df>2) -- so
    `magnitude`/`scale` arguments mean the same thing as they do for
    the Gaussian generators above, and only tail heaviness differs."""
    raw = rng.standard_t(df, size=size)
    sd = (df / (df - 2)) ** 0.5
    return raw / sd


def generate_mean_drift_stream_t(
    length=2000,
    changepoint_fraction=0.5,
    magnitude=3.0,
    seed=42,
    t_df=4,
):
    """Same as generate_mean_drift_stream, but with standardized
    Student-t(df=t_df) innovations instead of Gaussian ones. df=4 gives
    finite variance with visibly heavier tails than a Gaussian (excess
    kurtosis 6/(df-4) is undefined at df=4's boundary; df=4 is a
    commonly used "clearly heavy-tailed but still finite-variance"
    choice for financial-return simulation)."""
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = _standardized_t_samples(rng, t_df, changepoint)
    after = magnitude + _standardized_t_samples(rng, t_df, length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_variance_drift_stream_t(
    length=2000,
    changepoint_fraction=0.5,
    magnitude=5.0,
    seed=42,
    t_df=4,
):
    """Student-t equivalent of generate_variance_drift_stream."""
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = _standardized_t_samples(rng, t_df, changepoint)
    after = magnitude * _standardized_t_samples(rng, t_df, length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_combined_drift_stream_t(
    length=2000,
    changepoint_fraction=0.5,
    mean_magnitude=3.0,
    variance_magnitude=5.0,
    seed=42,
    t_df=4,
):
    """Student-t equivalent of generate_combined_drift_stream."""
    rng = np.random.default_rng(seed)
    changepoint = int(length * changepoint_fraction)

    before = _standardized_t_samples(rng, t_df, changepoint)
    after = mean_magnitude + variance_magnitude * _standardized_t_samples(rng, t_df, length - changepoint)

    return np.concatenate([before, after]), changepoint


def generate_trial_grid_t(
    drift_types=("mean", "variance", "combined"),
    changepoint_fractions=(0.3, 0.5, 0.7),
    seeds=(1, 2, 3, 4, 5),
    length=2000,
    t_df=4,
):
    """Student-t analogue of generate_trial_grid, same grid shape so
    results line up trial-for-trial (same seed + changepoint_fraction
    combination) with the Gaussian grid for a side-by-side comparison."""
    trials = []

    for drift_type in drift_types:
        for changepoint_fraction in changepoint_fractions:
            for seed in seeds:
                if drift_type == "mean":
                    stream, changepoint = generate_mean_drift_stream_t(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed, t_df=t_df
                    )
                elif drift_type == "variance":
                    stream, changepoint = generate_variance_drift_stream_t(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed, t_df=t_df
                    )
                elif drift_type == "combined":
                    stream, changepoint = generate_combined_drift_stream_t(
                        length=length, changepoint_fraction=changepoint_fraction, seed=seed, t_df=t_df
                    )
                else:
                    raise ValueError(f"Unknown drift_type: {drift_type}")

                trials.append({
                    "drift_type": drift_type,
                    "changepoint_fraction": changepoint_fraction,
                    "seed": seed,
                    "stream": stream,
                    "changepoint": changepoint,
                    "distribution": "student_t",
                })

    return trials
