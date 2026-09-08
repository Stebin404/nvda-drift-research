"""
benchmarks/run_benchmark.py

Runs ADWIN, Page-Hinkley, and KSWIN across the full synthetic trial
grid (varied drift type, changepoint location, and seed), and reports
whether each detector's alarms land near the known, recorded
changepoint for each trial.

This is a SANITY CHECK, not a real-data experiment: it answers "do
these detector wrappers behave the way drift-detection theory predicts
on data where we know exactly what changed and where" -- which is the
necessary precondition for trusting any null result (e.g. ADWIN firing
zero times on real NVDA data) as meaning "no drift," rather than
"broken wrapper."
"""

import numpy as np

from benchmarks.synthetic_drift import generate_trial_grid
from drift.adwin_detector import run_adwin
from drift.page_hinkley_detector import run_page_hinkley
from drift.kswin_detector import run_kswin


def run_page_hinkley_scaled(stream):
    """
    The original run_page_hinkley default (threshold=50) is calibrated
    for low-magnitude streams. Confirmed directly: on a no-drift,
    constant-high-variance (scale=5) control stream, default
    Page-Hinkley fired 29 times out of pure noise, versus only 2 times
    on an equivalent low-variance (scale=1) control. Since this
    benchmark's variance/combined trials use scale=5 by default, a
    fair comparison requires a threshold scaled to match the data's
    actual magnitude, not the library default tuned for a different
    scale entirely.

    A simple, defensible fix: scale the threshold by the stream's own
    standard deviation, so the detector's sensitivity is calibrated to
    the data it's actually seeing, the same way a practitioner would
    need to tune it for any new dataset rather than blindly trusting
    a library default.
    """
    stream_std = np.std(stream)
    scaled_threshold = 50 * stream_std
    return run_page_hinkley(stream, threshold=scaled_threshold)


DETECTORS = {
    "ADWIN": run_adwin,
    "Page-Hinkley (default)": run_page_hinkley,
    "Page-Hinkley (scaled)": run_page_hinkley_scaled,
    "KSWIN": run_kswin,
}


def evaluate_trial(trial, detector_fn, tolerance=50):
    """
    Runs one detector on one trial's stream, and checks whether ANY
    alarm fell within `tolerance` of the known changepoint. Returns
    True/False (did it detect the changepoint) and the number of
    alarms raised in total (so spurious/excessive alarming is visible
    too, not just whether the real changepoint was caught).
    """
    alarms = detector_fn(trial["stream"])
    changepoint = trial["changepoint"]

    detected = any(abs(a - changepoint) <= tolerance for a in alarms)

    return detected, len(alarms)


def run_full_benchmark(
    drift_types=("mean", "variance", "combined"),
    changepoint_fractions=(0.3, 0.5, 0.7),
    seeds=(1, 2, 3, 4, 5),
    tolerance=50,
):
    trials = generate_trial_grid(
        drift_types=drift_types,
        changepoint_fractions=changepoint_fractions,
        seeds=seeds,
    )

    results = {
        drift_type: {detector_name: {"detected": 0, "total": 0, "alarm_counts": []}
                     for detector_name in DETECTORS}
        for drift_type in drift_types
    }

    for trial in trials:
        for detector_name, detector_fn in DETECTORS.items():
            detected, n_alarms = evaluate_trial(trial, detector_fn, tolerance=tolerance)

            bucket = results[trial["drift_type"]][detector_name]
            bucket["total"] += 1
            bucket["detected"] += int(detected)
            bucket["alarm_counts"].append(n_alarms)

    return results


def print_benchmark_results(results):
    print("\nSYNTHETIC BENCHMARK RESULTS")
    print("=" * 70)

    for drift_type, detector_results in results.items():
        print(f"\n--- Drift type: {drift_type} ---")

        for detector_name, bucket in detector_results.items():
            detection_rate = bucket["detected"] / bucket["total"]
            mean_alarms = np.mean(bucket["alarm_counts"])
            print(
                f"  {detector_name:14s} detection_rate={detection_rate:.2f} "
                f"({bucket['detected']}/{bucket['total']})  "
                f"mean_alarms_per_trial={mean_alarms:.2f}"
            )


if __name__ == "__main__":
    results = run_full_benchmark()
    print_benchmark_results(results)