"""
benchmarks/run_benchmark_heavy_tailed.py

TODO item #2: reruns the three synthetic drift-type trials (mean,
variance, combined) under standardized Student-t(df=4) innovations
(benchmarks/synthetic_drift.py's *_t generators) using the SAME
multi-seed / multi-changepoint-location design as the existing
Gaussian benchmark (benchmarks/run_benchmark.py), and reports:

  - detection rate   : same definition as run_benchmark.py (any alarm
                        within `tolerance` of the true changepoint).
  - false-alarm rate  : mean number of alarms raised STRICTLY BEFORE
                        the changepoint per trial (there is no drift
                        to detect there, so every such alarm is, by
                        construction, spurious).
  - avg run length    : mean delay from the changepoint to the FIRST
                        post-changepoint alarm, over trials that
                        raised at least one (undefined/NaN otherwise;
                        reported as such rather than silently dropped
                        or zeroed).

Run from the repo root: python -m benchmarks.run_benchmark_heavy_tailed
"""

import json

import numpy as np

from benchmarks.synthetic_drift import generate_trial_grid, generate_trial_grid_t
from benchmarks.run_benchmark import DETECTORS, evaluate_trial


def evaluate_trial_detailed(trial, detector_fn, tolerance=50):
    alarms = detector_fn(trial["stream"])
    changepoint = trial["changepoint"]

    detected = any(abs(a - changepoint) <= tolerance for a in alarms)
    pre_change_alarms = [a for a in alarms if a < changepoint]
    post_change_alarms = sorted(a for a in alarms if a >= changepoint)
    run_length = (post_change_alarms[0] - changepoint) if post_change_alarms else None

    return {
        "detected": detected,
        "n_alarms": len(alarms),
        "n_false_alarms_pre_change": len(pre_change_alarms),
        "run_length": run_length,
    }


def run_grid(trials, tolerance=50):
    results = {}
    drift_types = sorted(set(t["drift_type"] for t in trials))
    for drift_type in drift_types:
        results[drift_type] = {name: {"detected": 0, "total": 0, "false_alarms": [], "run_lengths": []}
                                for name in DETECTORS}

    for trial in trials:
        for detector_name, detector_fn in DETECTORS.items():
            r = evaluate_trial_detailed(trial, detector_fn, tolerance=tolerance)
            bucket = results[trial["drift_type"]][detector_name]
            bucket["total"] += 1
            bucket["detected"] += int(r["detected"])
            bucket["false_alarms"].append(r["n_false_alarms_pre_change"])
            if r["run_length"] is not None:
                bucket["run_lengths"].append(r["run_length"])

    summary = {}
    for drift_type, detectors in results.items():
        summary[drift_type] = {}
        for name, bucket in detectors.items():
            summary[drift_type][name] = {
                "detection_rate": bucket["detected"] / bucket["total"],
                "false_alarm_rate_per_trial": float(np.mean(bucket["false_alarms"])),
                "avg_run_length": float(np.mean(bucket["run_lengths"])) if bucket["run_lengths"] else None,
            }
    return summary


def print_comparison(gaussian_summary, t_summary):
    print("\nSYNTHETIC BENCHMARK: GAUSSIAN vs. STUDENT-t(df=4) INNOVATIONS")
    print("=" * 90)
    for drift_type in gaussian_summary:
        print(f"\n--- Drift type: {drift_type} ---")
        print(f"{'Detector':24s} {'DetRate(G)':>11} {'DetRate(t)':>11} {'FalseAlarm(G)':>14} {'FalseAlarm(t)':>14} {'RunLen(G)':>10} {'RunLen(t)':>10}")
        for name in gaussian_summary[drift_type]:
            g = gaussian_summary[drift_type][name]
            t = t_summary[drift_type][name]
            rl_g = g["avg_run_length"] if g["avg_run_length"] is not None else float("nan")
            rl_t = t["avg_run_length"] if t["avg_run_length"] is not None else float("nan")
            print(
                f"{name:24s} {g['detection_rate']:>11.2f} {t['detection_rate']:>11.2f} "
                f"{g['false_alarm_rate_per_trial']:>14.2f} {t['false_alarm_rate_per_trial']:>14.2f} "
                f"{rl_g:>10.1f} {rl_t:>10.1f}"
            )


if __name__ == "__main__":
    gaussian_trials = generate_trial_grid()
    t_trials = generate_trial_grid_t()

    gaussian_summary = run_grid(gaussian_trials)
    t_summary = run_grid(t_trials)

    print_comparison(gaussian_summary, t_summary)

    with open("heavy_tailed_benchmark_results.json", "w") as f:
        json.dump({"gaussian": gaussian_summary, "student_t": t_summary}, f, indent=2, default=str)
    print("\nFull results written to heavy_tailed_benchmark_results.json")
