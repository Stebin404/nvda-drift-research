"""
evaluation/detector_metrics.py

FIX 1 (already present): matching each alarm to the FIRST ground-truth
point within tolerance (in list order) is a real bug whenever
ground-truth points cluster closely. This version matches each alarm
to its NEAREST ground-truth point within tolerance, using greedy
nearest-distance assignment.

FIX 2 (this revision): added `one_sided=True` support. Reviewer 3
(point 4) correctly identified that a symmetric |alarm - truth| <=
tolerance window lets an alarm BEFORE the earnings date count as a
"detection" of that earnings date -- which is not causally coherent
if the story is "the detector picks up the market's reaction to the
event." one_sided=True restricts matching to
    0 <= (alarm - truth) <= tolerance
i.e. the alarm must occur on or after the ground-truth date, within
the tolerance window. Default is one_sided=True going forward; pass
one_sided=False to reproduce the original (symmetric) numbers for a
side-by-side comparison table in the paper (recommended: report both,
since this is exactly the kind of protocol choice Reviewer 3 asked to
see a sensitivity check on).

FIX 3 (this revision): added `permutation_null_precision` -- shuffles
alarm-to-timeline positions (or, more precisely, resamples pseudo
ground-truth dates uniformly across the same test-fold span N times)
and recomputes precision/recall under each, giving an empirical chance
baseline. This directly answers Reviewer 3 point 4 / point 17 ("compare
against randomised event calendars to quantify chance precision and
recall").
"""

import random


def evaluate_detector(alarms, ground_truth, tolerance=30, one_sided=True):
    """
    Greedy nearest-match evaluation between detector alarms and
    ground-truth event positions.

    Parameters
    ----------
    alarms : list[int]
        Row-index positions where a detector raised an alarm.
    ground_truth : list[int]
        Row-index positions of verified earnings-event anchors.
    tolerance : int
        Matching window width, in rows (trading days).
    one_sided : bool
        If True (default, recommended), an alarm only matches a
        ground-truth event if it occurs on or after that event, within
        `tolerance` days: 0 <= alarm - truth <= tolerance. This is the
        causally coherent choice for a "detector reacts to an event"
        story.
        If False, reproduces the original symmetric-window behaviour:
        abs(alarm - truth) <= tolerance. Kept only for a reported
        side-by-side sensitivity comparison against the one_sided
        results -- do not use as the primary reported number without
        stating this explicitly.

    Returns
    -------
    dict with precision, recall, avg_delay, matched_alarms,
    matched_truth (the last two now as raw counts for CI computation
    -- see evaluation/confidence_intervals.py).
    """
    candidate_pairs = []

    for alarm in alarms:
        for truth in ground_truth:
            if one_sided:
                delta = alarm - truth
                if 0 <= delta <= tolerance:
                    candidate_pairs.append((delta, alarm, truth))
            else:
                distance = abs(alarm - truth)
                if distance <= tolerance:
                    candidate_pairs.append((distance, alarm, truth))

    candidate_pairs.sort(key=lambda pair: pair[0])

    matched_alarms = set()
    matched_truth = set()
    delays = []

    for distance, alarm, truth in candidate_pairs:
        if alarm in matched_alarms or truth in matched_truth:
            continue

        matched_alarms.add(alarm)
        matched_truth.add(truth)
        delays.append(distance)

    n_matched_alarms = len(matched_alarms)
    n_matched_truth = len(matched_truth)

    precision = n_matched_alarms / len(alarms) if alarms else 0
    recall = n_matched_truth / len(ground_truth) if ground_truth else 0
    avg_delay = sum(delays) / len(delays) if delays else None

    return {
        "precision": precision,
        "recall": recall,
        "avg_delay": avg_delay,
        "matched_alarms": n_matched_alarms,
        "matched_truth": n_matched_truth,
        "one_sided": one_sided,
        "tolerance": tolerance,
    }


def permutation_null_precision(
    alarms,
    n_ground_truth,
    fold_start,
    fold_end,
    tolerance=30,
    one_sided=True,
    n_trials=1000,
    seed=42,
):
    """
    Empirical chance-level precision/recall: draws `n_ground_truth`
    pseudo-event positions uniformly at random from [fold_start,
    fold_end) on each of n_trials trials, evaluates the SAME alarm
    list against each random calendar, and returns the distribution
    of precision/recall under the null "alarms are unrelated to
    earnings timing."

    Report the observed precision/recall against this null's mean and
    95th percentile, e.g.: "observed precision 0.600 vs. chance mean
    0.09 (95th pct 0.30, n=1000 permutations)". This is the
    randomisation-based baseline Reviewer 3 asked for in point 17 and
    Section 5.2.

    Parameters
    ----------
    alarms : list[int]
    n_ground_truth : int
        Number of pseudo-events to draw per trial (match the real
        ground-truth count for that fold).
    fold_start, fold_end : int
        Row-index bounds of the test fold the alarms came from -- the
        pseudo-events are drawn only from this span, not the whole
        price history, to keep the comparison apples-to-apples.
    """
    rng = random.Random(seed)
    precisions = []
    recalls = []

    span = list(range(fold_start, fold_end))
    if n_ground_truth > len(span):
        raise ValueError(
            f"n_ground_truth={n_ground_truth} exceeds fold span size {len(span)}."
        )

    for _ in range(n_trials):
        pseudo_truth = rng.sample(span, n_ground_truth)
        metrics = evaluate_detector(alarms, pseudo_truth, tolerance=tolerance, one_sided=one_sided)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    precisions.sort()
    recalls.sort()
    n = len(precisions)

    def pct(sorted_vals, p):
        idx = min(n - 1, int(p * n))
        return sorted_vals[idx]

    return {
        "n_trials": n_trials,
        "precision_mean": sum(precisions) / n,
        "precision_p95": pct(precisions, 0.95),
        "recall_mean": sum(recalls) / n,
        "recall_p95": pct(recalls, 0.95),
    }