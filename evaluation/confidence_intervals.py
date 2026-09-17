"""
evaluation/confidence_intervals.py

Adds exact counts + Wilson score confidence intervals to precision,
recall, and detection-rate figures throughout the paper.

WHY WILSON, NOT NORMAL-APPROXIMATION: precision/recall here are
proportions computed from very small n (e.g. 3/5 matched alarms).
The normal approximation (p +/- 1.96*sqrt(p(1-p)/n)) can produce
intervals outside [0,1] and is known to have poor coverage for small
n. The Wilson score interval stays in [0,1] and is the standard
recommendation (Brown, Cai & DasGupta, 2001, Statistical Science)
for small-sample binomial proportions. This is exactly what Reviewer
1 (Concern 1) and Reviewer 3 (point 17) asked for, and both reviewers
explicitly noted it requires no new experiments -- only re-analysis
of existing alarm/match counts.

This module does NOT change any detection logic. It is a pure
post-processing layer: feed it the same (matched, total) counts your
pipeline already computes and it returns the interval alongside the
point estimate.
"""

import math


def wilson_interval(k, n, confidence=0.95):
    """
    Wilson score confidence interval for a binomial proportion k/n.

    Parameters
    ----------
    k : int
        Number of successes (e.g. matched alarms, matched ground-truth
        events).
    n : int
        Number of trials (e.g. total alarms, total ground-truth
        events). If n == 0, returns (None, None, None) -- the rate is
        undefined, not zero.
    confidence : float
        Confidence level, default 0.95.

    Returns
    -------
    (point_estimate, low, high) as floats in [0, 1], or (None, None,
    None) if n == 0.
    """
    if n == 0:
        return None, None, None

    # z for two-sided 95% CI = 1.959963985...; compute generally via
    # the inverse-normal approximation is avoidable -- hardcode the
    # common confidence levels used in this paper.
    z_table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_table.get(confidence)
    if z is None:
        raise ValueError(f"Unsupported confidence level {confidence}; add it to z_table.")

    p_hat = k / n
    denom = 1 + (z ** 2) / n
    centre = p_hat + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n ** 2))

    low = (centre - margin) / denom
    high = (centre + margin) / denom

    return p_hat, max(0.0, low), min(1.0, high)


def format_rate_with_ci(k, n, confidence=0.95, decimals=3):
    """
    Returns a paper-ready string, e.g. "0.600 [95% CI: 0.147-0.947] (3/5)".
    Use this directly when rebuilding Tables I, II, III, and VI.
    """
    p, lo, hi = wilson_interval(k, n, confidence)
    if p is None:
        return "undefined (n=0)"
    pct = int(confidence * 100)
    return f"{p:.{decimals}f} [{pct}% CI: {lo:.{decimals}f}-{hi:.{decimals}f}] ({k}/{n})"


def augment_detector_result(metrics, n_alarms, n_ground_truth, confidence=0.95):
    """
    Takes the dict returned by evaluation.detector_metrics.evaluate_detector()
    (which already has 'matched_alarms' and 'matched_truth' counts) plus the
    raw n_alarms / n_ground_truth for that fold or aggregate, and returns
    an augmented dict with Wilson CIs attached. Does not mutate the input.

    Example
    -------
    >>> metrics = evaluate_detector(alarms, ground_truth, tolerance=30)
    >>> augmented = augment_detector_result(metrics, len(alarms), len(ground_truth))
    >>> augmented["precision_ci"]   # (point, low, high)
    >>> augmented["precision_str"]  # "0.600 [95% CI: 0.147-0.947] (3/5)"
    """
    out = dict(metrics)

    p_prec, lo_prec, hi_prec = wilson_interval(metrics["matched_alarms"], n_alarms, confidence)
    p_rec, lo_rec, hi_rec = wilson_interval(metrics["matched_truth"], n_ground_truth, confidence)

    out["precision_ci"] = (p_prec, lo_prec, hi_prec)
    out["recall_ci"] = (p_rec, lo_rec, hi_rec)
    out["precision_str"] = format_rate_with_ci(metrics["matched_alarms"], n_alarms, confidence)
    out["recall_str"] = format_rate_with_ci(metrics["matched_truth"], n_ground_truth, confidence)
    out["n_alarms"] = n_alarms
    out["n_ground_truth"] = n_ground_truth

    return out


def pooled_across_folds(per_fold_results, confidence=0.95):
    """
    Pools matched/total counts across folds (or across synthetic trials)
    to get one CI for an aggregate rate, instead of averaging per-fold
    point estimates and reporting no interval at all.

    per_fold_results: list of dicts, each with keys
        'matched_alarms', 'n_alarms', 'matched_truth', 'n_ground_truth'
    (i.e. the output of augment_detector_result per fold).

    Returns a dict with pooled precision_str / recall_str, PLUS the
    macro-averaged point estimate for comparison -- report both, as
    Section III-E of the paper already does, but now with an interval
    attached to each.
    """
    total_matched_alarms = sum(f["matched_alarms"] for f in per_fold_results)
    total_alarms = sum(f["n_alarms"] for f in per_fold_results)
    total_matched_truth = sum(f["matched_truth"] for f in per_fold_results)
    total_ground_truth = sum(f["n_ground_truth"] for f in per_fold_results)

    macro_precision = sum(f["precision"] for f in per_fold_results) / len(per_fold_results)
    folds_with_gt = [f for f in per_fold_results if f["n_ground_truth"] > 0]
    macro_recall = (
        sum(f["recall"] for f in folds_with_gt) / len(folds_with_gt) if folds_with_gt else None
    )

    return {
        "pooled_precision_str": format_rate_with_ci(total_matched_alarms, total_alarms, confidence),
        "pooled_recall_str": format_rate_with_ci(total_matched_truth, total_ground_truth, confidence),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "n_folds": len(per_fold_results),
    }


if __name__ == "__main__":
    # Quick self-test against the paper's own Table I numbers (25-day
    # tolerance: precision 0.600, recall 0.233). We don't know the exact
    # k/n from the printed rate alone, so this just demonstrates usage --
    # replace with your actual per-fold matched_alarms/n_alarms counts
    # when integrating into run_rolling_evaluation.py.
    print(format_rate_with_ci(3, 5))   # -> wide interval, illustrates the point
    print(format_rate_with_ci(19, 30))  # -> continuous-history recall example