"""
evaluation/detector_metrics.py

FIX over a naive implementation: matching each alarm to the FIRST
ground-truth point within tolerance (in list order) is a real bug
whenever ground-truth points cluster closely together. This version
matches each alarm to its NEAREST ground-truth point within tolerance,
using a greedy nearest-distance assignment so the closest pairs match
first, and no ground-truth point or alarm is claimed twice.
"""


def evaluate_detector(alarms, ground_truth, tolerance=30):
    """
    Greedy nearest-match evaluation between detector alarms and
    ground-truth event positions.

    Returns: precision, recall, avg_delay, matched_alarms, matched_truth
    """
    candidate_pairs = []

    for alarm in alarms:
        for truth in ground_truth:
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

    precision = len(matched_alarms) / len(alarms) if alarms else 0
    recall = len(matched_truth) / len(ground_truth) if ground_truth else 0
    avg_delay = sum(delays) / len(delays) if delays else None

    return {
        "precision": precision,
        "recall": recall,
        "avg_delay": avg_delay,
        "matched_alarms": len(matched_alarms),
        "matched_truth": len(matched_truth)
    }