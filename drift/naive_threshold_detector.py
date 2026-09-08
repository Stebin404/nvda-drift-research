"""
drift/naive_threshold_detector.py

A trivial, non-streaming-library baseline: fires an alarm whenever the
input stream exceeds its own trailing rolling mean by k standard
deviations. This exists purely to give KSWIN's detection rate a lower
bound to be compared against -- "detects something" is only a
meaningful claim if it beats a naive threshold rule, not just ADWIN
and Page-Hinkley.
"""

import numpy as np


def run_naive_threshold(stream, window=90, k=2.0):
    stream = np.asarray(stream, dtype=float)
    alarms = []

    for i in range(window, len(stream)):
        trailing = stream[i - window:i]
        mean = trailing.mean()
        std = trailing.std()
        if std == 0:
            continue
        if stream[i] > mean + k * std:
            alarms.append(i)

    return alarms