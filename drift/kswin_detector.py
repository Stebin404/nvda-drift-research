from river.drift import KSWIN

# River's KSWIN performs an internal random sub-sample of size
# `stat_size` from its window on every test (part of the
# Kolmogorov-Smirnov statistic computation). With seed=None (the
# river default), this makes KSWIN's alarm timings NON-DETERMINISTIC
# -- rerunning the exact same script on the exact same frozen data can
# and does produce different alarms, and therefore different
# precision/recall/delay numbers, across runs. This was true of the
# ORIGINAL (pre-revision) codebase too; it was simply never noticed
# because the paper's numbers were generated from a single run.
#
# Fixing this with a fixed seed makes every downstream result
# reproducible, as required by Reviewer 1 (Concern 4) and Reviewer 3
# (point 19, "no seeds provided").
DEFAULT_KSWIN_SEED = 42


def run_kswin(error_stream, seed=DEFAULT_KSWIN_SEED):

    detector = KSWIN(seed=seed)

    alarms = []

    for i, value in enumerate(error_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms