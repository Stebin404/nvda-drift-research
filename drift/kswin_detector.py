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


def run_kswin_seeded(train_stream, test_stream, seed=DEFAULT_KSWIN_SEED):
    """
    TODO item #4 (KSWIN state-seeding / warm-up redesign).

    The fold-based tournament (evaluation/run_rolling_evaluation.py,
    evaluation/bridge_experiment.py) calls run_kswin() on ONLY the
    test-fold stream, so a fresh KSWIN instance is reinitialized at
    the start of every fold and has to spend its warm-up period
    (river's default window_size=100) inside the ~150-200-row test
    window before it can be sensitive to anything. Section V-D already
    identifies this as the most likely reason fold-based recall
    (0.077 at tolerance=30) is so much lower than continuous,
    full-history recall (0.300) -- this function lets that hypothesis
    actually be tested rather than just argued for.

    `train_stream` is the SAME data the fold's model was allowed to
    train on (its purged training window, for the residual stream; its
    full pre-test history, for a raw feature stream) -- i.e. warm-up
    happens on data the detector is allowed to see, never on live test
    data. The detector consumes train_stream first (update-only, no
    alarms recorded, exactly mirroring how the persistent detector in
    retraining/retraining_engine.py is warmed up on
    volatility[:initial_train_size] before evaluation begins), then
    continues on test_stream, recording alarms as positions relative
    to the START of test_stream -- identical index convention to
    run_kswin(test_stream), so results are directly comparable.
    """
    detector = KSWIN(seed=seed)

    for value in train_stream:
        detector.update(float(value))

    alarms = []

    for i, value in enumerate(test_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms
