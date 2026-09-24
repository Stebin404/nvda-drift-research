from river.drift import PageHinkley


def run_page_hinkley(
        error_stream,
        delta=0.005,
        threshold=50,
        min_instances=30
):
    """
    min_instances added (TODO item #1, hyperparameter sensitivity sweep):
    river's PageHinkley exposes min_instances as a hyperparameter (the
    number of instances before the detector's mean/threshold checks
    become active) but the original wrapper hardcoded it to the library
    default with no way to sweep it. Kept as a keyword with the same
    default (30) so every existing call site is unaffected.
    """

    detector = PageHinkley(
        delta=delta,
        threshold=threshold,
        min_instances=min_instances
    )

    alarms = []

    for i, value in enumerate(error_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms
