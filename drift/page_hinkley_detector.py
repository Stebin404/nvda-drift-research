from river.drift import PageHinkley


def run_page_hinkley(
        error_stream,
        delta=0.005,
        threshold=50
):

    detector = PageHinkley(
        delta=delta,
        threshold=threshold
    )

    alarms = []

    for i, value in enumerate(error_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms