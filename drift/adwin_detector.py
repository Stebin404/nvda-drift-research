from river.drift import ADWIN


def run_adwin(error_stream, delta=0.002):

    detector = ADWIN(delta=delta)

    alarms = []

    for i, value in enumerate(error_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms