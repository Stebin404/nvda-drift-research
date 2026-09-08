from river.drift import KSWIN


def run_kswin(error_stream):

    detector = KSWIN()

    alarms = []

    for i, value in enumerate(error_stream):

        detector.update(float(value))

        if detector.drift_detected:
            alarms.append(i)

    return alarms