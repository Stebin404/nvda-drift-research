import numpy as np

from drift.adwin_detector import run_adwin
from drift.page_hinkley_detector import run_page_hinkley
from drift.kswin_detector import run_kswin

# A simple synthetic stream with an obvious mean jump partway through,
# just to confirm the wrappers run and CAN detect something -- this is
# not a real evaluation, just a smoke test.
np.random.seed(0)
stream = np.concatenate([
    np.random.normal(0, 1, 500),
    np.random.normal(5, 1, 500)
])

adwin_alarms = run_adwin(stream)
ph_alarms = run_page_hinkley(stream)
kswin_alarms = run_kswin(stream)

print("ADWIN alarms:", adwin_alarms)
print("Page-Hinkley alarms:", ph_alarms)
print("KSWIN alarms:", kswin_alarms)