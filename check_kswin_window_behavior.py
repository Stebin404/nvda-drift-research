from drift.kswin_detector import run_kswin
from river.drift import KSWIN

# Check KSWIN's actual default parameters
detector = KSWIN()
print("KSWIN default window_size:", getattr(detector, "window_size", "unknown"))
print("KSWIN default stat_size:", getattr(detector, "stat_size", "unknown"))

# Test: can KSWIN fire AT ALL on a 20-value window, even with an
# obvious artificial jump?
import numpy as np
obvious_jump = np.concatenate([np.full(10, 0.01), np.full(10, 0.05)])
alarms = run_kswin(obvious_jump)
print(f"\nKSWIN on an obvious 20-value jump stream: {alarms}")
print("(if this is empty, KSWIN structurally cannot fire on windows this small)")