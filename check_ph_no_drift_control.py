import numpy as np
from drift.page_hinkley_detector import run_page_hinkley

# Control: a stream with NO drift at all, but at the SAME high
# variance as the "after" segment of the synthetic variance-drift
# trials (scale=5, matching generate_variance_drift_stream's default
# magnitude). If Page-Hinkley fires repeatedly on this -- where there
# is genuinely nothing to detect -- that confirms the over-triggering
# is a threshold/scale mismatch, not real sensitivity to variance
# drift specifically.
rng = np.random.default_rng(99)
no_drift_high_variance = rng.normal(loc=0, scale=5, size=2000)

alarms = run_page_hinkley(no_drift_high_variance)
print(f"Page-Hinkley alarms on a NO-DRIFT, constant-high-variance stream: {len(alarms)}")
print(f"Alarm positions: {alarms[:20]}{'...' if len(alarms) > 20 else ''}")

# Compare: same check at the LOW variance scale (matching the
# "before" segment, scale=1) to see if alarm count drops accordingly.
no_drift_low_variance = rng.normal(loc=0, scale=1, size=2000)
alarms_low = run_page_hinkley(no_drift_low_variance)
print(f"\nPage-Hinkley alarms on a NO-DRIFT, constant-LOW-variance stream: {len(alarms_low)}")