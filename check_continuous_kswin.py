"""
check_continuous_kswin.py

Headless equivalent of app.py's continuous, full-history KSWIN run
(Section IV-E / Table VI's "KSWIN (raw volatility)" row): one
persistent KSWIN detector fed the entire Volatility_20 series with no
fold reinitialization, no LightGBM model involved -- so this result is
NOT affected by the rolling_origin.py / retraining_engine.py purge-gap
fix. This script exists only so the number is reproducible from the
command line instead of by hand via app.py's Streamlit slider.
"""

from data.loader import load_stock_data
from features.engineer import create_features
from drift.kswin_detector import run_kswin
from evaluation.ground_truth_external import get_earnings_ground_truth
from evaluation.detector_metrics import evaluate_detector

df = load_stock_data("NVDA")
df = create_features(df)
df = df.reset_index(drop=True)

volatility_stream = df["Volatility_20"].values
kswin_alarms = run_kswin(volatility_stream)
earnings_ground_truth = get_earnings_ground_truth(df)

for tolerance in [15, 30]:
    metrics = evaluate_detector(kswin_alarms, earnings_ground_truth, tolerance=tolerance)
    print(f"\ntolerance={tolerance}")
    print(f"  alarms={len(kswin_alarms)}  ground_truth={len(earnings_ground_truth)}")
    print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"avg_delay={metrics['avg_delay']}  matched={metrics['matched_truth']}")