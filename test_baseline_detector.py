from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from drift.naive_threshold_detector import run_naive_threshold
from evaluation.rolling_origin import generate_rolling_folds
from evaluation.ground_truth_external import get_earnings_ground_truth_for_test_fold
from evaluation.detector_metrics import evaluate_detector

full_df = load_stock_data("NVDA")
full_df = create_features(full_df)
full_df = create_target(full_df)
full_df = full_df.reset_index(drop=True)

folds = generate_rolling_folds(n_rows=len(full_df), n_folds=5, min_train_fraction=0.5, test_fraction=0.08)

precisions, recalls, delays = [], [], []

for fold in folds:
    stream = full_df["Volatility_20"].iloc[fold["test_start"]:fold["test_end"]].reset_index(drop=True).values
    alarms = run_naive_threshold(stream, window=90, k=2.0)

    ground_truth = get_earnings_ground_truth_for_test_fold(
        full_df,
        test_fold_start_index=fold["test_start"],
        test_fold_end_index=fold["test_end"],
        csv_path="data/earnings_dates_FINAL.csv",
    )

    metrics = evaluate_detector(alarms, ground_truth, tolerance=30)
    print(f"fold {fold['fold_id']}: alarms={len(alarms)} gt={len(ground_truth)} "
          f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} delay={metrics['avg_delay']}")

    precisions.append(metrics["precision"])
    if len(ground_truth) > 0:
        recalls.append(metrics["recall"])
    if metrics["avg_delay"] is not None:
        delays.append(metrics["avg_delay"])

import numpy as np
print(f"\nAGGREGATE: precision={np.mean(precisions):.3f} recall={np.mean(recalls) if recalls else 'undefined'} "
      f"delay={np.mean(delays) if delays else None}")