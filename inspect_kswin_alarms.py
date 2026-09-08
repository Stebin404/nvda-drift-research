from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from drift.kswin_detector import run_kswin
from evaluation.rolling_origin import generate_rolling_folds, run_rolling_fold
from evaluation.ground_truth_external import get_earnings_ground_truth_for_test_fold

full_df = load_stock_data("NVDA")
full_df = create_features(full_df)
full_df = create_target(full_df)
full_df = full_df.reset_index(drop=True)

folds = generate_rolling_folds(n_rows=len(full_df), n_folds=5, min_train_fraction=0.5, test_fraction=0.08)

for fold in folds:
    fold_result = run_rolling_fold(full_df, fold, build_lightgbm_model, FEATURES)
    alarms = run_kswin(fold_result["error_stream"])

    ground_truth = get_earnings_ground_truth_for_test_fold(
        full_df,
        test_fold_start_index=fold["test_start"],
        test_fold_end_index=fold["test_end"],
        csv_path="data/earnings_dates_FINAL.csv",
    )

    print(f"\nFold {fold['fold_id']}:")
    print(f"  KSWIN alarms (test-fold-relative positions): {alarms}")
    print(f"  Ground truth (test-fold-relative positions): {ground_truth}")

    if alarms and ground_truth:
        for a in alarms:
            distances = [abs(a - g) for g in ground_truth]
            print(f"    alarm at {a}: distances to ground truth = {distances}, min = {min(distances)}")
            