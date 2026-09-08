from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from evaluation.rolling_origin import generate_rolling_folds, run_all_folds

df = load_stock_data("NVDA")
df = create_features(df)
df = create_target(df)
df = df.reset_index(drop=True)

print("Full dataset shape:", df.shape)

folds = generate_rolling_folds(n_rows=len(df), n_folds=5, min_train_fraction=0.5, test_fraction=0.08)
for f in folds:
    print(f)

print("\nRunning all 5 folds (this trains 5 LightGBM models, may take a moment)...")
results = run_all_folds(df, build_lightgbm_model, FEATURES, n_folds=5, min_train_fraction=0.5, test_fraction=0.08)

for r in results:
    print(f"Fold {r['fold_id']}: MAE={r['mae']:.6f}, test_rows={len(r['error_stream'])}, "
          f"test_start={r['test_start_full_df_index']}, test_end={r['test_end_full_df_index']}")