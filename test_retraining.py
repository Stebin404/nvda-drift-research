from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from models.lightgbm_model import build_lightgbm_model, FEATURES
from retraining.retraining_engine import compare_retraining_strategies, print_comparison

df = load_stock_data("NVDA")
df = create_features(df)
df = create_target(df)
df = df.reset_index(drop=True)

EVAL_WINDOW = 400
initial_train_size = len(df) - EVAL_WINDOW

print(f"Total rows: {len(df)}, initial_train_size: {initial_train_size}, evaluating last {EVAL_WINDOW} rows")

results = compare_retraining_strategies(
    df, FEATURES, "Target", build_lightgbm_model,
    initial_train_size=initial_train_size,
    fixed_interval=30,
)

print_comparison(results)