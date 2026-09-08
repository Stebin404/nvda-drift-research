from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target

df = load_stock_data("NVDA")
df = create_features(df)
df = create_target(df)

print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nShape after features + target:")
print(df.shape)