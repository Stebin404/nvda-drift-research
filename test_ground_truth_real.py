from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from evaluation.ground_truth_external import get_earnings_ground_truth

df = load_stock_data("NVDA")
df = create_features(df)
df = create_target(df)
df = df.reset_index(drop=True)

print("Full dataframe shape:", df.shape)
print("Date range:", df['Date'].min(), "to", df['Date'].max())

indices = get_earnings_ground_truth(df)
print(f"\nMapped {len(indices)} earnings dates to row indices:")
print(indices)

print("\nCorresponding actual dates in price data:")
for idx in indices:
    print(f"  row {idx} -> {df.iloc[idx]['Date']}")
    