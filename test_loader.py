from data.loader import load_stock_data

df = load_stock_data("NVDA")

print(df.head())
print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())