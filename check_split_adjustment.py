"""
One-off diagnostic: confirm that yfinance's auto_adjust=True split
adjustment does not introduce an artificial spike in Volatility_20
around NVDA's two stock splits in the study period:
  - 2021-07-20 (4-for-1)
  - 2024-06-10 (10-for-1)

Logic: a correct split adjustment retroactively rescales ALL pre-split
prices by the split ratio, uniformly. This means Log_Return (which is
a ratio of consecutive Close prices) should be UNAFFECTED by the
adjustment on any single day, including the split day itself, because
both the numerator and denominator get scaled by the same factor
except exactly at the boundary day, where yfinance's adjustment
already accounts for it correctly. If adjustment were done wrong (or
not done, or done twice), we'd see an obvious one-day return spike of
roughly -75% (4:1) or -90% (10:1) on the split day, and Volatility_20
would spike for the following 20 days as that one huge return sits in
the rolling window.
"""
import pandas as pd
from data.loader import load_stock_data
from features.engineer import create_features

df = load_stock_data("NVDA")
df = create_features(df)
df = df.reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])

split_dates = {
    "4-for-1 (2021-07-20)": pd.Timestamp("2021-07-20"),
    "10-for-1 (2024-06-10)": pd.Timestamp("2024-06-10"),
}

for label, split_date in split_dates.items():
    window = df[(df["Date"] >= split_date - pd.Timedelta(days=15)) &
                (df["Date"] <= split_date + pd.Timedelta(days=25))]
    print(f"\n=== {label} ===")
    print(window[["Date", "Close", "Log_Return", "Volatility_20"]].to_string(index=False))

    max_abs_return = window["Log_Return"].abs().max()
    max_vol = window["Volatility_20"].max()
    print(f"\nMax |Log_Return| in window: {max_abs_return:.4f}")
    print(f"Max Volatility_20 in window: {max_vol:.4f}")