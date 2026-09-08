import yfinance as yf
import pandas as pd


def load_stock_data(
        ticker="NVDA",
        start="2018-01-01",
        end="2025-12-31"
):

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df