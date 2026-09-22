import os
import yfinance as yf
import pandas as pd

CACHE_DIR = os.path.dirname(__file__)


def _cache_path_for(ticker):
    return os.path.join(CACHE_DIR, f"{ticker.lower()}_ohlcv_frozen.csv")


def load_stock_data(
        ticker="NVDA",
        start="2018-01-01",
        end="2025-12-31"
):
    # Frozen snapshot for reproducibility: Yahoo Finance retroactively
    # revises historical adjusted-close values (e.g. after new dividend
    # data attaches), so re-pulling the "same" date range on different
    # days can silently change every downstream number (see repo notes /
    # paper Limitations). Once a frozen CSV exists for a ticker, it is
    # always preferred over a live pull, so every script in this repo
    # reads the exact same data from here to submission.
    #
    # NOTE (multi-asset expansion): this used to only cache NVDA
    # (hardcoded ticker == "NVDA" check). Any other ticker silently fell
    # through to an uncached live pull, which is a reproducibility gap,
    # not just an inconvenience -- fixed by keying the cache filename
    # off the ticker itself. Existing nvda_ohlcv_frozen.csv is read
    # unchanged (case-insensitive match on "nvda").
    cache_path = _cache_path_for(ticker)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    df.to_csv(cache_path, index=False)

    return df