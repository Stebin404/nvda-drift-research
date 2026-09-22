import numpy as np


def create_features(df):

    df = df.copy()

    # Log Return
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # Rolling Volatility
    df["Volatility_20"] = (
        df["Log_Return"]
        .rolling(20)
        .std()
    )

    # Moving Averages
    df["MA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    # Volume Change
    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    # Momentum
    df["Momentum_10"] = (
        df["Close"] -
        df["Close"].shift(10)
    )

    df["Price_to_MA20"] = df["Close"] / df["MA_20"]
    df["Price_to_MA50"] = df["Close"] / df["MA_50"]
    df["Momentum_10_Pct"] = df["Close"] / df["Close"].shift(10) - 1

    df.dropna(inplace=True)

    return df