import numpy as np


def create_target(df):

    df = df.copy()

    df["Target"] = (
        np.log(
            df["Close"].shift(-5)
            / df["Close"]
        )
    )

    df.dropna(inplace=True)

    return df