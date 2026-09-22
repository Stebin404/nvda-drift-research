from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "Log_Return",
    "Volatility_20",
    "MA_20",
    "MA_50",
    "Volume_Change",
    "Momentum_10"
]

FEATURES_SCALE_FREE = [
    "Log_Return",
    "Volatility_20",
    "Price_to_MA20",
    "Price_to_MA50",
    "Volume_Change",
    "Momentum_10_Pct"
]


def build_lightgbm_model():
    """
    Single source of truth for LightGBM hyperparameters. Both
    train_lightgbm() (original single-split entry point, kept for
    backward compatibility) and evaluation/rolling_origin.py's
    fold-by-fold training call this, so a hyperparameter change only
    needs to happen here once.
    """
    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )


def train_lightgbm(df):
    """
    Original behavior: single chronological 80/20 split. Kept
    unchanged so any existing script that imports this function
    continues to work without modification.
    """
    X = df[FEATURES]
    y = df["Target"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = build_lightgbm_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    error_stream = abs(y_test - preds)

    return {
        "model": model,
        "mae": mae,
        "predictions": preds,
        "actuals": y_test,
        "error_stream": error_stream,
        "dates": y_test.index
    }