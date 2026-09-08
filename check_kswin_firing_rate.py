from data.loader import load_stock_data
from features.engineer import create_features
from features.target import create_target
from drift.kswin_detector import run_kswin

df = load_stock_data("NVDA")
df = create_features(df)
df = create_target(df)
df = df.reset_index(drop=True)

volatility = df["Volatility_20"].values

eval_window = 400
initial_train_size = len(df) - eval_window
drift_check_window = 150

fire_count = 0
total_checks = 0

for i in range(initial_train_size, len(df)):
    window_start = max(0, i - drift_check_window)
    recent_volatility = volatility[window_start:i + 1]
    if len(recent_volatility) >= 130:
        total_checks += 1
        alarms = run_kswin(recent_volatility)
        if len(alarms) > 0:
            fire_count += 1

print(f"Total checks: {total_checks}")
print(f"Fired (alarms > 0): {fire_count}")
print(f"Fire rate: {fire_count / total_checks:.1%}" if total_checks else "no checks ran")