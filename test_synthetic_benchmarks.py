from benchmarks.synthetic_drift import generate_trial_grid

trials = generate_trial_grid(
    drift_types=("mean", "variance", "combined"),
    changepoint_fractions=(0.3, 0.5, 0.7),
    seeds=(1, 2, 3),
)

print(f"Total trials generated: {len(trials)}")
print(f"(Expected: 3 drift types x 3 changepoint fractions x 3 seeds = 27)")

# Spot-check a few trials to confirm they're actually different
for t in trials[:5]:
    print(f"\n{t['drift_type']}, changepoint_fraction={t['changepoint_fraction']}, seed={t['seed']}")
    print(f"  Changepoint at index: {t['changepoint']}")
    print(f"  Stream length: {len(t['stream'])}")
    print(f"  Mean before: {t['stream'][:t['changepoint']].mean():.3f}")
    print(f"  Mean after:  {t['stream'][t['changepoint']:].mean():.3f}")
    print(f"  Std before:  {t['stream'][:t['changepoint']].std():.3f}")
    print(f"  Std after:   {t['stream'][t['changepoint']:].std():.3f}")