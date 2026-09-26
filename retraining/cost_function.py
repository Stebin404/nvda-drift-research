"""
retraining/cost_function.py

TODO item #5: the paper reports "drift-triggered retraining uses about
one-third as many retrains as fixed-interval" as an efficiency finding
but never formalizes efficiency -- it's a qualitative comparison of two
numbers, not a cost trade-off. This module defines the simplest
defensible cost function that trades forecast error off against
retraining operations:

    J(lambda) = MAE + lambda * n_retrains

lambda is a per-retrain cost, in the same units as MAE, that a
practitioner would set based on their own retraining compute/ops
budget -- there is no single "correct" lambda, which is exactly why we
report J across a small sweep rather than picking one value and
declaring a winner. This is intentionally NOT a full Pareto-frontier
analysis (TODO explicitly scopes this as optional); it is one sentence
of extra rigor beyond "fewer retrains = cheaper."
"""


def compute_cost(mae, n_retrains, lam):
    """J = MAE + lambda * n_retrains."""
    return mae + lam * n_retrains


def cost_table(strategy_results, lambdas):
    """
    strategy_results: dict[str, dict] with at least 'mae' and
    'n_retrains' keys per strategy (i.e. directly the per-strategy
    dicts returned by retraining.retraining_engine.walk_forward_predict
    / compare_retraining_strategies).

    Returns dict[lambda][strategy] = J.
    """
    table = {}
    for lam in lambdas:
        table[lam] = {
            strategy: compute_cost(data["mae"], data["n_retrains"], lam)
            for strategy, data in strategy_results.items()
        }
    return table


def print_cost_table(table, title="RETRAINING COST SWEEP  J = MAE + lambda * n_retrains"):
    strategies = list(next(iter(table.values())).keys())
    print(f"\n{title}")
    header = f"{'lambda':>10} | " + " | ".join(f"{s:>16}" for s in strategies) + " | best"
    print(header)
    print("-" * len(header))
    for lam, row in table.items():
        best = min(row, key=row.get)
        cells = " | ".join(f"{row[s]:>16.5f}" for s in strategies)
        print(f"{lam:>10.5f} | {cells} | {best}")
