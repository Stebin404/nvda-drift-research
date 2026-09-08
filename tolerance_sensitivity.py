from evaluation.run_rolling_evaluation import run_full_rolling_evaluation, print_results

for tolerance in [10, 15, 20, 25, 30]:
    print(f"\n{'#'*70}\nTOLERANCE = {tolerance}\n{'#'*70}")
    results = run_full_rolling_evaluation(tolerance=tolerance)
    print_results(results)