#!/usr/bin/env bash
# verify_results.sh
#
# Run from the ROOT of nvda-drift-research, AFTER dropping in:
#   drift/kswin_detector.py          (seeded)
#   retraining/retraining_engine.py  (seeded)
#   evaluation/detector_metrics.py   (one-sided matching)
#   evaluation/confidence_intervals.py
#   evaluation/run_rolling_evaluation.py
#   evaluation/bridge_experiment.py
#   check_continuous_kswin.py
#   bootstrap_retraining_ci.py       (updated MAE values)
#   rebuild_tables.py
#
# Checks two things:
#   1. DETERMINISM -- run the same script twice, diff the output.
#      If they differ, something is still unseeded somewhere.
#   2. VALUES -- grep the output for the exact numbers we locked in,
#      so you can visually confirm nothing drifted.

set -uo pipefail
export PYTHONPATH="."

PASS=1

echo "=================================================================="
echo "1. Regression test -- matching logic correctness"
echo "=================================================================="
python test_detector_metrics.py
if [ $? -ne 0 ]; then
    echo "FAIL: test_detector_metrics.py did not pass."
    PASS=0
else
    echo "OK: test_detector_metrics.py passed."
fi

echo ""
echo "=================================================================="
echo "2. Determinism check -- rebuild_tables.py (Tables I, II, VI)"
echo "=================================================================="
python rebuild_tables.py > _verify_tables_run1.txt 2>&1
python rebuild_tables.py > _verify_tables_run2.txt 2>&1

if diff -q _verify_tables_run1.txt _verify_tables_run2.txt > /dev/null; then
    echo "OK: identical output across two runs -- KSWIN is deterministic."
else
    echo "FAIL: output differs between runs. Check that drift/kswin_detector.py"
    echo "      has seed=42 as the default, and that nothing else calls"
    echo "      river's KSWIN() directly without a seed."
    diff _verify_tables_run1.txt _verify_tables_run2.txt | head -20
    PASS=0
fi

echo ""
echo "=================================================================="
echo "3. Determinism check -- synthetic benchmark (Table III)"
echo "=================================================================="
python benchmarks/run_benchmark.py > _verify_bench_run1.txt 2>&1
python benchmarks/run_benchmark.py > _verify_bench_run2.txt 2>&1

if diff -q _verify_bench_run1.txt _verify_bench_run2.txt > /dev/null; then
    echo "OK: synthetic benchmark identical across two runs."
else
    echo "FAIL: synthetic benchmark output differs between runs."
    diff _verify_bench_run1.txt _verify_bench_run2.txt | head -20
    PASS=0
fi

echo ""
echo "=================================================================="
echo "4. Determinism check -- multi-window retraining (Table V)"
echo "=================================================================="
python -c "
from evaluation.rolling_retraining_evaluation import run_rolling_retraining_evaluation, print_rolling_retraining_results
print_rolling_retraining_results(run_rolling_retraining_evaluation())
" 2>/dev/null | grep -E "MAE=|MEAN MAE" > _verify_retrain_run1.txt

python -c "
from evaluation.rolling_retraining_evaluation import run_rolling_retraining_evaluation, print_rolling_retraining_results
print_rolling_retraining_results(run_rolling_retraining_evaluation())
" 2>/dev/null | grep -E "MAE=|MEAN MAE" > _verify_retrain_run2.txt

if diff -q _verify_retrain_run1.txt _verify_retrain_run2.txt > /dev/null; then
    echo "OK: retraining comparison identical across two runs."
    echo "--- MAE values ---"
    cat _verify_retrain_run1.txt
else
    echo "FAIL: retraining comparison differs between runs."
    diff _verify_retrain_run1.txt _verify_retrain_run2.txt
    PASS=0
fi

echo ""
echo "=================================================================="
echo "5. Value check -- compare against the numbers we locked in"
echo "=================================================================="

check_value() {
    local label="$1"
    local expected="$2"
    local file="$3"
    if grep -qF "$expected" "$file"; then
        echo "OK:   $label -> found '$expected'"
    else
        echo "FAIL: $label -> expected '$expected' NOT found in $file"
        PASS=0
    fi
}

check_value "Table II Volatility_20 one-sided precision" "0.800 [95% CI: 0.376-0.964] (4/5)" _verify_tables_run1.txt
check_value "Table VI continuous one-sided precision (tol=30)" "0.500 [95% CI: 0.290-0.710] (9/18)" _verify_tables_run1.txt
check_value "Table III mean-drift KSWIN detection rate" "detection_rate=1.00 (15/15)" _verify_bench_run1.txt
check_value "Table III variance-drift KSWIN detection rate" "detection_rate=0.87 (13/15)" _verify_bench_run1.txt
check_value "Table V drift_triggered window 0 MAE" "0.071582" _verify_retrain_run1.txt
check_value "Table V drift_triggered window 1 MAE" "0.077924" _verify_retrain_run1.txt

echo ""
echo "=================================================================="
echo "6. Bootstrap CI check (Table V significance test)"
echo "=================================================================="
python bootstrap_retraining_ci.py | tee _verify_bootstrap.txt
check_value "drift_triggered vs fixed CI still includes zero" "CI includes zero: True" _verify_bootstrap.txt

echo ""
echo "=================================================================="
if [ "$PASS" -eq 1 ]; then
    echo "ALL CHECKS PASSED. Numbers are reproducible and match the locked-in values."
else
    echo "SOME CHECKS FAILED -- see FAIL lines above before finalizing the paper."
fi
echo "=================================================================="

# Cleanup temp files (comment out these lines if you want to inspect them)
rm -f _verify_tables_run1.txt _verify_tables_run2.txt _verify_bench_run1.txt _verify_bench_run2.txt _verify_retrain_run1.txt _verify_retrain_run2.txt _verify_bootstrap.txt