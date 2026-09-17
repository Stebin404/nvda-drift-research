# verify_results.ps1
#
# Run from the ROOT of nvda-drift-research, in your activated venv,
# AFTER dropping in all the patched files:
#   drift/kswin_detector.py, retraining/retraining_engine.py,
#   evaluation/detector_metrics.py, evaluation/confidence_intervals.py,
#   evaluation/run_rolling_evaluation.py, evaluation/bridge_experiment.py,
#   check_continuous_kswin.py, bootstrap_retraining_ci.py, rebuild_tables.py
#
# Usage:
#   .\verify_results.ps1
#
# If you get an "execution of scripts is disabled" error, run this once
# in the same PowerShell window first, then re-run the script:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$env:PYTHONPATH = "."
$Pass = $true

function Check-Value {
    param($Label, $Expected, $FilePath)
    $found = Select-String -Path $FilePath -Pattern ([regex]::Escape($Expected)) -Quiet
    if ($found) {
        Write-Host "OK:   $Label -> found '$Expected'" -ForegroundColor Green
    } else {
        Write-Host "FAIL: $Label -> expected '$Expected' NOT found in $FilePath" -ForegroundColor Red
        $script:Pass = $false
    }
}

Write-Host "==================================================================" 
Write-Host "1. Regression test -- matching logic correctness"
Write-Host "=================================================================="
python test_detector_metrics.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: test_detector_metrics.py did not pass." -ForegroundColor Red
    $Pass = $false
} else {
    Write-Host "OK: test_detector_metrics.py passed." -ForegroundColor Green
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "2. Determinism check -- rebuild_tables.py (Tables I, II, VI)"
Write-Host "=================================================================="
python rebuild_tables.py 2>&1 | Out-File -Encoding utf8 _verify_tables_run1.txt
python rebuild_tables.py 2>&1 | Out-File -Encoding utf8 _verify_tables_run2.txt

$diff1 = Compare-Object (Get-Content _verify_tables_run1.txt) (Get-Content _verify_tables_run2.txt)
if (-not $diff1) {
    Write-Host "OK: identical output across two runs -- KSWIN is deterministic." -ForegroundColor Green
} else {
    Write-Host "FAIL: output differs between runs. Check drift/kswin_detector.py has seed=42 default," -ForegroundColor Red
    Write-Host "      and nothing else calls river's KSWIN() directly without a seed." -ForegroundColor Red
    $diff1 | Select-Object -First 20
    $Pass = $false
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "3. Determinism check -- synthetic benchmark (Table III)"
Write-Host "=================================================================="
python benchmarks/run_benchmark.py 2>&1 | Out-File -Encoding utf8 _verify_bench_run1.txt
python benchmarks/run_benchmark.py 2>&1 | Out-File -Encoding utf8 _verify_bench_run2.txt

$diff2 = Compare-Object (Get-Content _verify_bench_run1.txt) (Get-Content _verify_bench_run2.txt)
if (-not $diff2) {
    Write-Host "OK: synthetic benchmark identical across two runs." -ForegroundColor Green
} else {
    Write-Host "FAIL: synthetic benchmark output differs between runs." -ForegroundColor Red
    $diff2 | Select-Object -First 20
    $Pass = $false
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "4. Determinism check -- multi-window retraining (Table V)"
Write-Host "=================================================================="
$retrainScript = @"
from evaluation.rolling_retraining_evaluation import run_rolling_retraining_evaluation, print_rolling_retraining_results
print_rolling_retraining_results(run_rolling_retraining_evaluation())
"@

$retrainScript | python - 2>$null | Select-String "MAE=|MEAN MAE" | Out-File -Encoding utf8 _verify_retrain_run1.txt
$retrainScript | python - 2>$null | Select-String "MAE=|MEAN MAE" | Out-File -Encoding utf8 _verify_retrain_run2.txt

$diff3 = Compare-Object (Get-Content _verify_retrain_run1.txt) (Get-Content _verify_retrain_run2.txt)
if (-not $diff3) {
    Write-Host "OK: retraining comparison identical across two runs." -ForegroundColor Green
    Write-Host "--- MAE values ---"
    Get-Content _verify_retrain_run1.txt
} else {
    Write-Host "FAIL: retraining comparison differs between runs." -ForegroundColor Red
    $diff3
    $Pass = $false
}

Write-Host ""
Write-Host "=================================================================="
Write-Host "5. Value check -- compare against the numbers we locked in"
Write-Host "=================================================================="
Check-Value "Table II Volatility_20 one-sided precision" "0.800 [95% CI: 0.376-0.964] (4/5)" "_verify_tables_run1.txt"
Check-Value "Table VI continuous one-sided precision (tol=30)" "0.500 [95% CI: 0.290-0.710] (9/18)" "_verify_tables_run1.txt"
Check-Value "Table III mean-drift KSWIN detection rate" "detection_rate=1.00 (15/15)" "_verify_bench_run1.txt"
Check-Value "Table III variance-drift KSWIN detection rate" "detection_rate=0.87 (13/15)" "_verify_bench_run1.txt"
Check-Value "Table V drift_triggered window 0 MAE" "0.071582" "_verify_retrain_run1.txt"
Check-Value "Table V drift_triggered window 1 MAE" "0.077924" "_verify_retrain_run1.txt"

Write-Host ""
Write-Host "=================================================================="
Write-Host "6. Bootstrap CI check (Table V significance test)"
Write-Host "=================================================================="
python bootstrap_retraining_ci.py 2>&1 | Tee-Object _verify_bootstrap.txt
Check-Value "drift_triggered vs fixed CI still includes zero" "CI includes zero: True" "_verify_bootstrap.txt"

Write-Host ""
Write-Host "=================================================================="
if ($Pass) {
    Write-Host "ALL CHECKS PASSED. Numbers are reproducible and match the locked-in values." -ForegroundColor Green
} else {
    Write-Host "SOME CHECKS FAILED -- see FAIL lines above before finalizing the paper." -ForegroundColor Red
}
Write-Host "=================================================================="

Remove-Item _verify_tables_run1.txt, _verify_tables_run2.txt, _verify_bench_run1.txt, _verify_bench_run2.txt, _verify_retrain_run1.txt, _verify_retrain_run2.txt, _verify_bootstrap.txt -ErrorAction SilentlyContinue