$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\seed_average_clean_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$ckpt = "models/saved/t20_std_checkpoint.joblib"

# CLEAN ONE-VARIABLE TEST. The 08-08 attempt changed seed-averaging AND
# train_frac at once, so its Sharpe collapse could not be attributed. This
# holds train_frac at the PRODUCTION 0.70 (T=919 OOS) and changes ONLY the
# seed handling: average the 4 seeds into one deployable model instead of
# reporting the best of 4. That legitimately drops the DSR multiplicity from
# n_trials=20 to 5 (hurdle 0.995 -> 0.625, required Sharpe 1.86 -> 1.49).
# Separate --out path so the production checkpoint is never touched.
Add-Content $log "===== STEP 1: train T+20 @ train_frac=0.70 : $(Get-Date -Format o) ====="
cmd /c "`"$py`" train_models.py --tb-horizon 20 --train-frac 0.70 --out `"$ckpt`" >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: training (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== STEP 1 DONE : $(Get-Date -Format o) ====="

Add-Content $log "`n===== STEP 2: seed-averaged sweep + DSR : $(Get-Date -Format o) ====="
cmd /c "`"$py`" scripts/analyze_dsr_pass_attempt.py `"$ckpt`" >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: sweep (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== ALL DONE : $(Get-Date -Format o) ====="
