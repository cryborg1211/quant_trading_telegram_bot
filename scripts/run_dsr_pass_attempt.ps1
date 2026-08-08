$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\dsr_pass_attempt_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$ckpt = "models/saved/t20_extended_oos_checkpoint.joblib"

# STEP 1 -- train T+20 with an EXTENDED OOS window (train_frac 0.70 -> 0.46,
# T ~919 -> ~1500 days). Written to a SEPARATE --out path so the production
# checkpoint (models/saved/v3_training_checkpoint.joblib) is never touched.
Add-Content $log "===== STEP 1: train T+20 extended-OOS checkpoint : $(Get-Date -Format o) ====="
cmd /c "`"$py`" train_models.py --tb-horizon 20 --train-frac 0.46 --out `"$ckpt`" >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: training (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== STEP 1 DONE : $(Get-Date -Format o) ====="

# STEP 2 -- seed-averaged sweep + honest DSR/PBO.
Add-Content $log "`n===== STEP 2: seed-averaged sweep + DSR : $(Get-Date -Format o) ====="
cmd /c "`"$py`" scripts/analyze_dsr_pass_attempt.py `"$ckpt`" >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: sweep (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== ALL DONE : $(Get-Date -Format o) ====="
