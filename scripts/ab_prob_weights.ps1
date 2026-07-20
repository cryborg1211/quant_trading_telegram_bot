$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\ab_prob_weights_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== prob-weights A/B (T+5 ckpt, hold=30, thr=0.44, no-save) : $(Get-Date -Format o) ====="
cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days 30 --sweep-thresholds 0.44 --prob-weights --no-save >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: prob-weights A/B (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== prob-weights A/B DONE : $(Get-Date -Format o) ====="
