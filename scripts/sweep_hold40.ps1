$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\sweep_hold40_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== hold_days=40 (T+5 ckpt, thr=0.44, no-save) : $(Get-Date -Format o) ====="
cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days 40 --sweep-thresholds 0.44 --no-save >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: hold_days=40 (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== hold_days=40 DONE : $(Get-Date -Format o) ====="
