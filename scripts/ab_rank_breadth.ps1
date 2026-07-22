$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\ab_rank_breadth_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== rank_breadth A/B (T+5 ckpt, hold=30, thr=0.44, no-save) : $(Get-Date -Format o) ====="
Add-Content $log "`n----- BASELINE: cross_sectional (default) -----"
cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days 30 --sweep-thresholds 0.44 --no-save >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: baseline (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "`n----- CANDIDATE: rank_breadth -----"
cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days 30 --sweep-thresholds 0.44 --admission-mode rank_breadth --no-save >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED: rank_breadth (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== DONE : $(Get-Date -Format o) ====="
