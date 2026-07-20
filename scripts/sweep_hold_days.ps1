$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\sweep_hold_days_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== hold-days sweep (T+5 checkpoint, thr=0.44, no-save) : $(Get-Date -Format o) ====="
foreach ($hold in @(20, 40)) {
    Add-Content $log "`n===== hold_days=$hold : $(Get-Date -Format o) ====="
    cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days $hold --sweep-thresholds 0.44 --no-save >> `"$log`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Add-Content $log "FAILED: hold_days=$hold (exit $LASTEXITCODE)"
        exit 1
    }
    Add-Content $log "===== hold_days=$hold DONE : $(Get-Date -Format o) ====="
}
Add-Content $log "`nHOLD-DAYS SWEEP COMPLETE : $(Get-Date -Format o)"
