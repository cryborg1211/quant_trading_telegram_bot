$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\retrain_t5_resume_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== T+5 backtest + save (resume) : $(Get-Date -Format o) ====="
cmd /c "`"$py`" run_backtest.py --mode tranche --hold-days 30 >> `"$log`" 2>&1"
$code = $LASTEXITCODE
if ($code -ne 0) {
    Add-Content $log "FAILED: T+5 backtest + save (exit $code)"
    exit 1
}
Add-Content $log "===== T+5 backtest + save DONE : $(Get-Date -Format o) ====="
Add-Content $log "`nALL RETRAINS COMPLETE : $(Get-Date -Format o)"
