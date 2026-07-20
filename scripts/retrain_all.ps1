$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\retrain_all_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Step($name, $argLine) {
    Add-Content $log "`n===== $name : $(Get-Date -Format o) ====="
    cmd /c "`"$py`" $argLine >> `"$log`" 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Add-Content $log "FAILED: $name (exit $code)"
        Write-Host "FAILED: $name (exit $code)"
        exit 1
    }
    Add-Content $log "===== $name DONE : $(Get-Date -Format o) ====="
}

Step "T+20 train checkpoint" "train_models.py --tb-horizon 20"
Step "T+20 backtest + save" "run_backtest.py --mode tranche --hold-days 30"
Step "T+5 train checkpoint" "train_models.py --tb-horizon 5"
Step "T+5 backtest + save" "run_backtest.py --mode tranche --hold-days 30"

Add-Content $log "`nALL RETRAINS COMPLETE : $(Get-Date -Format o)"
Write-Host "ALL RETRAINS COMPLETE"
