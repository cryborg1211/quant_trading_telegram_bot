$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\impulse_attack_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== Impulse fast-attack research (read-only) : $(Get-Date -Format o) ====="
cmd /c "`"$py`" scripts/analyze_impulse_attack.py >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) {
    Add-Content $log "FAILED (exit $LASTEXITCODE)"
    exit 1
}
Add-Content $log "===== DONE : $(Get-Date -Format o) ====="
