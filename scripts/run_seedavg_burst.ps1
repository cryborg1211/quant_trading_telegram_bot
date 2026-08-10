$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\seedavg_burst_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Add-Content $log "===== Seed-avg x burst sizing (read-only) : $(Get-Date -Format o) ====="
cmd /c "`"$py`" scripts/analyze_seedavg_burst_sizing.py >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Add-Content $log "FAILED (exit $LASTEXITCODE)"; exit 1 }
Add-Content $log "===== DONE : $(Get-Date -Format o) ====="
