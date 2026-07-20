$py = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
Set-Location "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$log = "logs\pipeline_verify_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Add-Content $log "===== full_pipeline verify run : $(Get-Date -Format o) ====="
cmd /c "`"$py`" main.py --task full_pipeline >> `"$log`" 2>&1"
$code = $LASTEXITCODE
if ($code -ne 0) {
    Add-Content $log "FAILED: full_pipeline (exit $code)"
    exit 1
}
Add-Content $log "===== full_pipeline verify run DONE : $(Get-Date -Format o) ====="
