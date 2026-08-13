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

# SERVE-PARITY SWEEP (12-08-26). `--serve-parity` makes the sweep optimise the
# threshold under the conditions serve ACTUALLY runs:
#   * --gate-offset 0     the swept up_threshold IS the engine's trading gate.
#                         Previously the engine traded 5pp below it, so the value
#                         serve admits on was never the value that was optimised.
#   * the four production defensive layers PRESENT (sector cap 2, open-cohort
#     dedup, hysteresis 2, drift+breadth exposure brake). None was in the
#     backtest before, and measuring them found every one costs Sharpe while
#     buying drawdown — so the threshold must be chosen with them switched on,
#     not tuned bare and then deployed behind them.
#
# Grid is 0.46..0.41 in 0.01 steps: SIX levels, deliberately no more than the
# five it replaces, because DSR penalises trial count and a wider grid makes any
# winner cheaper than it looks. The 12-08 sweep showed the useful region sits
# below serve's 0.46, which is why the grid is centred there rather than
# reaching up to 0.50.
#
# GOLDEN selection is capped by CONFIG.trading.golden_max_mean_dd_pp (14.0), read
# automatically — no flag needed here. It exists because the 12-08 full sweep
# found the threshold does NOT move Sharpe (all six levels span 0.056 while ONE
# level's 4-seed spread is 0.30) while mean DD IS monotone in the gate, so an
# unconstrained max-PnL objective walks toward maximum drawdown on a run nobody
# is watching. 14.0 admits the 0.43 arm the dry sweep chose (-13.88%) and blocks
# 0.42 (-14.83%) / 0.41 (-16.20%). Watch the log for
# `GOLDEN drawdown budget BINDING` — that means PnL wanted a deeper arm.
$sweep = "0.46,0.45,0.44,0.43,0.42,0.41"
$btArgs = "--mode tranche --hold-days 30 --serve-parity --sweep-thresholds $sweep"

Step "MR-LGBM retrain" "src/models/train_mr_lgbm.py"
Step "T+20 train checkpoint" "train_models.py --tb-horizon 20"
Step "T+20 backtest + save" "run_backtest.py $btArgs"
Step "T+5 train checkpoint" "train_models.py --tb-horizon 5"
Step "T+5 backtest + save" "run_backtest.py $btArgs"

Add-Content $log "`nALL RETRAINS COMPLETE : $(Get-Date -Format o)"
Write-Host "ALL RETRAINS COMPLETE"
