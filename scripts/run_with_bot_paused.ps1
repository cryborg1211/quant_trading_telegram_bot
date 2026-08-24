<#
.SYNOPSIS
    Pause the Telegram bot, run a DuckDB-writing job, always restart the bot.

.DESCRIPTION
    WHY THIS EXISTS (24-08-26)
    --------------------------
    src/data/db_engine.py's DuckDBEngine is a process-wide singleton that opens
    the core DB READ-WRITE and holds that connection for the life of the process
    (its own docstring: "DO NOT pass read_only= or config= here"). DuckDB permits
    exactly ONE writing process, so while run_bot.py is up, any other process
    opening data/quant_v6_core.duckdb read-write dies with:

        IOException: Cannot open file ... used by another process.
        File is already open in ...python.exe (PID 19940)

    That killed the 24-08 15:30 cron AFTER the crawl and the PAID sentiment stage
    had already run - the operator got a crash alert instead of signals. It was
    not a one-off: the bot holds the lock for its whole lifetime, so every EOD run
    collides with a running bot. run_backtest/train_models reach the same DB
    (src/backtest/pipeline.py core_duckdb), so the weekly retrain is exposed too.

    This wrapper serialises them: stop the bot, WAIT for the lock to actually
    clear, run the job, and restart the bot in a finally block so a crashing job
    can never leave the bot down.

.PARAMETER Command
    Arguments passed to python, e.g. "main.py --task full_pipeline".

.PARAMETER Script
    A PowerShell script to run instead of a python command - the weekly retrain
    is a .ps1 (scripts\retrain_all.ps1), and it reaches the same core DuckDB via
    src/backtest/pipeline.py, so it needs the same serialisation.

.PARAMETER Label
    Short tag used in the log filename.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_with_bot_paused.ps1 -Command "main.py --task full_pipeline" -Label eod

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_with_bot_paused.ps1 -Script scripts\retrain_all.ps1 -Label retrain
#>
param(
    [string]$Command,
    [string]$Script,
    [string]$Label = "job"
)

if (-not $Command -and -not $Script) { throw "Pass either -Command (python args) or -Script (a .ps1)." }
if ($Command -and $Script) { throw "Pass -Command OR -Script, not both." }

$ErrorActionPreference = "Continue"
$repo = "C:\Users\caokh\Desktop\vscode\stock_price_v3"
$py   = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
Set-Location $repo
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $repo

$log = Join-Path $repo ("logs\{0}_{1}.log" -f $Label, (Get-Date -Format 'yyyyMMdd_HHmmss'))
function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

# Match on the COMMAND LINE, never the image name: several unrelated python
# processes (MCP servers, a running retrain) share python.exe and must survive.
function Get-BotProcs {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*run_bot.py*' }
}

# Process exit and file-handle release are not simultaneous. Confirm the lock is
# genuinely gone rather than assuming, or the job hits the very error this
# wrapper exists to prevent.
function Wait-DbFree([int]$TimeoutSec = 60) {
    $probe = Join-Path $env:TEMP "db_lock_probe.py"
    $src = @(
        'import sys, duckdb'
        'try:'
        '    duckdb.connect(sys.argv[1]).close()'
        'except Exception as exc:'
        '    print(exc); sys.exit(1)'
        'sys.exit(0)'
    ) -join "`n"
    Set-Content -Path $probe -Value $src -Encoding utf8
    $dbPath = Join-Path $repo "data\quant_v6_core.duckdb"
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        & $py $probe $dbPath | Out-Null
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

$botWasRunning = $false
$exitCode = 1
try {
    $bots = @(Get-BotProcs)
    if ($bots.Count -gt 0) {
        $botWasRunning = $true
        Say ("Bot running (PID {0}) - stopping so it releases the DuckDB write lock." -f ($bots.ProcessId -join ', '))
        foreach ($b in $bots) { Stop-Process -Id $b.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 2
        if (@(Get-BotProcs).Count -gt 0) { Say "WARNING: a run_bot.py process survived Stop-Process." }
    } else {
        Say "No bot running - nothing to pause."
    }

    if (-not (Wait-DbFree 60)) {
        Say "ABORT: core DuckDB still locked after 60s. NOT running the job - it would crash mid-way and waste the paid sentiment stage."
        $exitCode = 2
    } else {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        if ($Script) {
            Say ("DB lock free. Running script: {0}" -f $Script)
            cmd /c "powershell -ExecutionPolicy Bypass -File `"$Script`" >> `"$log`" 2>&1"
        } else {
            Say ("DB lock free. Running: python {0}" -f $Command)
            cmd /c "`"$py`" $Command >> `"$log`" 2>&1"
        }
        $exitCode = $LASTEXITCODE
        $sw.Stop()
        Say ("Job finished in {0:N0}s with exit code {1}." -f $sw.Elapsed.TotalSeconds, $exitCode)
    }
}
finally {
    # ALWAYS restore the bot, including after a crashing job - otherwise one bad
    # EOD run silently leaves the operator without a bot until they notice.
    if ($botWasRunning) {
        if (@(Get-BotProcs).Count -eq 0) {
            # CAPTURE THE BOT'S OUTPUT (24-08-26). The first version started it
            # bare, so when a restarted bot later died it left NO trace and the
            # cause was unrecoverable - exactly what happened to PID 8128. The
            # bot logs to stderr, so without this the post-mortem is guesswork.
            $stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
            $botOut  = Join-Path $repo ("logs\bot_{0}.log" -f $stamp)
            $botErr  = Join-Path $repo ("logs\bot_{0}.err" -f $stamp)
            Start-Process -FilePath $py -ArgumentList "run_bot.py" `
                -WorkingDirectory $repo -WindowStyle Hidden `
                -RedirectStandardOutput $botOut -RedirectStandardError $botErr | Out-Null
            Say ("Bot output -> {0} (+ .err)" -f $botOut)
            Start-Sleep -Seconds 5
            $now = @(Get-BotProcs)
            if ($now.Count -gt 0) { Say ("Bot restarted (PID {0})." -f ($now.ProcessId -join ', ')) }
            else { Say "ERROR: bot restart FAILED - start it manually." }
        } else {
            Say "Bot already running again - no restart needed."
        }
    } else {
        Say "Bot was not running before; leaving it off."
    }
    Say ("Log: {0}" -f $log)
}

exit $exitCode
