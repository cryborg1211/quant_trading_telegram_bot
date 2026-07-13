<#
.SYNOPSIS
    Windows auto-restart supervisor for the Quant V6 Telegram bot.

.DESCRIPTION
    The Windows equivalent of the Linux systemd unit deploy/quant-v6-bot.service
    (`Restart=on-failure`). `python run_bot.py` blocks forever on Telegram
    polling, but a transient network/DNS blip (httpx.ConnectError: getaddrinfo
    failed) can raise an unhandled NetworkError, at which point run_polling
    returns and the process simply exits — with nothing to bring it back.

    This script relaunches the bot whenever it exits, with a short backoff so a
    hard crash-loop (e.g. bad token) can't spin the CPU. Run this INSTEAD of
    `python run_bot.py`.

.PARAMETER Python
    Interpreter to use. Defaults to the conda `stock` env's python by ABSOLUTE
    path when it exists (a bare "python" resolved through a child shell's PATH
    has bitten before — 13-07-26: profile put package-less Python 3.11 first →
    ModuleNotFoundError: telegram). Falls back to "python" only when the conda
    path is absent (other machines). Override explicitly if needed:
        -Python C:\path\to\other\python.exe

.PARAMETER BackoffSeconds
    Seconds to wait between a bot exit and the next relaunch. Default 10.

.EXAMPLE
    conda activate stock
    powershell -ExecutionPolicy Bypass -File run_bot_forever.ps1

.NOTES
    Stop everything with Ctrl+C in this window (stops the supervisor and the
    bot together). Only ONE copy of the bot may poll the token at a time — do
    not run this while another `run_bot.py` is already live, or Telegram
    returns a getUpdates Conflict.
#>
param(
    [string]$Python = "",
    [int]$BackoffSeconds = 10
)

if (-not $Python) {
    $condaPython = "C:\Users\caokh\anaconda3\envs\stock\python.exe"
    if (Test-Path $condaPython) { $Python = $condaPython } else { $Python = "python" }
}

$ErrorActionPreference = "Stop"
$Script = Join-Path $PSScriptRoot "run_bot.py"

if (-not (Test-Path $Script)) {
    Write-Host "[supervisor] cannot find run_bot.py next to this script ($Script)." -ForegroundColor Red
    exit 1
}

Write-Host "[supervisor] watchdog started. Ctrl+C to stop the bot and the supervisor." -ForegroundColor Cyan
Write-Host "[supervisor] interpreter: $Python | script: $Script | backoff: ${BackoffSeconds}s" -ForegroundColor Cyan

$restarts = 0
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($restarts -eq 0) {
        Write-Host "[supervisor $ts] launching bot..." -ForegroundColor Green
    } else {
        Write-Host "[supervisor $ts] launching bot (restart #$restarts)..." -ForegroundColor Green
    }

    & $Python $Script
    $code = $LASTEXITCODE

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[supervisor $ts] bot exited (exit code=$code). Relaunching in ${BackoffSeconds}s -- press Ctrl+C now to stop." -ForegroundColor Yellow
    $restarts++
    Start-Sleep -Seconds $BackoffSeconds
}
