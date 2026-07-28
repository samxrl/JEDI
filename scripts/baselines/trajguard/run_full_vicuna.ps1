# Fully build and evaluate TrajGuard (Vicuna-7B) with the project virtual environment.
# Each run creates a separate log that preserves complete terminal output for model loading, construction, and evaluation.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$python = 'R:\SARC\venv\Scripts\python.exe'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDirectory = Join-Path $projectRoot 'scripts\baselines\trajguard\logs'
$logPath = Join-Path $logDirectory "trajguard_vicuna_7b_v1_5_full_$timestamp.log"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

function Invoke-LoggedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    "`n=== $(Get-Date -Format o) ===" | Tee-Object -FilePath $logPath -Append
    "& '$python' $($Arguments -join ' ')" | Tee-Object -FilePath $logPath -Append
    # Python logging writes to stderr by default; append it to the log to prevent PowerShell from misreporting it.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @Arguments *>> $logPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code: $exitCode"
    }
}

"TrajGuard Vicuna-7B full run log: $logPath" | Tee-Object -FilePath $logPath
try {
    Invoke-LoggedPython @(
        'scripts/baselines/trajguard/build_artifacts.py',
        '--config', 'scripts/baselines/trajguard/artifact_config.yaml'
    )
    Invoke-LoggedPython @(
        'scripts/baselines/trajguard/run_evaluation.py',
        '--config', 'scripts/baselines/trajguard/evaluation_config.yaml'
    )
    "`n=== $(Get-Date -Format o) COMPLETE ===" | Tee-Object -FilePath $logPath -Append
    Write-Host "Completed. Full log: $logPath"
}
catch {
    "`n=== $(Get-Date -Format o) FAILED ===" | Tee-Object -FilePath $logPath -Append
    ($_ | Out-String).TrimEnd() | Tee-Object -FilePath $logPath -Append
    exit 1
}
