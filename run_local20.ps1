param(
    [ValidateSet('Serve', 'Run', 'Evaluate', 'Status')][string]$Stage = 'Status',
    [string]$RunName = 'm3_trial01',
    [string]$LlmConfig = (Join-Path $PSScriptRoot 'src\configs\llm.local.json'),
    [int]$AttackSteps = 1000,
    [ValidateSet('legacy_pixel','fair_pixel','frequency','frequency_momentum')][string]$Method = 'legacy_pixel',
    [Nullable[double]]$FrequencyAxisRatio = $null,
    [Nullable[double]]$FrequencyMomentum = $null,
    [int]$NumSteps = 20,
    [ValidateRange(1,20)][int]$Trials = 20,
    [ValidateSet('full','fragment')][string]$MemoryMode = 'full',
    [switch]$IncludeTextBaseline,
    [switch]$TextBaselineOnly,
    [ValidateRange(1,100)][int]$TextTopK = 3,
    [string]$SourceRun = 'm3_1000steps_trial01',
    [string]$Python = 'D:\Anaconda\envs\minja-vision\python.exe'
)
$ErrorActionPreference = 'Stop'
if ($Method -ne 'frequency_momentum' -and ($null -ne $FrequencyAxisRatio -or $null -ne $FrequencyMomentum)) {
    throw 'FrequencyAxisRatio and FrequencyMomentum require Method frequency_momentum.'
}
if ($TextBaselineOnly -and $Stage -ne 'Evaluate') { throw 'TextBaselineOnly requires Stage Evaluate.' }
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:NO_PROXY = 'localhost,127.0.0.1,' + $env:NO_PROXY
$bundle = Join-Path $PSScriptRoot 'src\output\local20'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'setup_report.json'))) {
    throw 'Local data setup is incomplete.'
}
Set-Location -LiteralPath $PSScriptRoot
$textBaselineArgs = @('--text-top-k', "$TextTopK")
$frequencyArgs = @()
if ($Method -ne 'legacy_pixel') { $frequencyArgs += @('--method', $Method) }
if ($null -ne $FrequencyAxisRatio) { $frequencyArgs += @('--frequency-axis-ratio', "$FrequencyAxisRatio") }
if ($null -ne $FrequencyMomentum) { $frequencyArgs += @('--frequency-momentum', "$FrequencyMomentum") }
if ($IncludeTextBaseline) { $textBaselineArgs += '--include-text-baseline' }
if ($TextBaselineOnly) { $textBaselineArgs += '--text-baseline-only' }
if ($Stage -eq 'Serve') {
    & $Python -u -m src.environment.local_webshop --root $bundle
} elseif ($Stage -eq 'Run') {
    & $Python -u -m src.experiments.local_experiment run --root $bundle --run-name $RunName `
        --llm-config $LlmConfig --attack-steps $AttackSteps --num-steps $NumSteps `
        --trials $Trials --memory-mode $MemoryMode @textBaselineArgs @frequencyArgs
} elseif ($Stage -eq 'Evaluate') {
    & $Python -u -m src.experiments.local_experiment evaluate --root $bundle --run-name $RunName `
        --llm-config $LlmConfig --attack-steps $AttackSteps --num-steps $NumSteps `
        --source-run $SourceRun --trials $Trials --memory-mode $MemoryMode @textBaselineArgs @frequencyArgs
} else {
    Get-Content -LiteralPath (Join-Path $bundle 'setup_report.json') -Encoding utf8
    exit 0
}
if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed with exit code $LASTEXITCODE" }
