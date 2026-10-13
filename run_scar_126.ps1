param(
    [ValidateSet('Prepare','Run','Evaluate','Resume','Check','Status')][string]$Stage = 'Evaluate',
    [ValidateSet('all','adversarial','clean','no_memory')][string]$Condition = 'adversarial',
    [string]$Root,
    [string]$SourceRoot,
    [string]$SelectionReport,
    [string]$ReuseResultsManifest,
    [ValidateRange(1,65535)][int]$Port = 3007,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$scarPython = 'D:\Anaconda\envs\minja-vision\python.exe'
if (-not (Test-Path -LiteralPath $scarPython -PathType Leaf)) { throw 'The minja Python environment is missing.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    $scarArguments = @('-X','utf8','-u','-B','-m','src.experiments.reused_scar_pipeline',
        '--stage',$Stage.ToLower(),'--condition',$Condition,'--port',"$Port",'--device',$Device)
    foreach ($scarOption in @(
        @('--root',$Root), @('--source-root',$SourceRoot),
        @('--selection-report',$SelectionReport), @('--reuse-results-manifest',$ReuseResultsManifest))) {
        if ($scarOption[1]) { $scarArguments += $scarOption }
    }
    & $scarPython @scarArguments
    if ($LASTEXITCODE -ne 0) { throw "SCAR pipeline stopped: $LASTEXITCODE" }
} finally {
    Pop-Location
}
