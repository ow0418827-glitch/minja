param(
    [ValidateSet('Prepare','Optimize','Run','Evaluate','Resume','Check','Status')]
    [string]$Stage = 'Run',
    [ValidateRange(1,65535)][int]$Port = 3007,
    [ValidateSet('frequency_momentum','ssa')][string]$Method = 'frequency_momentum',
    [ValidateRange(1,2147483647)][int]$AttackSteps = 500,
    [string]$Root,
    [string]$SourceRoot,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [Nullable[double]]$FrequencyAxisRatio,
    [Nullable[double]]$FrequencyMomentum,
    [Nullable[int]]$SsaSamples,
    [Nullable[double]]$SsaRho,
    [Nullable[double]]$SsaSigma,
    [Nullable[double]]$SsaMomentum,
    [string]$SsaOptions,
    [ValidateSet('full','instruction','semantic')][string]$IndexText = 'full',
    [ValidateRange(0,1)][double]$KeyImageWeight = 0.5,
    [ValidateRange(0,1)][double]$QueryImageWeight = 0.5,
    [ValidateSet('all','adversarial','clean','no_memory')][string]$Condition
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pipelinePython = 'D:\Anaconda\envs\minja-vision\python.exe'
if (-not (Test-Path -LiteralPath $pipelinePython -PathType Leaf)) { throw 'The minja Python environment is missing.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    $pipelineArguments = @('-X', 'utf8', '-u', '-m', 'src.experiments.reused_frequency_pipeline',
        $Stage.ToLower(), '--port', $Port, '--method', $Method, '--attack-steps', $AttackSteps,
        '--device', $Device)
    if ($Root) { $pipelineArguments += @('--root', $Root) }
    if ($SourceRoot) { $pipelineArguments += @('--source-root', $SourceRoot) }
    if ($SsaOptions) { $pipelineArguments += @('--ssa-options', $SsaOptions) }
    if ($Condition) { $pipelineArguments += @('--condition', $Condition) }
    $pipelineArguments += @('--index-text', $IndexText, '--key-image-weight', "$KeyImageWeight", '--query-image-weight', "$QueryImageWeight")
    foreach ($parameter in @(
        @('FrequencyAxisRatio', '--frequency-axis-ratio', $FrequencyAxisRatio),
        @('FrequencyMomentum', '--frequency-momentum', $FrequencyMomentum),
        @('SsaSamples', '--ssa-samples', $SsaSamples),
        @('SsaRho', '--ssa-rho', $SsaRho),
        @('SsaSigma', '--ssa-sigma', $SsaSigma),
        @('SsaMomentum', '--ssa-momentum', $SsaMomentum))) {
        if ($null -ne $parameter[2]) {
            $pipelineArguments += @($parameter[1], [string]::Format(
                [Globalization.CultureInfo]::InvariantCulture, '{0}', $parameter[2]))
        }
    }
    & $pipelinePython @pipelineArguments
    if ($LASTEXITCODE -ne 0) { throw "Old-B frequency pipeline stopped: $LASTEXITCODE" }
} finally {
    Pop-Location
}
