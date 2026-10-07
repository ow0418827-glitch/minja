param(
    [ValidateSet('Prepare','Check','Run','Status')]
    [string]$Stage = 'Run',
    [ValidateSet('frequency_momentum','ssa')][string]$Method = 'frequency_momentum',
    [string]$Root,
    [Nullable[double]]$FrequencyAxisRatio = $null,
    [Nullable[double]]$FrequencyMomentum = $null,
    [Nullable[int]]$SsaSamples = $null,
    [Nullable[double]]$SsaRho = $null,
    [Nullable[double]]$SsaSigma = $null,
    [Nullable[double]]$SsaMomentum = $null,
    [switch]$ModelCheck,
    [switch]$AllowCodeUpdateResume,
    [switch]$AllowBoundHistoryRepairResume
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
if ($ModelCheck -and $Stage -ne 'Check') { throw 'ModelCheck requires stage Check.' }
if ($Method -ne 'frequency_momentum' -and ($null -ne $FrequencyAxisRatio -or $null -ne $FrequencyMomentum)) {
    throw 'FrequencyAxisRatio and FrequencyMomentum require Method frequency_momentum.'
}
if ($Method -ne 'ssa' -and ($null -ne $SsaSamples -or $null -ne $SsaRho -or $null -ne $SsaSigma -or $null -ne $SsaMomentum)) {
    throw 'SsaSamples, SsaRho, SsaSigma and SsaMomentum require Method ssa.'
}
if (-not $Root) {
    $rootName = if ($Method -eq 'ssa') { 'nine_groups_identity_ssa_126' } else { 'nine_groups_identity_frequency_126' }
    $Root = Join-Path $PSScriptRoot ('src\output\' + $rootName)
}
$parameters = @{
    Stage = $Stage
    Root = $Root
    ReferenceOverrides = (Join-Path $PSScriptRoot 'src\data\reference_overrides_identity_126.json')
    Group = @('1', '2', '6')
    Method = $Method
    AttackSteps = 1000
    EpsilonLevels = 16
    SourceHistoryMode = 'attacker'
    HistoryProductMode = 'observed'
    RetrievalEncoder = 'clip'
    MemoryKeyMode = 'joint'
    MemoryQueryMode = 'joint'
    MemoryMode = 'full'
    MemoryTopK = 1
    IndexText = 'full'
    TestMode = 'targeted'
    QuerySpecificity = 'vague'
    QueryImageMode = 'reference'
    PolicyQueryImage = $true
    IncludeControls = $true
    ReserveTests = 5
    NumSteps = 20
    Device = 'cuda'
    Port = 3006
}
if ($Method -eq 'frequency_momentum') {
    $parameters.FrequencyAxisRatio = if ($null -ne $FrequencyAxisRatio) { $FrequencyAxisRatio } else { 0.25 }
    $parameters.FrequencyMomentum = if ($null -ne $FrequencyMomentum) { $FrequencyMomentum } else { 1.0 }
}
if ($null -ne $SsaSamples) { $parameters.SsaSamples = $SsaSamples }
if ($null -ne $SsaRho) { $parameters.SsaRho = $SsaRho }
if ($null -ne $SsaSigma) { $parameters.SsaSigma = $SsaSigma }
if ($null -ne $SsaMomentum) { $parameters.SsaMomentum = $SsaMomentum }
if ($ModelCheck) { $parameters.ModelCheck = $true }
if ($AllowCodeUpdateResume) { $parameters.AllowCodeUpdateResume = $true }
if ($AllowBoundHistoryRepairResume) { $parameters.AllowBoundHistoryRepairResume = $true }
Write-Output ("Starting stage {0}, groups 1/2/6, method {1}, 1000 steps." -f $Stage, $Method)
& (Join-Path $PSScriptRoot 'run_nine_groups.ps1') @parameters
