param(
    [ValidateSet('Prepare','Check','Run','Status')]
    [string]$Stage = 'Run',
    [switch]$ModelCheck,
    [switch]$AllowCodeUpdateResume,
    [switch]$AllowBoundHistoryRepairResume
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
if ($ModelCheck -and $Stage -ne 'Check') { throw 'ModelCheck requires stage Check.' }
$parameters = @{
    Stage = $Stage
    Root = (Join-Path $PSScriptRoot 'src\output\nine_groups_identity_frequency_126')
    ReferenceOverrides = (Join-Path $PSScriptRoot 'src\data\reference_overrides_identity_126.json')
    Group = @('1', '2', '6')
    Method = 'frequency_momentum'
    AttackSteps = 1000
    FrequencyAxisRatio = 0.25
    FrequencyMomentum = 1.0
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
if ($ModelCheck) { $parameters.ModelCheck = $true }
if ($AllowCodeUpdateResume) { $parameters.AllowCodeUpdateResume = $true }
if ($AllowBoundHistoryRepairResume) { $parameters.AllowBoundHistoryRepairResume = $true }
Write-Output ("Starting stage {0}, groups 1/2/6, frequency momentum, 1000 steps." -f $Stage)
& (Join-Path $PSScriptRoot 'run_nine_groups.ps1') @parameters
