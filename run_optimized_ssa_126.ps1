param(
    [ValidateSet('Prepare','Optimize','Run','Check','Status')][string]$Stage = 'Optimize',
    [string]$Root = '.\src\output\nine_groups_old_b_ssa_optimized_126',
    [string]$SourceRoot = '.\src\output\nine_groups_vague_top1_batch_126',
    [int]$AttackSteps = 300,
    [int]$SsaSamples = 5,
    [ValidateSet('full','instruction','semantic')][string]$IndexText = 'full',
    [double]$KeyImageWeight = 0.9,
    [double]$QueryImageWeight = 0.9,
    [string]$SsaOptions = (Join-Path $PSScriptRoot 'src/configs/ssa_fast.json'),
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$Port = 3007
)

# 独立目录的新实验；默认止于图片与离线检索，不启动购物回合。
$parameters = @{
    Stage = $Stage; Root = $Root; SourceRoot = $SourceRoot
    Method = 'ssa'; AttackSteps = $AttackSteps; SsaSamples = $SsaSamples
    SsaOptions = $SsaOptions; IndexText = $IndexText
    KeyImageWeight = $KeyImageWeight; QueryImageWeight = $QueryImageWeight
    Device = $Device; Port = $Port
}
& (Join-Path $PSScriptRoot 'run_old_b_frequency_126.ps1') @parameters
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
