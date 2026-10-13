param(
    [ValidateSet('BuildCatalog','Prepare','Check','Collect','Attack','Finalize','Serve','Train','Evaluate','Run','Status')]
    [string]$Stage = 'Status',
    [string]$Root = (Join-Path $PSScriptRoot 'src\output\nine_groups_attacker_history_joint'),
    [string]$Catalog = (Join-Path $PSScriptRoot 'src\output\nine_group_catalog_full'),
    [string]$WebshopSource = (Join-Path (Split-Path $PSScriptRoot -Parent) 'MINJA-main\rap\webshop'),
    [string]$Targets = (Join-Path (Split-Path $PSScriptRoot -Parent) 'MINJA-main\rap\output\paper3gb\catalog\targets.json'),
    [string]$LlmConfig = (Join-Path $PSScriptRoot 'src\configs\llm.local.json'),
    [Alias('indication_prompt')]
    [string]$IndicationPrompt = (Join-Path $PSScriptRoot 'src\data\indication_prompt_template_neutral.json'),
    [ValidateRange(1,100)][int]$IndicationLevel = 1,
    [ValidateRange(1,100000)][int]$AttackSteps = 1000,
    [ValidateSet('legacy_pixel','fair_pixel','frequency','frequency_momentum','ssa')][string]$Method = 'legacy_pixel',
    [Nullable[double]]$FrequencyAxisRatio = $null,
    [Nullable[double]]$FrequencyMomentum = $null,
    [Nullable[int]]$SsaSamples = $null,
    [Nullable[double]]$SsaRho = $null,
    [Nullable[double]]$SsaSigma = $null,
    [Nullable[double]]$SsaMomentum = $null,
    [string]$SsaOptions,
    [ValidateRange(1,255)][int]$EpsilonLevels = 16,
    [string]$OptimizationConfig,
    [string]$ReferenceOverrides,
    [ValidateRange(1,200)][int]$NumSteps = 20,
    [ValidateRange(1,20)][int]$NormalRetries = 2,
    [ValidateRange(0,5)][int]$HistoryReserveCount = 0,
    [ValidateRange(1,50)][int]$HistoryMaxAttempts = 3,
    [ValidateSet('observed','planned')][string]$HistoryProductMode = 'observed',
    [ValidateSet('attacker','victim')][string]$SourceHistoryMode = 'attacker',
    [ValidateRange(1,15)][int]$MemoryTopK = 1,
    [ValidateSet('full','fragment')][string]$MemoryMode = 'full',
    [ValidateRange(12800,2000000)][int]$MemoryPromptBudget = 128000,
    [ValidateRange(1,131072)][int]$PolicyMaxTokens = 2048,
    [ValidateSet('clip','gme')][string]$RetrievalEncoder = 'clip',
    [string]$GmePython = 'D:\Anaconda\envs\dual-visual\python.exe',
    [string]$GmeModel = 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
    [ValidateSet('joint','image')][string]$MemoryKeyMode = 'joint',
    [ValidateSet('joint','image')][string]$MemoryQueryMode = 'joint',
    [ValidateSet('full','instruction','semantic')][string]$IndexText = 'full',
    [ValidateRange(0,1)][double]$KeyImageWeight = 0.5,
    [ValidateRange(0,1)][double]$QueryImageWeight = 0.5,
    [ValidateSet('targeted','heldout')][string]$TestMode = 'targeted',
    [ValidateSet('vague','strict')][string]$QuerySpecificity = 'vague',
    [ValidateRange(0,5)][int]$ReserveTests = 5,
    [ValidateSet('reference','task')][string]$QueryImageMode = 'reference',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [ValidateRange(1,65535)][int]$Port = 3004,
    [string[]]$Group,
    [switch]$IncludeControls,
    [switch]$PolicyQueryImage = $true,
    [switch]$NoPolicyQueryImage,
    [switch]$ModelCheck,
    [switch]$RetryErrors,
    [switch]$AllowCodeUpdateResume,
    [switch]$AllowBoundHistoryRepairResume,
    [switch]$ReuseFrozenMemory,
    [switch]$DirectImages,
    [string]$Python = 'D:\Anaconda\envs\minja-vision\python.exe'
)
$ErrorActionPreference = 'Stop'
if ($Method -ne 'frequency_momentum' -and ($null -ne $FrequencyAxisRatio -or $null -ne $FrequencyMomentum)) {
    throw 'FrequencyAxisRatio and FrequencyMomentum require Method frequency_momentum.'
}
if ($Method -ne 'ssa' -and ($null -ne $SsaSamples -or $null -ne $SsaRho -or $null -ne $SsaSigma -or $null -ne $SsaMomentum -or $SsaOptions)) {
    throw 'SsaSamples, SsaRho, SsaSigma and SsaMomentum require Method ssa.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:NO_PROXY = 'localhost,127.0.0.1,' + $env:NO_PROXY
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Stage -eq 'BuildCatalog') {
        if (Test-Path -LiteralPath (Join-Path $Catalog 'data_manifest.json')) {
            Write-Output 'The disk catalog already exists. Choose a new Catalog to build another one.'
        } else {
            & $Python -X utf8 -u -m src.environment.full_webshop build `
                --raw-items (Join-Path $WebshopSource 'data\items_shuffle.json') `
                --raw-attributes (Join-Path $WebshopSource 'data\items_ins_v2.json') `
                --human-goals (Join-Path $WebshopSource 'data\items_human_ins.json') `
                --destination $Catalog --webshop-source $WebshopSource --targets $Targets --search-backend lucene
            if ($LASTEXITCODE -ne 0) { throw "Catalog build failed: $LASTEXITCODE" }
        }
    } else {
        $experimentArgs = @('-X','utf8','-u','-m','src.experiments.nine_group_experiment', $Stage.ToLower(),
            '--root',$Root,'--catalog',$Catalog,'--webshop-source',$WebshopSource,
            '--llm-config',$LlmConfig,'--indication-prompt',$IndicationPrompt,'--indication-level',"$IndicationLevel",
            '--attack-steps',"$AttackSteps",'--method',$Method,'--epsilon-levels',"$EpsilonLevels",
            '--num-steps',"$NumSteps",'--memory-top-k',"$MemoryTopK",'--memory-mode',$MemoryMode,
            '--normal-retries',"$NormalRetries",'--history-max-attempts',"$HistoryMaxAttempts",
            '--history-reserve-count',"$HistoryReserveCount",
            '--history-product-mode',$HistoryProductMode,
            '--source-history-mode',$SourceHistoryMode,
            '--memory-prompt-budget',"$MemoryPromptBudget",'--query-image-mode',$QueryImageMode,
            '--policy-max-tokens',"$PolicyMaxTokens",
            '--retrieval-encoder',$RetrievalEncoder,'--gme-python',$GmePython,'--gme-model',$GmeModel,
            '--memory-key-mode',$MemoryKeyMode,'--memory-query-mode',$MemoryQueryMode,
            '--index-text',$IndexText,'--test-mode',$TestMode,'--query-specificity',$QuerySpecificity,
            '--reserve-tests',"$ReserveTests",
            '--device',$Device,'--port',"$Port")
        foreach ($item in $Group) { $experimentArgs += @('--group', $item) }
        if ($OptimizationConfig) { $experimentArgs += @('--optimization-config', $OptimizationConfig) }
        if ($ReferenceOverrides) { $experimentArgs += @('--reference-overrides', $ReferenceOverrides) }
        if ($null -ne $FrequencyAxisRatio) { $experimentArgs += @('--frequency-axis-ratio', "$FrequencyAxisRatio") }
        if ($null -ne $FrequencyMomentum) { $experimentArgs += @('--frequency-momentum', "$FrequencyMomentum") }
        if ($null -ne $SsaSamples) { $experimentArgs += @('--ssa-samples', "$SsaSamples") }
        if ($null -ne $SsaRho) { $experimentArgs += @('--ssa-rho', "$SsaRho") }
        if ($null -ne $SsaSigma) { $experimentArgs += @('--ssa-sigma', "$SsaSigma") }
        if ($null -ne $SsaMomentum) { $experimentArgs += @('--ssa-momentum', "$SsaMomentum") }
        if ($SsaOptions) { $experimentArgs += @('--ssa-options', $SsaOptions) }
        $experimentArgs += @('--key-image-weight', "$KeyImageWeight", '--query-image-weight', "$QueryImageWeight")
        if ($IncludeControls) { $experimentArgs += '--include-controls' }
        if ($NoPolicyQueryImage -or -not $PolicyQueryImage) {
            $experimentArgs += '--no-policy-query-image'
        } else {
            $experimentArgs += '--policy-query-image'
        }
        if ($ModelCheck) { $experimentArgs += '--model-check' }
        if ($RetryErrors) { $experimentArgs += '--retry-errors' }
        if ($AllowCodeUpdateResume) { $experimentArgs += '--allow-code-update-resume' }
        if ($AllowBoundHistoryRepairResume) { $experimentArgs += '--allow-bound-history-repair-resume' }
        if ($ReuseFrozenMemory) { $experimentArgs += '--reuse-frozen-memory' }
        if ($DirectImages) { $experimentArgs += '--direct-images' }
        & $Python @experimentArgs
        if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed: $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
