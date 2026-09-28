# prod_headroom_check.ps1 -- step 049 production headroom health check.
#
# Why: the 046 production config runs with only a few hundred MiB of DISCRETE-GPU
# headroom. Step 049 showed that an external squatter of just 64 MiB on the
# DISCRETE GPU pushes decode from ~121 to ~85 tok/s, 256 MiB to ~73, 512 MiB to
# ~20 -- and the evicted weight pages go to the SHARED (system-RAM) segment and
# never come back until the process restarts. GPU clocks stay at full boost
# throughout, so this is a placement cliff, not throttling.
#
# ADAPTER ACCOUNTING (corrected 2026-09-29, user input): the desktop's
# "dedicated" usage (DWM/msedge/explorer/Bettbox, ~530 MiB) belongs to the
# INTEGRATED GPU adapter, not the discrete one. \GPU Adapter Memory(*) has one
# instance per adapter, so summing them and subtracting from the discrete card's
# capacity is WRONG (it under-reports headroom by the iGPU's share). This script
# therefore reports the per-adapter split and derives headroom from nvidia-smi
# (discrete GPU only). On the discrete adapter only the engine is resident.
#
# The discriminating signal is the PER-PROCESS split, not nvidia-smi's
# memory.used (adapter commit stays flat either way):
#   engine_shared - (offload region + ~106 MiB) = weight pages pushed to system RAM
# Step 049 calibration (8k steady, warmup 0):
#   +0 MiB -> 121-127 tok/s | +102 -> 111 | +278 -> 92 | +558 -> 56
#
# exit 0 = healthy (evicted <= WarnMib)
# exit 2 = WARN   (WarnMib < evicted <= DegradedMib)
# exit 3 = DEGRADED (evicted > DegradedMib) -- expect a large decode loss;
#          restarting the engine only helps AFTER the external squatter is gone.
# exit 4 = counter/service unavailable
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\prod_headroom_check.ps1
#   ... -OffloadMib 8192 -WarnMib 100 -DegradedMib 250
param(
    [int]$OffloadMib    = 8192,
    [int]$WarnMib       = 100,
    [int]$DegradedMib   = 250,
    [int]$BaselineExtraMib = 106
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

function Get-MB([double]$b) { return [math]::Round($b / 1MB, 1) }

$adD = (Get-Counter -Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$pD  = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$pS  = (Get-Counter -Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
if (-not $adD -or -not $pD -or -not $pS) { Write-Host "counter error: GPU counters unavailable"; exit 4 }

# engine = the python process holding the most DEDICATED GPU memory
$eng = $null
foreach ($s in ($pD | Sort-Object CookedValue -Descending)) {
    if ($s.InstanceName -match '^pid_(\d+)') {
        $p = Get-Process -Id $Matches[1] -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -like "python*") { $eng = $s; break }
    }
}
if (-not $eng) { Write-Host "engine python process not found (service down?)"; exit 4 }

$engPid = [int]($eng.InstanceName -replace '^pid_(\d+)_.*$', '$1')
$engLuid = if ($eng.InstanceName -match '(luid_0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)') { $Matches[1] } else { "" }
$engDed = Get-MB $eng.CookedValue
$engShr = 0.0
foreach ($s in $pS) { if ($s.InstanceName -match "^pid_${engPid}_") { $engShr = Get-MB $s.CookedValue } }

# discrete adapter = the one the engine lives on
$dGpuDed = 0.0
$otherDed = 0.0
foreach ($a in $adD) {
    if ($engLuid -and $a.InstanceName -like "*${engLuid}*") { $dGpuDed = Get-MB $a.CookedValue }
    else { $otherDed += (Get-MB $a.CookedValue) }
}

$nv = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
$nvUsed = $null; $nvTotal = $null
if ($nv) {
    $parts = $nv.Split(",") | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 2) { [void][int]::TryParse($parts[0], [ref]$nvUsed); [void][int]::TryParse($parts[1], [ref]$nvTotal) }
}

$evicted = [math]::Round($engShr - $OffloadMib - $BaselineExtraMib, 1)
$verdict = "OK"; $code = 0
if ($evicted -gt $DegradedMib) { $verdict = "DEGRADED"; $code = 3 }
elseif ($evicted -gt $WarnMib)  { $verdict = "WARN";     $code = 2 }

Write-Host ("engine pid={0} on adapter {1}: dedicated={2} MiB shared={3} MiB" -f $engPid, $engLuid, $engDed, $engShr)
Write-Host ("adapters: discrete(engine)={0} MiB   other adapters (iGPU/desktop)={1} MiB" -f $dGpuDed, [math]::Round($otherDed, 1))
if ($nvTotal) {
    Write-Host ("nvidia-smi (discrete only): used={0} total={1} -> headroom {2} MiB" -f $nvUsed, $nvTotal, ($nvTotal - $nvUsed))
}
Write-Host ("evicted weight pages (engine shared - offload {0} - {1}) = {2} MiB" -f $OffloadMib, $BaselineExtraMib, $evicted)
Write-Host ("verdict = {0}  (step-049 calibration: 0->121-127, 102->111, 278->92, 558->56 tok/s)" -f $verdict)
if ($verdict -ne "OK") {
    Write-Host "NOTE: the usual cause is a NEW allocation on the DISCRETE GPU (another CUDA"
    Write-Host "      process/game/tool) -- iGPU desktop usage does NOT count."
    Write-Host "      Restarting the engine helps only after that squatter has released its VRAM."
}
exit $code
