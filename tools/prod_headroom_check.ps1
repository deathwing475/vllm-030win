# prod_headroom_check.ps1 -- step 049 production headroom health check.
#
# Why: the 046 production config runs with only ~356 MiB of adapter dedicated
# VRAM left over (15,946 / 16,303 MiB). Step 049 showed that an external
# squatter of just 64 MiB pushes decode from ~121 to ~85 tok/s, 256 MiB to ~73,
# 512 MiB to ~20 -- and the evicted weight pages go to the SHARED (system-RAM)
# segment and never come back until the process restarts. GPU clocks stay at
# full boost throughout, so this is a placement cliff, not throttling.
#
# The discriminating signal is the PER-PROCESS split, not nvidia-smi's
# memory.used (adapter commit stays flat at ~15.9 GiB either way):
#   engine_dedicated + engine_shared == 23,796 MiB (constant)
#   engine_shared - (offload region + ~106 MiB) = weight pages pushed to system RAM
# Step 049 calibration (8k steady, warmup 0):
#   +0 MiB -> 121-127 tok/s | +102 -> 111 | +278 -> 92 | +558 -> 56
#
# exit 0 = healthy (evicted <= WarnMib)
# exit 2 = WARN   (WarnMib < evicted <= DegradedMib)
# exit 3 = DEGRADED (evicted > DegradedMib) -- expect a large decode loss;
#          restarting the engine only helps AFTER the external squatter is gone.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\prod_headroom_check.ps1
#   ... -OffloadMib 8192 -WarnMib 100 -DegradedMib 250
param(
    [int]$OffloadMib    = 8192,
    [int]$WarnMib       = 100,
    [int]$DegradedMib   = 250,
    [int]$CardTotalMib  = 16303,
    [int]$BaselineExtraMib = 106
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

function Get-MB([double]$b) { return [math]::Round($b / 1MB, 1) }

$ded = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$shr = (Get-Counter -Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
if (-not $ded -or -not $shr) { Write-Host "counter error: GPU Process Memory unavailable"; exit 4 }

$adD = (Get-Counter -Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples |
        Measure-Object CookedValue -Sum

# engine = the process holding the most DEDICATED GPU memory whose image is python
$eng = $null
foreach ($s in ($ded | Sort-Object CookedValue -Descending)) {
    if ($s.InstanceName -match '^pid_(\d+)') {
        $p = Get-Process -Id $Matches[1] -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -like "python*") { $eng = $s; break }
    }
}
if (-not $eng) { Write-Host "engine python process not found (service down?)"; exit 4 }

$engPid = [int]($eng.InstanceName -replace '^pid_(\d+)_.*$', '$1')
$engDed = Get-MB $eng.CookedValue
$engShr = 0.0
foreach ($s in $shr) { if ($s.InstanceName -match "^pid_${engPid}_") { $engShr = Get-MB $s.CookedValue } }

$evicted = [math]::Round($engShr - $OffloadMib - $BaselineExtraMib, 1)
$adapterDed = Get-MB $adD.Sum
$headroom = [math]::Round($CardTotalMib - $adapterDed, 1)

$verdict = "OK"
$code = 0
if ($evicted -gt $DegradedMib) { $verdict = "DEGRADED"; $code = 3 }
elseif ($evicted -gt $WarnMib)  { $verdict = "WARN";     $code = 2 }

Write-Host ("engine pid={0} dedicated={1} MiB shared={2} MiB" -f $engPid, $engDed, $engShr)
Write-Host ("adapter dedicated={0} MiB of {1} -> headroom {2} MiB" -f $adapterDed, $CardTotalMib, $headroom)
Write-Host ("evicted weight pages (shared - offload {0} - {1}) = {2} MiB" -f $OffloadMib, $BaselineExtraMib, $evicted)
Write-Host ("verdict = {0}  (step-049 calibration: 0->121-127, 102->111, 278->92, 558->56 tok/s)" -f $verdict)
if ($verdict -ne "OK") {
    Write-Host "NOTE: an external VRAM squatter (browser/DWM/AV/other process) is the usual cause."
    Write-Host "      Restarting the engine helps only after that squatter has released its VRAM."
}
exit $code
