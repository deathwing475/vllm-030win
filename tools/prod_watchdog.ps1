# prod_watchdog.ps1 — production launcher watchdog: boot, probe, restart if slow.
#
# Slow-boot problem (step 042, 2026-09-28): random slow boots cost ~19-28%
# decode (8k steady 83-99 vs 115-127 tok/s on the production config). The
# engine log / torch accounting show NO trace (Initial free memory 14.68 GiB
# identical fast vs slow) — the defect lives at the WDDM placement layer.
#
# Detection verdict (measured, census1 + step-040/041 boots):
#   - VRAM fingerprint is NOT reliable: slow boots were seen at 15,474 MiB,
#     fast boots at 15,486 MiB (12 MiB apart) — a shadow variable at best.
#   - A direct performance probe separates cleanly:
#       slow boots: single runs 83.45-99.39 tok/s (median <= 98.52)
#       fast boots: single runs 115.29-126.82 tok/s (median >= 117.12)
#     Verdict rule: 8k probe x3 (anchor_longctx.py, warmup 0), median
#     steady >= ProbeThreshold (default 105) => fast; else slow.
#
# Behavior: kill leftovers -> boot the production launcher -> poll /health ->
# settle -> run the probe -> below threshold: kill and retry (up to
# -MaxRestarts). On success or exhausted retries the service is LEFT RUNNING.
#   exit 0 = fast-tier boot verified
#   exit 2 = retries exhausted, last boot left running (slow tier)
#   exit 3 = boot never became healthy
#   exit 4 = probe error (needle miss / request failure) — service kept
#
# Notes:
#   - The probe writes its 8k prompt into the prefix cache; harmless for real
#     traffic (LRU evicts), verified by the step-027 soak.
#   - Re-calibrate ProbeThreshold if the production config changes.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\prod_watchdog.ps1
#   powershell -NoProfile -File tools\prod_watchdog.ps1 -MaxRestarts 3
param(
    [string]$Launcher = "G:\qwen3.8model\vllm-030win-git\tools\serve_gsq_prod029_n2.cmd",
    [string]$WorkDir  = "G:\qwen3.8model\Qwen3.8-27B-3Bit-GSQ",
    [string]$LogDir   = "G:\qwen3.8model\prod029_logs",
    [string]$PyExe    = "G:\qwen3.8model\vllm-win029\Scripts\python.exe",
    [string]$ProbeTool = "G:\qwen3.8model\vllm-030win-git\tools\anchor_longctx.py",
    [int]$ProbeThreshold = 105,
    [int]$ProbeRepeats   = 3,
    [int]$MaxRestarts  = 3,
    [int]$SettleSec    = 10,
    [int]$HealthTimeoutSec = 420,
    [int]$Port = 8080
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$WatchLog = Join-Path $LogDir "watchdog.log"

function Write-Watch([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $WatchLog -Value $line -Encoding UTF8
}

function Kill-EngineProcesses {
    # Match the engine by command line (iron rule: Path/Image filters miss the
    # multiprocessing.spawn orphans, whose image is plain python.exe).
    $me = $PID
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $me -and $_.CommandLine -match "multiprocessing\.spawn|vllm\.entrypoints"
    } | ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
    }
}

function Test-Health {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

function Get-UsedMib {
    try {
        $out = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        $v = 0
        if ([int]::TryParse(($out | Select-Object -First 1).Trim(), [ref]$v)) { return $v }
    } catch {}
    return $null
}

function Start-Boot([int]$n) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outLog = Join-Path $LogDir ("watchdog_boot_{0}_p{1}.out.log" -f $stamp, $n)
    $errLog = Join-Path $LogDir ("watchdog_boot_{0}_p{1}.err.log" -f $stamp, $n)
    Write-Watch ("boot{0}: launching {1}" -f $n, $Launcher)
    Start-Process -FilePath "cmd.exe" -ArgumentList "/d", "/c", $Launcher `
        -WorkingDirectory $WorkDir -WindowStyle Hidden `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog | Out-Null
    return @{ n = $n; out = $outLog; err = $errLog; t0 = Get-Date }
}

function Wait-Healthy([datetime]$t0) {
    $deadline = $t0.AddSeconds($HealthTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        if (Test-Health) { return $true }
    }
    return $false
}

function Invoke-Probe([int]$n) {
    # anchor_longctx: 8k needle, warmup 0, repeats 3; median steady is the verdict.
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $jsonPath = Join-Path $LogDir ("watchdog_probe_{0}_p{1}.json" -f $stamp, $n)
    $args = @(
        $ProbeTool, "--base", "http://127.0.0.1:$Port",
        "--lengths", "8000", "--warmup", "0",
        "--repeats", "$ProbeRepeats",
        "--arm", "watchdog_p$n", "--out", $jsonPath
    )
    $p = Start-Process -FilePath $PyExe -ArgumentList $args `
        -WorkingDirectory $WorkDir -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir "watchdog_probe_last.out.log") `
        -RedirectStandardError (Join-Path $LogDir "watchdog_probe_last.err.log")
    if ($p.ExitCode -ne 0 -or -not (Test-Path $jsonPath)) {
        Write-Watch ("probe p{0}: FAILED (rc={1})" -f $n, $p.ExitCode)
        return $null
    }
    try {
        $d = Get-Content -Path $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $r0 = $d.results[0]
        $steadies = @()
        foreach ($rep in $r0.repeats) {
            if ($rep.PSObject.Properties.Name -contains "steady_tok_s") {
                $steadies += [double]$rep.steady_tok_s
            }
        }
        if ($steadies.Count -eq 0) { return $null }
        $sorted = $steadies | Sort-Object
        $median = $sorted[[int][math]::Floor($sorted.Count / 2)]
        $needleMiss = $steadies.Count - ($r0.repeats | Where-Object { $_.needle_hit } | Measure-Object).Count
        return @{ median = [math]::Round($median, 2); all = ($steadies -join "/"); needle_miss = $needleMiss }
    } catch {
        Write-Watch ("probe p{0}: parse error {1}" -f $n, $_.Exception.Message)
        return $null
    }
}

# ---- main ----
Write-Watch ("watchdog begin: launcher={0} probeThreshold={1} maxRestarts={2}" -f $Launcher, $ProbeThreshold, $MaxRestarts)

$attempt = 0
while ($true) {
    Kill-EngineProcesses
    Start-Sleep -Seconds 6

    $b = Start-Boot $attempt
    if (-not (Wait-Healthy $b.t0)) {
        Write-Watch ("boot{0}: health FAILED after {1}s" -f $attempt, $HealthTimeoutSec)
        if ($attempt -ge $MaxRestarts) {
            Write-Watch "boot: giving up (unhealthy), leaving processes for inspection"
            exit 3
        }
        $attempt++
        continue
    }

    Start-Sleep -Seconds $SettleSec
    $used = Get-UsedMib

    $probe = Invoke-Probe $attempt
    if ($null -eq $probe) {
        Write-Watch ("boot{0}: probe failed -> keep service for inspection" -f $attempt)
        exit 4
    }
    if ($probe.needle_miss -gt 0) {
        Write-Watch ("boot{0}: probe needle MISS x{1} (steady {2}) -> correctness issue, NOT restarting; keep service" -f $attempt, $probe.needle_miss, $probe.all)
        exit 4
    }

    $bootS = [int]((Get-Date) - $b.t0).TotalSeconds
    if ($probe.median -lt $ProbeThreshold) {
        Write-Watch ("boot{0}: SLOW verdict median={1} < {2} tok/s (runs {3}; vram {4} MiB; boot {5}s) -> kill and retry" -f $attempt, $probe.median, $ProbeThreshold, $probe.all, $used, $bootS)
        if ($attempt -ge $MaxRestarts) {
            Write-Watch ("boot{0}: retries exhausted, keeping this (slow-tier) service running" -f $attempt)
            exit 2
        }
        $attempt++
        continue
    }

    Write-Watch ("boot{0}: FAST verdict median={1} >= {2} tok/s (runs {3}; vram {4} MiB; boot {5}s) -> service ready" -f $attempt, $probe.median, $ProbeThreshold, $probe.all, $used, $bootS)
    exit 0
}
