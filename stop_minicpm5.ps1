# stop_minicpm5.ps1 - kill EVERY vLLM instance for this deployment.
#
# Why this exists: killing "the PID listening on 8081" is not enough. Two vLLM
# instances can both bind 0.0.0.0:8081 (SO_REUSEPORT), so a port-based kill
# leaves one behind holding ~2.3 GiB of VRAM. That happened, and the leftover
# was misread as "BGE-M3 won't fit". Match on the command line instead.
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'vllm\.entrypoints' -or $_.CommandLine -match 'multiprocessing-fork'
}
if (-not $targets) {
    Write-Output 'no vLLM python processes found'
} else {
    foreach ($t in $targets) {
        Write-Output ("killing pid {0} (parent {1})" -f $t.ProcessId, $t.ParentProcessId)
        Stop-Process -Id $t.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 6
$left = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'vllm\.entrypoints' -or $_.CommandLine -match 'multiprocessing-fork'
}
if ($left) { Write-Output ("WARNING: still alive: " + ($left.ProcessId -join ',')) }
else { Write-Output 'clean: no vLLM processes remain' }
