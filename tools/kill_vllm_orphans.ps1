# kill_vllm_orphans.ps1 — kill vllm serve + EngineCore spawn orphans by command line
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq 'python.exe' -and $_.CommandLine -and (
    $_.CommandLine -like '*cli.main serve*' -or
    $_.CommandLine -like '*multiprocessing.spawn*' -or
    $_.CommandLine -like '*vllm.entrypoints*'
  )
} | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force
  Write-Output ("killed " + $_.ProcessId)
}
