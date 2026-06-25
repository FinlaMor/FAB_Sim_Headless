# keep_collecting.ps1 — durable CC self-play collection.
# Runs cc_selfplay in back-to-back batches with advancing seeds, forever,
# surviving normal batch completion AND crashes. Stop by killing this process
# tree (the wrapper + its current python child).
#
# Gameplay bot = cc_warm4 (promoted champion 2026-06-25). Sideboard = BC + 0.7 exploration.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Joseph\Desktop\FAB_Sim_Headless"
Set-Location $root
$env:PYTHONPATH = $root
$py = "C:\Users\Joseph\AppData\Local\Programs\Python\Python312\python.exe"
$base = 500000
$batch = 0
while ($true) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $log = "outputs\collect_${stamp}_seed${base}.err"
  Write-Output "[keep-collecting] batch=$batch base-seed=$base log=$log start=$(Get-Date -Format o)"
  & $py -m python.gameplay.cc_selfplay `
      --adapters 8000-8007 --pairs 200 --games 2 `
      --model outputs/models/sideboard/sideboard_bc.pt `
      --gameplay-model outputs/models/cc_warm4/iql_gameplay.pt `
      --explore-sideboard 0.7 --step-cap 800 --base-seed $base 2>> $log
  Write-Output "[keep-collecting] batch=$batch exited code=$LASTEXITCODE end=$(Get-Date -Format o)"
  $base += 1000
  $batch += 1
  Start-Sleep -Seconds 5   # brief settle so a crash-looping adapter can recover
}
