# qdq with olive
$configFiles = @("google_bert_qdq_npu_squad.json", "intel_bert_qdq_npu_glue_mrpc.json", "llmlingua2_bert_qdq_npu_meetingbank.json")

foreach ($configFile in $configFiles) {
  Write-Host "Running olive with config: $configFile"

  $command = "olive run --config $configFile"

  try {
    & $command
    # $output = Invoke-Expression $command 2>&1

    if ($LASTEXITCODE -ne 0) {
      Write-Warning "Command '$command' returned non-zero exit code: $($LASTEXITCODE)"
      #Write-Warning "Error output was: $($output -join "`r`n")"
    } else {
      Write-Host "Command '$command' completed successfully."
    }
  }
  catch {
    Write-Error "An error occurred while running '$command': $($_.Exception.Message)"
  }
}

# eval with qdq'ed models
$tasks = @("scl-glue-mrpc", "qa-squad", "tcl-llmlingua2-meetingbank")

foreach ($task in $tasks) {
  Write-Host "Evaluating: $task"

  $command = "python eval.py --task $task"

  try {
    & $cmd
  }
  catch {
    Write-Error "An error occurred while running '$command': $($_.Exception.Message)"
  }
}

Write-Host "All Done!"