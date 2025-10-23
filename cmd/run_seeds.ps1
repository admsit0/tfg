Param(
  [string]$BaseConfig = "configs/fashion_cv_l1_dropoutLows_l2.yaml",
  [int[]]$Seeds = @(17, 42, 50, 190, 224)
)

Write-Host "BaseConfig: $BaseConfig"
Write-Host "Seeds: $($Seeds -join ', ')"

$python = "python"

& $python "cmd/run_seeds.py" --base-config $BaseConfig --seeds $Seeds
