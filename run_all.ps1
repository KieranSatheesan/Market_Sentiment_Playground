# run_all.ps1
$ErrorActionPreference = "Stop"

# Activate venv
& ".\.venv\Scripts\Activate.ps1"

# Make sure logs dir exists
New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null

# Months to process
$months = @(
  @{ m="05"; cfg="configs/rs_2025-05.yaml" },
  @{ m="06"; cfg="configs/rs_2025-06.yaml" },
  @{ m="07"; cfg="configs/rs_2025-07.yaml" }
)

foreach ($job in $months) {
  Write-Host "=== Processing 2025-$($job.m) ===" -ForegroundColor Cyan
  python "scripts/filter_rs_month.py" --config $job.cfg
  Write-Host "=== Finished 2025-$($job.m) ===" -ForegroundColor Green
}
