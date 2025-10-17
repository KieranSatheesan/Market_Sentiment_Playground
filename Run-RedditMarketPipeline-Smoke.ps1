<# 
Runs build_reddit_daily → fetch_prices → merge_features → enrich_features
for a single auto-detected day that has both annotated.parquet and exploded.parquet.

Usage:
  pwsh .\Run-RedditMarketPipeline-Smoke.ps1 [-Python "C:\path\to\python.exe"] [-Batch 150] [-Force]
#>

param(
  [string]$Python,
  [int]$Batch = 150,
  [switch]$Force
)

# --- Paths (repo-root relative) ---
$RepoRoot        = $PSScriptRoot
$ScriptsDir      = Join-Path $RepoRoot "scripts"
$SubmissionRoot  = Join-Path $RepoRoot "data\derived\submission_tickers"
$OutRedditDaily  = Join-Path $RepoRoot "data\derived\featuresets\reddit_daily"
$OutPrices       = Join-Path $RepoRoot "data\derived\featuresets\prices_yahoo"
$OutMerged       = Join-Path $RepoRoot "data\derived\featuresets\merged"
$OutEnriched     = Join-Path $RepoRoot "data\derived\featuresets\merged_enriched"

function Fail($m){ Write-Host "[ERROR] $m" -ForegroundColor Red; exit 1 }
function Info($m){ Write-Host $m -ForegroundColor Yellow }
function Ok($m){ Write-Host $m -ForegroundColor Green }

function Resolve-Python {
  param([string]$P)
  if ($P) { return $P }
  $candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    "py", "python", "python3"
  )
  foreach ($c in $candidates) {
    try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { return $c } } catch {}
  }
  Fail "Python not found. Provide -Python path."
}

$py = Resolve-Python -P $Python

# pick 1 day with both annotated + exploded
if (!(Test-Path $SubmissionRoot)) { Fail "Not found: $SubmissionRoot" }
$dayDir = Get-ChildItem -Path $SubmissionRoot -Directory |
  Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' } |
  Sort-Object Name |
  Where-Object {
    (Test-Path (Join-Path $_.FullName 'annotated.parquet')) -and
    (Test-Path (Join-Path $_.FullName 'exploded.parquet'))
  } | Select-Object -First 1

if (-not $dayDir) { Fail "No suitable day folders under $SubmissionRoot" }
$Day = $dayDir.Name
Info "[Smoke] Day: $Day"

$forceFlag = @()
if ($Force) { $forceFlag = @("--force") }

# ensure outputs exist
$null = New-Item -ItemType Directory -Force -Path $OutRedditDaily, $OutPrices, $OutMerged, $OutEnriched | Out-Null

# 1) build_reddit_daily
& $py (Join-Path $ScriptsDir "build_reddit_daily.py") `
  --in-root $SubmissionRoot `
  --out-root $OutRedditDaily `
  --start $Day --end $Day @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "build_reddit_daily failed" } else { Ok "build_reddit_daily OK" }

# 2) fetch_prices
& $py (Join-Path $ScriptsDir "fetch_prices.py") `
  --reddit-daily-root $OutRedditDaily `
  --out-root $OutPrices `
  --start $Day --end $Day `
  --batch $Batch @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "fetch_prices failed" } else { Ok "fetch_prices OK" }

# 3) merge_features
& $py (Join-Path $ScriptsDir "merge_features.py") `
  --reddit-root $OutRedditDaily `
  --prices-root $OutPrices `
  --out-root $OutMerged `
  --start $Day --end $Day @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "merge_features failed" } else { Ok "merge_features OK" }

# 4) enrich_features
& $py (Join-Path $ScriptsDir "enrich_features.py") `
  --merged-root $OutMerged `
  --out-root $OutEnriched `
  --start $Day --end $Day `
  --lags 1 2 3 `
  --rolls 3 7 14 30 @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "enrich_features failed" } else { Ok "enrich_features OK" }

Ok "`nSmoke run complete for $Day"
