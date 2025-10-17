<# 
Runs build_reddit_daily → fetch_prices → merge_features → enrich_features
across the full available date range under data/derived/submission_tickers.
Optionally clamp with -Start/-End (YYYY-MM-DD).

Usage:
  pwsh .\Run-RedditMarketPipeline-Full.ps1 [-Python "C:\path\to\python.exe"] [-Batch 150] [-Force] [-Start YYYY-MM-DD] [-End YYYY-MM-DD]
#>

param(
  [string]$Python,
  [int]$Batch = 150,
  [switch]$Force,
  [string]$Start,
  [string]$End
)

# --- Paths ---
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

if (!(Test-Path $SubmissionRoot)) { Fail "Not found: $SubmissionRoot" }

# derive min/max day from folder names unless overridden
$dayDirs = Get-ChildItem -Path $SubmissionRoot -Directory |
  Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' } |
  Sort-Object Name

if ($dayDirs.Count -eq 0) { Fail "No day folders like YYYY-MM-DD under $SubmissionRoot" }

if (-not $Start -or -not ($Start -match '^\d{4}-\d{2}-\d{2}$')) { $Start = $dayDirs[0].Name }
if (-not $End   -or -not ($End   -match '^\d{4}-\d{2}-\d{2}$')) { $End   = $dayDirs[-1].Name }

Info "[Full] Range: $Start → $End"

$forceFlag = @()
if ($Force) { $forceFlag = @("--force") }

# ensure outputs exist
$null = New-Item -ItemType Directory -Force -Path $OutRedditDaily, $OutPrices, $OutMerged, $OutEnriched | Out-Null

# 1) build_reddit_daily
& $py (Join-Path $ScriptsDir "build_reddit_daily.py") `
  --in-root $SubmissionRoot `
  --out-root $OutRedditDaily `
  --start $Start --end $End @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "build_reddit_daily failed" } else { Ok "build_reddit_daily OK" }

# 2) fetch_prices
& $py (Join-Path $ScriptsDir "fetch_prices.py") `
  --reddit-daily-root $OutRedditDaily `
  --out-root $OutPrices `
  --start $Start --end $End `
  --batch $Batch @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "fetch_prices failed" } else { Ok "fetch_prices OK" }

# 3) merge_features
& $py (Join-Path $ScriptsDir "merge_features.py") `
  --reddit-root $OutRedditDaily `
  --prices-root $OutPrices `
  --out-root $OutMerged `
  --start $Start --end $End @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "merge_features failed" } else { Ok "merge_features OK" }

# 4) enrich_features
& $py (Join-Path $ScriptsDir "enrich_features.py") `
  --merged-root $OutMerged `
  --out-root $OutEnriched `
  --start $Start --end $End `
  --lags 1 2 3 `
  --rolls 3 7 14 30 @forceFlag
if ($LASTEXITCODE -ne 0){ Fail "enrich_features failed" } else { Ok "enrich_features OK" }

Ok "`nFull run complete for $Start → $End"
