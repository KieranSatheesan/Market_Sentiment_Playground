# run_reddit_ticker_pipeline
param(
  [string]$CleanRoot = "data\RedditDumps\cleaned",
  [string]$ReqDir    = "batch\Requests",
  [string]$ResDir    = "batch\Results",
  [string]$OutRoot   = "data\derived\submission_tickers",
  [string]$Model     = "gpt-4.1-mini",
  [string]$Schema    = "full",
  [int]$GroupSize    = 16,
  [int]$MaxChars     = 1500
)

Write-Host "STEP 1: prepare requests..."
py scripts\prepare_requests.py `
  --clean_root $CleanRoot `
  --out_dir $ReqDir `
  --model $Model `
  --schema $Schema `
  --group-size $GroupSize `
  --max-chars $MaxChars

Write-Host "STEP 2: submit & collect..."
py scripts\submit_and_collect.py `
  --requests_dir $ReqDir `
  --results_dir $ResDir `
  --desc "reddit_ticker_sentiment_full"

Write-Host "STEP 3: parse & explode..."
$results = Get-ChildItem $ResDir -Filter "res_req_*.jsonl"
foreach ($r in $results) {
  $day = ($r.BaseName -replace '^res_req_', '')
  $outDir = Join-Path $OutRoot $day
  if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

  py scripts\parse_results.py `
    --clean_day_dir (Join-Path $CleanRoot $day) `
    --results_jsonl (Join-Path $ResDir $r.Name) `
    --out_dir $outDir

  py scripts\explode_per_ticker.py `
    --annotated_parquet (Join-Path $outDir "annotated.parquet") `
    --out_parquet (Join-Path $outDir "exploded.parquet")
}

Write-Host "ALL DONE ✅"
