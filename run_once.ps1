$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "== AI Quant Research System: one-click setup ==" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Host "Installing dependencies..."
& $Python -m pip install --upgrade pip
& $Pip install -r requirements.txt

if (-not (Test-Path "config.toml")) {
    Write-Host "Creating config.toml from config.example.toml..."
    Copy-Item "config.example.toml" "config.toml"
}

$ConfigText = Get-Content -LiteralPath "config.toml" -Raw
if ($ConfigText -match 'provider\s*=\s*"heuristic"') {
    $ConfigText = $ConfigText -replace 'rate_limit_per_minute\s*=\s*50', 'rate_limit_per_minute = 3000'
    Set-Content -LiteralPath "config.toml" -Value $ConfigText -Encoding UTF8
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"

Write-Host "Initializing DuckDB..."
& $Python -m ai_quant_research_system.cli init-db --config config.toml

Write-Host "Loading example universe..."
& $Python -m ai_quant_research_system.cli ingest-universe --config config.toml --csv examples\sp500_constituents.csv

Write-Host "Loading example prices..."
& $Python -m ai_quant_research_system.cli ingest-prices --config config.toml --csv examples\daily_prices.csv

Write-Host "Downloading current S&P 500 universe and Yahoo Finance prices..."
$YahooStart = (Get-Date).AddDays(-730).ToString("yyyy-MM-dd")
$YahooEnd = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
Write-Host "Yahoo Finance date range: $YahooStart to $YahooEnd"
& $Python -m ai_quant_research_system.cli ingest-sp500-yfinance-prices --config config.toml --start $YahooStart --end $YahooEnd

Write-Host "Downloading Yahoo Finance benchmark ETFs..."
& $Python -m ai_quant_research_system.cli ingest-yfinance-prices --config config.toml --tickers SPY QQQ --start $YahooStart --end $YahooEnd

Write-Host "Loading example benchmarks..."
& $Python -m ai_quant_research_system.cli ingest-benchmark --config config.toml --csv examples\benchmark_daily.csv

Write-Host "Loading example news..."
& $Python -m ai_quant_research_system.cli ingest-news --config config.toml --jsonl examples\raw_news.jsonl

Write-Host "Downloading full S&P 500 news from Yahoo Finance and SEC EDGAR..."
$SecSince = (Get-Date).AddDays(-90).ToString("yyyy-MM-dd")
Write-Host "SEC filing lookback starts: $SecSince"
& $Python -m ai_quant_research_system.cli ingest-sp500-news-builtins --config config.toml --yahoo-limit-per-ticker 3 --sec-since $SecSince --sec-limit-per-ticker 2

Write-Host "Rebuilding clean views..."
& $Python -m ai_quant_research_system.cli build-clean-views --config config.toml

Write-Host "Running Phase 2 news factor pipeline..."
& $Python -m ai_quant_research_system.cli run-news-factor-pipeline --config config.toml --limit 5000

Write-Host "Building Phase 3 factor table..."
& $Python -m ai_quant_research_system.cli build-factor-table --config config.toml

Write-Host "Training Phase 4 prediction models..."
& $Python -m ai_quant_research_system.cli train-models --config config.toml

Write-Host "Running Phase 5-7 signals, portfolio, and constrained optimization..."
& $Python -m ai_quant_research_system.cli optimize-portfolio --config config.toml --max-drawdown-limit 0.10 --transaction-bps 10 --slippage-bps 5 --initial-capital 1000000 --execution-price-model open

Write-Host "Running trust audit reports..."
& $Python -m ai_quant_research_system.cli run-trust-audit --config config.toml

Write-Host "Current row counts..."
& $Python -m ai_quant_research_system.cli count-rows --config config.toml

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Database: $ProjectRoot\data\ai_quant.duckdb"
Write-Host ""
Write-Host "To run commands manually later:"
Write-Host '  .\.venv\Scripts\Activate.ps1'
Write-Host '  $env:PYTHONPATH="src"'
Write-Host '  python -m ai_quant_research_system.cli --help'

Read-Host "Press Enter to exit"
