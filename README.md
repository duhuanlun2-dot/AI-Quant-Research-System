# AI Quant Research System

Research-grade AI quant pipeline for local experimentation with market data ingestion, news scoring, factor construction, chronological OOS modeling, portfolio backtesting, execution-aware simulation, and trust-audit reports.

> Research only. This project is not financial advice, not an investment recommendation, and not a production trading system. Backtest and model outputs can be wrong, biased, stale, or non-reproducible if the underlying data is incomplete.
>
> This program was developed with Vibe Coding. If you find bugs, questionable assumptions, or reproducibility issues, please open an issue or submit feedback.

## What It Does

- Point-in-time S&P 500 universe support, with strict checks for real historical membership data.
- Daily OHLCV and benchmark market data ingestion.
- Multi-source news archive with effective trading date assignment.
- Local heuristic or OpenAI-compatible LLM news scoring.
- Local DuckDB storage plus Parquet backup hooks.
- Data quality rules for adjusted prices, missing values, abnormal returns, and trading-day alignment.
- Factor IC / RankIC reporting, feature importance, final holdout, and walk-forward training.
- Strategy vs SPY visualization, transaction costs, slippage, execution price models, holdings export, and trust audits.

## Important Limitations

- Historical S&P 500 survivorship-bias control requires a legal PIT universe file from a real data vendor. Current Wikipedia constituents are not enough.
- Yahoo Finance / yfinance data is convenient for research but not a professional source of truth.
- Built-in news collectors are best-effort and may be rate-limited or incomplete.
- Execution simulation is bar-level, not order-book-level.
- The GUI and one-click scripts are intended for local research workflows.

## Safety And Secrets

- Do not commit `config.toml`, `.env`, DuckDB files, model outputs, or API keys.
- Commit `config.example.toml` only.
- Prefer environment variables for external LLM keys, for example `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, or `OPENROUTER_API_KEY`.
- Review data-source licenses before publishing derived datasets.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy config.example.toml config.toml
```

Fill API keys in `config.toml` when using real external providers.

Optional advanced tree models:

```powershell
pip install -e ".[advanced-models]"
```

## Quick Demo

This minimal local flow uses example data and the built-in heuristic scorer:

```powershell
python -m ai_quant_research_system.cli init-db --config config.toml
python -m ai_quant_research_system.cli ingest-universe --config config.toml --csv examples/sp500_constituents.csv
python -m ai_quant_research_system.cli ingest-prices --config config.toml --csv examples/daily_prices.csv
python -m ai_quant_research_system.cli ingest-benchmark --config config.toml --csv examples/benchmark_daily.csv
python -m ai_quant_research_system.cli ingest-news --config config.toml --jsonl examples/raw_news.jsonl
python -m ai_quant_research_system.cli build-clean-views --config config.toml
python -m ai_quant_research_system.cli run-news-factor-pipeline --config config.toml --limit 100
python -m ai_quant_research_system.cli build-factor-table --config config.toml
python -m ai_quant_research_system.cli train-models --config config.toml --models rf
python -m ai_quant_research_system.cli run-backtest --config config.toml
```

## Common commands

GUI launcher:

```powershell
.\run_ui.bat
```

The GUI provides separate **Fetch / Update Data**, **Run Modeling Pipeline**, **Run Backtest**, and **Optimize Portfolio** workflows, model selection, local/external news scoring selection, parameter fields, live logs, visualization, row-count refresh, and a **Force Stop All Tasks** button for killing stuck project CLI processes. Existing price/news data is skipped by default unless you enable force refresh.

```powershell
python -m ai_quant_research_system.cli init-db --config config.toml
python -m ai_quant_research_system.cli ingest-universe --config config.toml --csv examples/sp500_constituents.csv
python -m ai_quant_research_system.cli ingest-prices --config config.toml --csv examples/daily_prices.csv
python -m ai_quant_research_system.cli ingest-sp500-current --config config.toml
python -m ai_quant_research_system.cli ingest-sp500-yfinance-prices --config config.toml --days 45
python -m ai_quant_research_system.cli ingest-news --config config.toml --jsonl examples/raw_news.jsonl
python -m ai_quant_research_system.cli ingest-news-builtins --config config.toml --tickers AAPL MSFT SPY QQQ
python -m ai_quant_research_system.cli build-clean-views --config config.toml
python -m ai_quant_research_system.cli run-news-factor-pipeline --config config.toml --limit 1000
python -m ai_quant_research_system.cli build-factor-table --config config.toml
python -m ai_quant_research_system.cli train-models --config config.toml
python -m ai_quant_research_system.cli run-backtest --config config.toml
python -m ai_quant_research_system.cli backup-parquet --config config.toml
```

The CSV/JSONL ingestion paths are production-friendly staging interfaces. Provider-specific collectors can be added behind the same normalized records without changing the database contract.

`ingest-sp500-current` fetches the current S&P 500 list from Wikipedia. This is useful for live research, but it is not a historical point-in-time constituent source. Use Sharadar, CRSP, or another licensed PIT source for bias-safe historical backtests.

Historical universe mode:

```powershell
python -m ai_quant_research_system.cli ingest-universe --config config.toml --csv path\to\sp500_constituents_pit.csv
python -m ai_quant_research_system.cli build-factor-table --config config.toml --require-pit-universe
```

The required CSV shape is shown in `examples/sp500_constituents_pit_template.csv`:

- `date`: source snapshot / ingestion date for the membership record.
- `ticker`: normalized ticker.
- `company`: company name.
- `entry_date`: first date the company was in the S&P 500.
- `exit_date`: last date in the S&P 500, blank for active members.
- `gics_sector`, `gics_industry`: sector metadata used for factor and exposure reports.
- `source`: vendor or data source label.

When enough historical membership metadata is available, `build-factor-table` uses point-in-time membership filtering: a stock contributes factors only on dates between its `entry_date` and `exit_date`. If the table looks like a current-only constituent list, the command falls back to latest-constituent mode unless `--require-pit-universe` is set. This conservative check avoids treating Wikipedia current constituents plus date-added fields as a complete historical universe.

For the system to treat the universe as real PIT data, the table must pass stricter checks:

- At least 80% of rows must have `entry_date` or `exit_date`.
- Removed historical members must be present with non-empty `exit_date`.
- The file must contain enough removed tickers to prove it is not just the current S&P 500 list.
- Rows with `entry_date > exit_date` are rejected.
- Active member counts at the price start/end dates must be plausible for S&P 500 membership.

Use the GUI **Import Historical Universe** button to select a vendor CSV, then click **Check Universe**. When the status becomes `PIT ready`, enable **Require historical PIT universe** before building factors or running the full pipeline.

Phase 2 uses a local heuristic scorer by default (`[llm].provider = "heuristic"`) so the pipeline can run without paid API keys. It writes per-article scores to `news_scores`, daily raw aggregates to `news_factors_raw`, and final composite factors to `news_factors`.

Built-in news collectors:

- `ingest-yfinance-news`: pulls recent Yahoo Finance news for selected tickers.
- `ingest-sec-news`: pulls SEC EDGAR 8-K / 10-Q / 10-K filing events.
- `ingest-news-builtins`: runs both collectors and writes normalized rows into `raw_news`.
- `ingest-sp500-news-builtins`: pulls the current S&P 500 universe, then collects Yahoo Finance news and SEC filings for every ticker. Use conservative per-ticker limits to avoid source throttling.

Example full S&P 500 news pull:

```powershell
python -m ai_quant_research_system.cli ingest-sp500-news-builtins --config config.toml --yahoo-limit-per-ticker 3 --sec-days 90 --sec-limit-per-ticker 2
python -m ai_quant_research_system.cli build-clean-views --config config.toml
python -m ai_quant_research_system.cli run-news-factor-pipeline --config config.toml --limit 5000
python -m ai_quant_research_system.cli build-factor-table --config config.toml
```

Phase 3 builds `factor_table` with raw traditional factors, AI news factors, and `z_` cross-sectional standardized factors for model training.

During factor construction the CLI prints `Universe mode`. For serious historical backtests, require `point_in_time_constituents`; `latest_constituents_fallback` means survivorship bias is still present.

Phase 4 builds `future_excess_return_5d`, writes a `dataset.parquet` artifact, trains three chronological-split regression models, stores artifacts under `models/<model_version>/`, writes `model_predictions`, and records metadata in `model_registry`.

Model selection:

```powershell
python -m ai_quant_research_system.cli train-models --config config.toml --models lightgbm rf
```

Phase 5-7 converts predictions into trading signals, target positions, and an out-of-sample backtest against SPY. The default policy buys the top 10% ranked stocks, weights them by predicted score, caps each stock at 5%, drops positions below 0.5%, renormalizes uncapped weights, rebalances every 5 trading days, excludes benchmark ETFs (`SPY`, `QQQ`) from holdings, and subtracts 10 bps transaction cost plus 5 bps slippage. Rebalances are treated as after-close decisions, so new weights begin earning returns from the next trading day.

Backtest example:

```powershell
python -m ai_quant_research_system.cli run-backtest --config config.toml --top-pct 0.10 --max-weight 0.05 --min-weight 0.005 --rebalance-days 5 --transaction-bps 10 --slippage-bps 5 --initial-capital 1000000
```

Constrained optimization example:

```powershell
python -m ai_quant_research_system.cli optimize-portfolio --config config.toml --max-drawdown-limit 0.10 --transaction-bps 10 --slippage-bps 5 --initial-capital 1000000
```

The optimizer grid-searches selection ratio, max single-name weight, min single-name weight, rebalance interval, and target exposure. It first filters candidates with max drawdown within the configured limit, then selects the highest out-of-sample Sharpe ratio. The final weights are written to `optimized_weights.csv` and `optimized_weights.json`.

Walk-forward validation:

```powershell
python -m ai_quant_research_system.cli train-walk-forward-models --config config.toml --models rf --min-train-days 252 --step-days 21 --final-holdout-pct 0.20
```

The walk-forward command repeatedly trains only on past dates, predicts the next window, and reserves the final holdout segment for reporting. It is intentionally not part of the default one-click run because it can be slower.

Trust audit:

```powershell
python -m ai_quant_research_system.cli run-trust-audit --config config.toml
```

The audit writes survivorship-bias checks, sector/beta exposure, liquidity, parameter sensitivity, monthly returns, and multi-window backtest files under `models/<model_version>/`.

Execution layer:

```powershell
python -m ai_quant_research_system.cli optimize-portfolio --config config.toml --execution-price-model open
```

Signals are converted into orders after the signal date close, then filled on the next trading day using the selected execution price model: `open`, `vwap`, or `close`. The system writes order, fill, and current-position files so the output is closer to a paper-trading blotter than a pure weight curve.

Backtest artifacts are written under `models/<model_version>/`:

- `backtest_report.png`: strategy vs SPY, drawdown, and turnover.
- `backtest_metrics.json`: return, SPY return, excess return, CAGR, Sharpe, max drawdown, and turnover.
- `latest_holdings.csv` / `latest_holdings.json`: current rebalance holdings and weights.
- `optimized_weights.csv` / `optimized_weights.json`: final constrained optimizer weights.
- `optimization_report.csv` / `optimization_summary.json`: all tested candidates and the selected Sharpe-optimal candidate under the drawdown constraint.
- `backtest_trades.csv`: simulated rebalance trades and costs.
- `execution_orders.csv`, `execution_fills.csv`, `current_positions.csv`, `latest_current_positions.csv`: order-driven execution layer outputs.
- `trust_audit_report.md` / `trust_audit_summary.json`: credibility audit and caveats.
- `sector_beta_exposure.csv`, `liquidity_report.csv`, `parameter_sensitivity.csv`, `multi_window_backtest.csv`, `monthly_returns.csv`: supporting audit files.

News scoring providers:

- `provider = "heuristic"` uses the built-in local scorer and needs no API key.
- `provider = "openai_compatible"` calls a `/chat/completions` compatible external API. Set `[llm].base_url`, `[llm].model`, and either `[llm].api_key` or the environment variable named by `[llm].api_key_env`.

The GUI **News Scoring** panel can write these `[llm]` settings for common OpenAI-compatible providers:

- OpenAI-compatible: `base_url = "https://api.openai.com/v1"`, `api_key_env = "OPENAI_API_KEY"`.
- DeepSeek-compatible: `base_url = "https://api.deepseek.com/v1"`, `api_key_env = "DEEPSEEK_API_KEY"`.
- Qwen-compatible: `base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"`, `api_key_env = "DASHSCOPE_API_KEY"`.
- OpenRouter-compatible: `base_url = "https://openrouter.ai/api/v1"`, `api_key_env = "OPENROUTER_API_KEY"`.
- Custom OpenAI-compatible: edit `base_url`, `model`, and `api_key_env` directly in the UI.

Historical PIT universe data is separate from LLM scoring. A model can score news, but it cannot create a bias-safe historical index membership table; that still needs a real vendor CSV with removed members and exit dates.

## Responsible Use

This repository is intended for education and research. Before using outputs in any investment process, validate data rights, point-in-time membership, corporate actions, benchmark construction, transaction costs, liquidity, execution assumptions, and live monitoring.

## License

MIT. See `LICENSE`.
