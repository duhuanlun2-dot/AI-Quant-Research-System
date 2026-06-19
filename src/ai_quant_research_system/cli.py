from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .config import load_config
from .database import connect
from .database import backup_to_parquet, build_clean_views, init_db
from .factor_engine import build_factor_table, universe_coverage
from .ingest import (
    insert_benchmark,
    insert_constituents,
    insert_news,
    insert_prices,
    read_benchmark_csv,
    read_constituents_csv,
    read_news_jsonl,
    read_prices_csv,
)
from .news_factor_engine import aggregate_news_factors, pending_news_count, score_news_batch
from .news_collectors import fetch_sec_filings_news, fetch_yahoo_finance_news
from .model_engine import train_prediction_models, train_walk_forward_models
from .portfolio_engine import run_portfolio_backtest, run_portfolio_optimization
from .trust_audit_engine import run_trust_audit
from .universe import fetch_current_sp500_constituents
from .yahoo_finance import fetch_daily_prices


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.toml", help="Path to TOML config file.")


def filter_tickers_missing_price_range(db_path: Path, tickers: list[str], start: date, end: date) -> list[str]:
    end_cover = end - timedelta(days=3)
    normalized = [ticker.upper().replace(".", "-") for ticker in tickers]
    with connect(db_path) as con:
        rows = con.execute(
            """
            SELECT ticker, min(date) AS min_date, max(date) AS max_date, count(*) AS n
            FROM daily_prices
            WHERE ticker IN (SELECT * FROM unnest(?))
            GROUP BY ticker
            """,
            [normalized],
        ).fetchall()
    covered = {
        row[0]
        for row in rows
        if row[1] is not None and row[2] is not None and row[1] <= start and row[2] >= end_cover and row[3] > 10
    }
    missing = [ticker for ticker in normalized if ticker not in covered]
    if covered:
        print(f"Skipping {len(covered)} tickers with existing price coverage.")
    return missing


def filter_tickers_missing_news(db_path: Path, tickers: list[str], source_kind: str, min_rows: int) -> list[str]:
    normalized = [ticker.upper().replace(".", "-") for ticker in tickers]
    if min_rows <= 0:
        return normalized
    if source_kind == "sec":
        condition = "source = 'SEC EDGAR'"
    else:
        condition = "source <> 'SEC EDGAR'"
    with connect(db_path) as con:
        rows = con.execute(
            f"""
            SELECT ticker, count(*) AS n
            FROM raw_news
            WHERE ticker IN (SELECT * FROM unnest(?))
              AND {condition}
            GROUP BY ticker
            """,
            [normalized],
        ).fetchall()
    covered = {row[0] for row in rows if row[1] >= min_rows}
    if covered:
        print(f"Skipping {len(covered)} tickers with existing {source_kind} news.")
    return [ticker for ticker in normalized if ticker not in covered]


def cmd_init_db(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    init_db(cfg.database.path)
    print(f"Initialized DuckDB at {cfg.database.path}")


def cmd_ingest_universe(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = read_constituents_csv(args.csv)
    count = insert_constituents(cfg.database.path, rows)
    print(f"Loaded {count} constituent rows.")


def cmd_ingest_sp500_current(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = fetch_current_sp500_constituents(as_of=args.as_of)
    count = insert_constituents(cfg.database.path, rows)
    print(f"Loaded {count} current S&P 500 constituent rows.")


def cmd_ingest_prices(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = read_prices_csv(args.csv, threshold=cfg.ingestion.suspicious_return_threshold)
    count = insert_prices(cfg.database.path, rows)
    print(f"Loaded {count} price rows.")


def cmd_ingest_yfinance_prices(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = args.tickers
    if not args.force:
        tickers = filter_tickers_missing_price_range(cfg.database.path, tickers, args.start, args.end)
    if not tickers:
        print("All requested tickers already have price coverage. Nothing to download.")
        return
    rows = fetch_daily_prices(tickers, args.start, args.end)
    count = insert_prices(cfg.database.path, rows)
    print(f"Loaded {count} yfinance price rows.")


def cmd_ingest_sp500_yfinance_prices(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    end = args.end or (date.today() + timedelta(days=1))
    start = args.start or (end - timedelta(days=args.days))
    as_of = args.as_of or date.today()

    rows = fetch_current_sp500_constituents(as_of=as_of)
    insert_constituents(cfg.database.path, rows)
    tickers = [row.ticker for row in rows]

    if not args.force:
        tickers = filter_tickers_missing_price_range(cfg.database.path, tickers, start, end)
    if not tickers:
        print("All S&P 500 tickers already have price coverage. Nothing to download.")
        return
    price_rows = fetch_daily_prices(tickers, start=start, end=end, batch_size=args.batch_size)
    count = insert_prices(cfg.database.path, price_rows)
    print(f"Loaded {len(tickers)} S&P 500 tickers and {count} yfinance price rows.")


def cmd_count_rows(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    with connect(cfg.database.path) as con:
        for table in (
            "sp500_constituents",
            "daily_prices",
            "benchmark_daily",
            "raw_news",
            "clean_news",
            "news_scores",
            "news_factors_raw",
            "news_factors",
            "factor_table",
            "model_predictions",
            "model_registry",
            "trading_signals",
            "portfolio_positions",
            "backtest_results",
            "backtest_trades",
            "execution_orders",
            "execution_fills",
            "current_positions",
        ):
            count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count}")


def cmd_ingest_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = read_benchmark_csv(args.csv)
    count = insert_benchmark(cfg.database.path, rows)
    print(f"Loaded {count} benchmark rows.")


def cmd_ingest_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = read_news_jsonl(args.jsonl)
    count = insert_news(cfg.database.path, rows)
    print(f"Loaded {count} news rows.")


def cmd_ingest_yfinance_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = fetch_yahoo_finance_news(args.tickers, limit_per_ticker=args.limit_per_ticker)
    count = insert_news(cfg.database.path, rows)
    print(f"Loaded {count} Yahoo Finance news rows.")


def cmd_ingest_sec_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    forms = set(args.forms) if args.forms else {"8-K", "10-Q", "10-K"}
    rows = fetch_sec_filings_news(
        args.tickers,
        user_agent=cfg.sources.sec_user_agent,
        forms=forms,
        since=args.since,
        limit_per_ticker=args.limit_per_ticker,
    )
    count = insert_news(cfg.database.path, rows)
    print(f"Loaded {count} SEC EDGAR news rows.")


def cmd_ingest_builtin_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    yahoo_rows = fetch_yahoo_finance_news(args.tickers, limit_per_ticker=args.yahoo_limit_per_ticker)
    sec_rows = fetch_sec_filings_news(
        args.tickers,
        user_agent=cfg.sources.sec_user_agent,
        forms=set(args.sec_forms),
        since=args.sec_since,
        limit_per_ticker=args.sec_limit_per_ticker,
    )
    count = insert_news(cfg.database.path, [*yahoo_rows, *sec_rows])
    print(
        f"Loaded {count} built-in news rows "
        f"({len(yahoo_rows)} Yahoo Finance, {len(sec_rows)} SEC EDGAR)."
    )


def cmd_ingest_sp500_builtin_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    as_of = args.as_of or date.today()
    sec_since = args.sec_since or (date.today() - timedelta(days=args.sec_days))
    universe_rows = fetch_current_sp500_constituents(as_of=as_of)
    insert_constituents(cfg.database.path, universe_rows)
    tickers = [row.ticker for row in universe_rows]

    yahoo_rows = []
    sec_rows = []
    if not args.skip_yahoo:
        yahoo_tickers = tickers if args.force else filter_tickers_missing_news(
            cfg.database.path,
            tickers,
            "yahoo",
            args.yahoo_limit_per_ticker,
        )
        yahoo_rows = fetch_yahoo_finance_news(yahoo_tickers, limit_per_ticker=args.yahoo_limit_per_ticker)
    if not args.skip_sec:
        sec_tickers = tickers if args.force else filter_tickers_missing_news(
            cfg.database.path,
            tickers,
            "sec",
            args.sec_limit_per_ticker,
        )
        sec_rows = fetch_sec_filings_news(
            sec_tickers,
            user_agent=cfg.sources.sec_user_agent,
            forms=set(args.sec_forms),
            since=sec_since,
            limit_per_ticker=args.sec_limit_per_ticker,
        )
    count = insert_news(cfg.database.path, [*yahoo_rows, *sec_rows])
    print(
        f"Loaded {len(tickers)} S&P 500 tickers and {count} built-in news rows "
        f"({len(yahoo_rows)} Yahoo Finance, {len(sec_rows)} SEC EDGAR)."
    )


def cmd_build_clean_views(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    build_clean_views(cfg.database.path)
    print("Rebuilt clean views.")


def cmd_score_news(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = score_news_batch(
        str(cfg.database.path),
        cfg.llm,
        limit=args.limit,
        retry_failed=args.retry_failed,
    )
    print(
        "News scoring complete: "
        f"pending={result['pending']} success={result['success']} failed={result['failed']}"
    )


def cmd_aggregate_news_factors(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    count = aggregate_news_factors(str(cfg.database.path))
    print(f"Aggregated {count} news factor rows.")


def cmd_run_news_factor_pipeline(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    before = pending_news_count(str(cfg.database.path), cfg.llm.prompt_version)
    result = score_news_batch(
        str(cfg.database.path),
        cfg.llm,
        limit=args.limit,
        retry_failed=args.retry_failed,
    )
    count = aggregate_news_factors(str(cfg.database.path))
    after = pending_news_count(str(cfg.database.path), cfg.llm.prompt_version)
    print(
        "News factor pipeline complete: "
        f"pending_before={before} scored_success={result['success']} "
        f"scored_failed={result['failed']} pending_after={after} factor_rows={count}"
    )


def cmd_build_factor_table(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    try:
        result = build_factor_table(cfg.database.path, require_pit_universe=args.require_pit_universe)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        f"Built factor_table rows={result.rows} "
        f"date_range={result.start_date}..{result.end_date}"
    )
    print(f"Universe mode: {result.universe_mode}")
    if result.universe_coverage:
        print(
            "Universe coverage: "
            f"rows={result.universe_coverage.get('total_rows')} "
            f"tickers={result.universe_coverage.get('distinct_tickers')} "
            f"snapshots={result.universe_coverage.get('snapshot_dates')} "
            f"entry_rows={result.universe_coverage.get('rows_with_entry_date')} "
            f"exit_rows={result.universe_coverage.get('rows_with_exit_date')} "
            f"serious_pit={result.universe_coverage.get('has_serious_pit_history')}"
        )
    if result.low_coverage:
        print("Low coverage factors (<80% non-null):")
        for factor, coverage in result.low_coverage.items():
            print(f"  {factor}: {coverage:.1%}")
    if result.high_correlation_pairs:
        print("Highly correlated factor pairs (|corr| > 0.95):")
        for left, right, corr in result.high_correlation_pairs:
            print(f"  {left} vs {right}: {corr:.3f}")


def cmd_check_universe(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    coverage = universe_coverage(cfg.database.path)
    data = coverage.to_dict()
    status = "point_in_time_ready" if coverage.has_serious_pit_history else "current_or_partial_universe"
    print(f"Universe status: {status}")
    print(f"Rows: {data['total_rows']}")
    print(f"Tickers: {data['distinct_tickers']}")
    print(f"Snapshot dates: {data['snapshot_dates']}")
    print(f"Entry-date rows: {data['rows_with_entry_date']}")
    print(f"Exit-date rows: {data['rows_with_exit_date']}")
    print(f"Invalid membership rows: {data['invalid_membership_rows']}")
    print(f"Active members at price start: {data['active_at_price_start']}")
    print(f"Active members at price end: {data['active_at_price_end']}")
    print(f"Dated ratio: {data['dated_ratio']:.1%}")
    print(f"Membership date range: {data['min_entry_date']}..{data['max_exit_date']}")
    print(f"Price date range: {data['price_start_date']}..{data['price_end_date']}")
    print(f"Serious PIT history: {data['has_serious_pit_history']}")
    if not coverage.has_serious_pit_history:
        for reason in data.get("rejection_reasons", []):
            print(f"Rejected because: {reason}")
        print(
            "Interpretation: current-only or partial constituent data; historical backtests still carry "
            "survivorship-bias risk."
        )


def cmd_train_models(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_root = args.artifact_dir or Path(args.config).parent / "models"
    result = train_prediction_models(cfg.database.path, artifact_root, selected_models=args.models)
    print(
        f"Trained model_version={result.model_version} "
        f"dataset_rows={result.dataset_rows} train={result.train_rows} "
        f"validation={result.validation_rows} test={result.test_rows} "
        f"predictions={result.prediction_rows}"
    )
    print(f"Artifacts: {result.artifact_dir}")
    split_summary = result.metrics.get("split_summary", {})
    final_split = split_summary.get("final_holdout", {}) if split_summary else {}
    if final_split:
        print(
            "Strict OOS final holdout: "
            f"rows={final_split.get('rows')} "
            f"dates={final_split.get('start')}..{final_split.get('end')} "
            f"embargo_days={split_summary.get('embargo_trading_days')}"
        )
    final_metrics = result.metrics.get("final_holdout_ensemble", {})
    if final_metrics:
        print(
            "Final holdout ensemble metrics: "
            f"IC={final_metrics.get('ic')} "
            f"RankIC={final_metrics.get('rank_ic')} "
            f"RMSE={final_metrics.get('rmse')} "
            f"HitRate={final_metrics.get('hit_rate')}"
        )
    test_metrics = result.metrics.get("test_ensemble", {})
    if test_metrics:
        print(
            "Development test ensemble metrics: "
            f"IC={test_metrics.get('ic')} "
            f"RankIC={test_metrics.get('rank_ic')} "
            f"RMSE={test_metrics.get('rmse')} "
            f"HitRate={test_metrics.get('hit_rate')}"
        )
    returns = result.metrics.get("final_holdout_return_proxy") or result.metrics.get("test_return_proxy", {})
    if returns:
        print(
            "Primary OOS return proxy: "
            f"LongShort={returns.get('long_short_total_return')} "
            f"LongOnly={returns.get('long_only_total_return')} "
            f"SP500={returns.get('sp500_total_return')} "
            f"LongMinusSP500={returns.get('long_minus_sp500_total_return')} "
            f"ShortOnly={returns.get('short_only_total_return')}"
        )
    weights = result.metrics.get("ensemble_weights", {})
    if weights:
        print(f"Validation RankIC ensemble weights: {weights}")
    artifacts = result.metrics.get("artifacts", {})
    if artifacts:
        print(f"Out-of-sample plot: {artifacts.get('out_of_sample_plot')}")
        print(f"Stock selection guide: {artifacts.get('stock_selection_guide_markdown')}")
        print(f"Factor IC report: {artifacts.get('factor_ic_report')}")
        print(f"Feature importance: {artifacts.get('feature_importance')}")
        print(f"OOS split report: {artifacts.get('oos_split_report')}")


def cmd_train_walk_forward_models(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_root = args.artifact_dir or Path(args.config).parent / "models"
    result = train_walk_forward_models(
        cfg.database.path,
        artifact_root,
        selected_models=args.models,
        min_train_days=args.min_train_days,
        step_days=args.step_days,
        final_holdout_pct=args.final_holdout_pct,
    )
    print(
        f"Walk-forward trained model_version={result.model_version} "
        f"dataset_rows={result.dataset_rows} train={result.train_rows} "
        f"validation={result.validation_rows} final_holdout={result.test_rows} "
        f"predictions={result.prediction_rows}"
    )
    print(f"Artifacts: {result.artifact_dir}")
    metrics = result.metrics.get("final_holdout_ensemble") or result.metrics.get("walk_forward_ensemble", {})
    print(
        "Walk-forward metrics: "
        f"IC={metrics.get('ic')} "
        f"RankIC={metrics.get('rank_ic')} "
        f"RMSE={metrics.get('rmse')} "
        f"HitRate={metrics.get('hit_rate')}"
    )


def cmd_run_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_root = args.artifact_dir or Path(args.config).parent / "models"
    result = run_portfolio_backtest(
        cfg.database.path,
        artifact_root,
        model_version=None if args.model_version == "latest" else args.model_version,
        top_pct=args.top_pct,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        rebalance_days=args.rebalance_days,
        transaction_bps=args.transaction_bps,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
        exclude_tickers=args.exclude_tickers,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        execution_price_model=args.execution_price_model,
    )
    print(
        f"Backtest complete model_version={result.model_version} "
        f"date_range={result.start_date}..{result.end_date} "
        f"signals={result.signal_rows} positions={result.position_rows} "
        f"trades={result.trade_rows} rows={result.result_rows}"
    )
    print(f"Artifacts: {result.artifact_dir}")
    print(
        "Backtest metrics: "
        f"Return={result.metrics.get('total_return')} "
        f"SP500={result.metrics.get('benchmark_return')} "
        f"Excess={result.metrics.get('excess_return')} "
        f"CAGR={result.metrics.get('cagr')} "
        f"Sharpe={result.metrics.get('sharpe')} "
        f"MaxDrawdown={result.metrics.get('max_drawdown')} "
        f"AvgTurnover={result.metrics.get('avg_turnover')}"
    )
    print(f"Backtest plot: {Path(result.artifact_dir) / 'backtest_report.png'}")


def cmd_optimize_portfolio(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_root = args.artifact_dir or Path(args.config).parent / "models"
    result = run_portfolio_optimization(
        cfg.database.path,
        artifact_root,
        model_version=None if args.model_version == "latest" else args.model_version,
        max_drawdown_limit=args.max_drawdown_limit,
        transaction_bps=args.transaction_bps,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
        exclude_tickers=args.exclude_tickers,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        execution_price_model=args.execution_price_model,
    )
    print(
        f"Optimization complete model_version={result.model_version} "
        f"candidates={result.candidate_count} feasible={result.feasible_count}"
    )
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Best params: {result.best_params}")
    print(
        "Best metrics: "
        f"Return={result.best_metrics.get('total_return')} "
        f"SP500={result.best_metrics.get('benchmark_return')} "
        f"Excess={result.best_metrics.get('excess_return')} "
        f"Sharpe={result.best_metrics.get('sharpe')} "
        f"MaxDrawdown={result.best_metrics.get('max_drawdown')} "
        f"AvgTurnover={result.best_metrics.get('avg_turnover')}"
    )
    print(f"Optimized weights: {Path(result.artifact_dir) / 'optimized_weights.csv'}")


def cmd_run_trust_audit(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    artifact_root = args.artifact_dir or Path(args.config).parent / "models"
    result = run_trust_audit(
        cfg.database.path,
        artifact_root,
        model_version=None if args.model_version == "latest" else args.model_version,
    )
    print(f"Trust audit complete model_version={result.model_version}")
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Report: {result.report_path}")
    print(f"Universe bias status: {result.summary.get('constituent_bias', {}).get('status')}")
    print(f"Portfolio beta: {result.summary.get('exposure', {}).get('portfolio_beta')}")


def cmd_backup_parquet(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    backup_to_parquet(cfg.database.path, cfg.database.parquet_backup_dir)
    print(f"Exported Parquet backups to {cfg.database.parquet_backup_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiqrs", description="AI Quant Research System Phase 1 CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-db", help="Create DuckDB schema and clean views.")
    add_config_arg(init)
    init.set_defaults(func=cmd_init_db)

    universe = sub.add_parser("ingest-universe", help="Load point-in-time S&P 500 constituents from CSV.")
    add_config_arg(universe)
    universe.add_argument("--csv", type=Path, required=True)
    universe.set_defaults(func=cmd_ingest_universe)

    sp500_current = sub.add_parser("ingest-sp500-current", help="Download current S&P 500 constituents from Wikipedia.")
    add_config_arg(sp500_current)
    sp500_current.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    sp500_current.set_defaults(func=cmd_ingest_sp500_current)

    prices = sub.add_parser("ingest-prices", help="Load daily OHLCV prices from CSV.")
    add_config_arg(prices)
    prices.add_argument("--csv", type=Path, required=True)
    prices.set_defaults(func=cmd_ingest_prices)

    yf_prices = sub.add_parser("ingest-yfinance-prices", help="Download daily adjusted OHLCV from Yahoo Finance.")
    add_config_arg(yf_prices)
    yf_prices.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT SPY.")
    yf_prices.add_argument("--start", required=True, type=date.fromisoformat)
    yf_prices.add_argument("--end", required=True, type=date.fromisoformat)
    yf_prices.add_argument("--force", action="store_true", help="Download even if local price coverage exists.")
    yf_prices.set_defaults(func=cmd_ingest_yfinance_prices)

    sp500_yf = sub.add_parser(
        "ingest-sp500-yfinance-prices",
        help="Download current S&P 500 constituents and recent adjusted OHLCV from Yahoo Finance.",
    )
    add_config_arg(sp500_yf)
    sp500_yf.add_argument("--days", type=int, default=45)
    sp500_yf.add_argument("--start", type=date.fromisoformat)
    sp500_yf.add_argument("--end", type=date.fromisoformat)
    sp500_yf.add_argument("--as-of", type=date.fromisoformat)
    sp500_yf.add_argument("--batch-size", type=int, default=25)
    sp500_yf.add_argument("--force", action="store_true", help="Download even if local price coverage exists.")
    sp500_yf.set_defaults(func=cmd_ingest_sp500_yfinance_prices)

    benchmark = sub.add_parser("ingest-benchmark", help="Load benchmark and macro data from CSV.")
    add_config_arg(benchmark)
    benchmark.add_argument("--csv", type=Path, required=True)
    benchmark.set_defaults(func=cmd_ingest_benchmark)

    news = sub.add_parser("ingest-news", help="Load raw news from JSONL.")
    add_config_arg(news)
    news.add_argument("--jsonl", type=Path, required=True)
    news.set_defaults(func=cmd_ingest_news)

    yf_news = sub.add_parser("ingest-yfinance-news", help="Download recent Yahoo Finance news for tickers.")
    add_config_arg(yf_news)
    yf_news.add_argument("--tickers", nargs="+", required=True)
    yf_news.add_argument("--limit-per-ticker", type=int, default=10)
    yf_news.set_defaults(func=cmd_ingest_yfinance_news)

    sec_news = sub.add_parser("ingest-sec-news", help="Download recent SEC EDGAR filing events for tickers.")
    add_config_arg(sec_news)
    sec_news.add_argument("--tickers", nargs="+", required=True)
    sec_news.add_argument("--forms", nargs="*", default=["8-K", "10-Q", "10-K"])
    sec_news.add_argument("--since", type=date.fromisoformat)
    sec_news.add_argument("--limit-per-ticker", type=int, default=20)
    sec_news.set_defaults(func=cmd_ingest_sec_news)

    builtin_news = sub.add_parser("ingest-news-builtins", help="Download Yahoo Finance news and SEC filings.")
    add_config_arg(builtin_news)
    builtin_news.add_argument("--tickers", nargs="+", required=True)
    builtin_news.add_argument("--yahoo-limit-per-ticker", type=int, default=10)
    builtin_news.add_argument("--sec-forms", nargs="*", default=["8-K", "10-Q", "10-K"])
    builtin_news.add_argument("--sec-since", type=date.fromisoformat)
    builtin_news.add_argument("--sec-limit-per-ticker", type=int, default=10)
    builtin_news.set_defaults(func=cmd_ingest_builtin_news)

    sp500_news = sub.add_parser(
        "ingest-sp500-news-builtins",
        help="Download Yahoo Finance news and SEC filings for current S&P 500 tickers.",
    )
    add_config_arg(sp500_news)
    sp500_news.add_argument("--as-of", type=date.fromisoformat)
    sp500_news.add_argument("--yahoo-limit-per-ticker", type=int, default=3)
    sp500_news.add_argument("--sec-forms", nargs="*", default=["8-K", "10-Q", "10-K"])
    sp500_news.add_argument("--sec-since", type=date.fromisoformat)
    sp500_news.add_argument("--sec-days", type=int, default=90)
    sp500_news.add_argument("--sec-limit-per-ticker", type=int, default=2)
    sp500_news.add_argument("--skip-yahoo", action="store_true")
    sp500_news.add_argument("--skip-sec", action="store_true")
    sp500_news.add_argument("--force", action="store_true", help="Fetch news even if local rows already exist.")
    sp500_news.set_defaults(func=cmd_ingest_sp500_builtin_news)

    views = sub.add_parser("build-clean-views", help="Recreate daily_prices_clean and clean_news.")
    add_config_arg(views)
    views.set_defaults(func=cmd_build_clean_views)

    score_news = sub.add_parser("score-news", help="Score pending clean_news rows into news_scores.")
    add_config_arg(score_news)
    score_news.add_argument("--limit", type=int, default=100)
    score_news.add_argument("--retry-failed", action="store_true")
    score_news.set_defaults(func=cmd_score_news)

    aggregate_news = sub.add_parser("aggregate-news-factors", help="Aggregate news_scores into news_factors.")
    add_config_arg(aggregate_news)
    aggregate_news.set_defaults(func=cmd_aggregate_news_factors)

    news_pipeline = sub.add_parser("run-news-factor-pipeline", help="Score news and aggregate final daily news factors.")
    add_config_arg(news_pipeline)
    news_pipeline.add_argument("--limit", type=int, default=100)
    news_pipeline.add_argument("--retry-failed", action="store_true")
    news_pipeline.set_defaults(func=cmd_run_news_factor_pipeline)

    factor_table = sub.add_parser(
        "build-factor-table",
        help="Build Phase 3 factor_table from price, sector, benchmark, and news factors.",
    )
    add_config_arg(factor_table)
    factor_table.add_argument(
        "--require-pit-universe",
        action="store_true",
        help="Fail unless sp500_constituents contains a serious historical point-in-time universe.",
    )
    factor_table.set_defaults(func=cmd_build_factor_table)

    check_universe = sub.add_parser("check-universe", help="Report whether the S&P 500 universe looks point-in-time ready.")
    add_config_arg(check_universe)
    check_universe.set_defaults(func=cmd_check_universe)

    train_models = sub.add_parser(
        "train-models",
        help="Build Phase 4 labels, train walk-forward models, and write model_predictions.",
    )
    add_config_arg(train_models)
    train_models.add_argument("--artifact-dir", type=Path)
    train_models.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "rf"], choices=["lightgbm", "xgboost", "rf"])
    train_models.set_defaults(func=cmd_train_models)

    walk_forward = sub.add_parser(
        "train-walk-forward-models",
        help="Train rolling walk-forward models and reserve a final holdout window.",
    )
    add_config_arg(walk_forward)
    walk_forward.add_argument("--artifact-dir", type=Path)
    walk_forward.add_argument("--models", nargs="+", default=["rf"], choices=["lightgbm", "xgboost", "rf"])
    walk_forward.add_argument("--min-train-days", type=int, default=252)
    walk_forward.add_argument("--step-days", type=int, default=21)
    walk_forward.add_argument("--final-holdout-pct", type=float, default=0.20)
    walk_forward.set_defaults(func=cmd_train_walk_forward_models)

    backtest = sub.add_parser(
        "run-backtest",
        help="Convert model predictions into positions and run an out-of-sample portfolio backtest.",
    )
    add_config_arg(backtest)
    backtest.add_argument("--artifact-dir", type=Path)
    backtest.add_argument("--model-version", default="latest", help="Model version to backtest, or latest.")
    backtest.add_argument("--top-pct", type=float, default=0.10, help="Fraction of ranked stocks to buy.")
    backtest.add_argument("--max-weight", type=float, default=0.05, help="Maximum target weight per stock.")
    backtest.add_argument("--min-weight", type=float, default=0.005, help="Minimum target weight per stock after score weighting.")
    backtest.add_argument("--rebalance-days", type=int, default=5, help="Rebalance every N trading days.")
    backtest.add_argument("--transaction-bps", type=float, default=10.0, help="One-way transaction cost in bps.")
    backtest.add_argument("--slippage-bps", type=float, default=5.0, help="Estimated slippage in bps.")
    backtest.add_argument("--initial-capital", type=float, default=1_000_000.0)
    backtest.add_argument("--exclude-tickers", nargs="*", default=["SPY", "QQQ"], help="Tickers excluded from holdings.")
    backtest.add_argument("--min-avg-dollar-volume", type=float, default=0.0, help="Minimum 20d average dollar volume.")
    backtest.add_argument("--execution-price-model", choices=["open", "vwap", "close"], default="open")
    backtest.set_defaults(func=cmd_run_backtest)

    optimize = sub.add_parser(
        "optimize-portfolio",
        help="Grid-search portfolio settings to maximize Sharpe subject to a max drawdown limit.",
    )
    add_config_arg(optimize)
    optimize.add_argument("--artifact-dir", type=Path)
    optimize.add_argument("--model-version", default="latest", help="Model version to optimize, or latest.")
    optimize.add_argument("--max-drawdown-limit", type=float, default=0.10)
    optimize.add_argument("--transaction-bps", type=float, default=10.0)
    optimize.add_argument("--slippage-bps", type=float, default=5.0)
    optimize.add_argument("--initial-capital", type=float, default=1_000_000.0)
    optimize.add_argument("--exclude-tickers", nargs="*", default=["SPY", "QQQ"], help="Tickers excluded from holdings.")
    optimize.add_argument("--min-avg-dollar-volume", type=float, default=50_000_000.0, help="Minimum 20d average dollar volume.")
    optimize.add_argument("--execution-price-model", choices=["open", "vwap", "close"], default="open")
    optimize.set_defaults(func=cmd_optimize_portfolio)

    audit = sub.add_parser("run-trust-audit", help="Write survivorship, exposure, liquidity, sensitivity, and multi-window audit reports.")
    add_config_arg(audit)
    audit.add_argument("--artifact-dir", type=Path)
    audit.add_argument("--model-version", default="latest", help="Model version to audit, or latest.")
    audit.set_defaults(func=cmd_run_trust_audit)

    backup = sub.add_parser("backup-parquet", help="Export DuckDB tables to partitioned Parquet.")
    add_config_arg(backup)
    backup.set_defaults(func=cmd_backup_parquet)

    counts = sub.add_parser("count-rows", help="Print table row counts.")
    add_config_arg(counts)
    counts.set_defaults(func=cmd_count_rows)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
