from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sp500_constituents (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    company VARCHAR NOT NULL,
    entry_date DATE,
    exit_date DATE,
    gics_sector VARCHAR,
    gics_industry VARCHAR,
    source VARCHAR,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    vwap DOUBLE,
    adj_factor DOUBLE,
    source VARCHAR,
    is_suspended BOOLEAN DEFAULT false,
    is_imputed BOOLEAN DEFAULT false,
    suspicious_return BOOLEAN DEFAULT false,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS benchmark_daily (
    date DATE PRIMARY KEY,
    spy_close DOUBLE,
    qqq_close DOUBLE,
    vix DOUBLE,
    yield_2y DOUBLE,
    yield_10y DOUBLE,
    yield_spread DOUBLE,
    ingested_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS raw_news (
    news_id VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    summary VARCHAR,
    body VARCHAR,
    source VARCHAR NOT NULL,
    url VARCHAR,
    published_at TIMESTAMP NOT NULL,
    market_session VARCHAR,
    effective_date DATE,
    is_sec_filing BOOLEAN DEFAULT false,
    content_hash VARCHAR,
    ingested_at TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_raw_news_ticker_effective_date
ON raw_news (ticker, effective_date);

CREATE TABLE IF NOT EXISTS ingestion_audit (
    run_id VARCHAR,
    dataset VARCHAR,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR,
    rows_written BIGINT,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS news_scores (
    news_id VARCHAR PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    effective_date DATE NOT NULL,
    sentiment DOUBLE,
    importance DOUBLE,
    surprise DOUBLE,
    risk DOUBLE,
    revenue_impact DOUBLE,
    margin_impact DOUBLE,
    confidence DOUBLE,
    score_status VARCHAR NOT NULL DEFAULT 'pending',
    prompt_version VARCHAR NOT NULL,
    error_message VARCHAR,
    scored_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_scores_ticker_date
ON news_scores (ticker, effective_date);

CREATE TABLE IF NOT EXISTS news_factors_raw (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    news_sentiment DOUBLE,
    news_importance DOUBLE,
    news_surprise DOUBLE,
    news_risk DOUBLE,
    news_revenue_impact DOUBLE,
    news_margin_impact DOUBLE,
    news_coverage BIGINT,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS news_factors (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    news_sentiment DOUBLE,
    news_importance DOUBLE,
    news_surprise DOUBLE,
    news_risk DOUBLE,
    news_revenue_impact DOUBLE,
    news_margin_impact DOUBLE,
    news_coverage BIGINT,
    news_ai_score DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS factor_table (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    gics_sector VARCHAR,
    momentum_5d DOUBLE,
    momentum_20d DOUBLE,
    momentum_60d DOUBLE,
    volatility_20d DOUBLE,
    volume_change_5d DOUBLE,
    rsi_14 DOUBLE,
    market_return_5d DOUBLE,
    sector_return_5d DOUBLE,
    beta DOUBLE,
    news_sentiment DOUBLE,
    news_importance DOUBLE,
    news_surprise DOUBLE,
    news_risk DOUBLE,
    news_revenue_impact DOUBLE,
    news_margin_impact DOUBLE,
    news_coverage DOUBLE,
    news_ai_score DOUBLE,
    z_momentum_5d DOUBLE,
    z_momentum_20d DOUBLE,
    z_momentum_60d DOUBLE,
    z_volatility_20d DOUBLE,
    z_volume_change_5d DOUBLE,
    z_rsi_14 DOUBLE,
    z_market_return_5d DOUBLE,
    z_sector_return_5d DOUBLE,
    z_beta DOUBLE,
    z_news_sentiment DOUBLE,
    z_news_importance DOUBLE,
    z_news_surprise DOUBLE,
    z_news_risk DOUBLE,
    z_news_revenue_impact DOUBLE,
    z_news_margin_impact DOUBLE,
    z_news_coverage DOUBLE,
    z_news_ai_score DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    predicted_excess_return_5d DOUBLE,
    model_version VARCHAR NOT NULL,
    lightgbm_pred DOUBLE,
    xgboost_pred DOUBLE,
    rf_pred DOUBLE,
    predicted_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker, model_version)
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version VARCHAR PRIMARY KEY,
    trained_at TIMESTAMP DEFAULT current_timestamp,
    train_start DATE,
    train_end DATE,
    validation_start DATE,
    validation_end DATE,
    test_start DATE,
    test_end DATE,
    feature_columns VARCHAR,
    model_params_json VARCHAR,
    metrics_json VARCHAR,
    artifact_dir VARCHAR,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS trading_signals (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    predicted_excess_return_5d DOUBLE,
    prediction_rank BIGINT,
    signal VARCHAR NOT NULL,
    target_weight DOUBLE,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker, model_version)
);

CREATE TABLE IF NOT EXISTS portfolio_positions (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    target_weight DOUBLE NOT NULL,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, ticker, model_version)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    date DATE NOT NULL,
    portfolio_value DOUBLE,
    strategy_return DOUBLE,
    benchmark_value DOUBLE,
    benchmark_return DOUBLE,
    drawdown DOUBLE,
    gross_exposure DOUBLE,
    net_exposure DOUBLE,
    turnover DOUBLE,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (date, model_version)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    from_weight DOUBLE,
    to_weight DOUBLE,
    trade_weight DOUBLE,
    trade_value DOUBLE,
    transaction_cost DOUBLE,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS execution_orders (
    order_id VARCHAR PRIMARY KEY,
    signal_date DATE NOT NULL,
    order_date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    target_weight DOUBLE,
    current_weight DOUBLE,
    order_weight DOUBLE,
    estimated_order_value DOUBLE,
    model_version VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'created',
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS execution_fills (
    fill_id VARCHAR PRIMARY KEY,
    order_id VARCHAR NOT NULL,
    fill_date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    fill_price DOUBLE,
    fill_shares DOUBLE,
    fill_value DOUBLE,
    transaction_cost DOUBLE,
    execution_price_model VARCHAR,
    model_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS current_positions (
    as_of_date DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    shares DOUBLE,
    market_price DOUBLE,
    market_value DOUBLE,
    target_weight DOUBLE,
    current_weight DOUBLE,
    unrealized_pnl DOUBLE,
    model_version VARCHAR NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (as_of_date, ticker, model_version)
);
"""


CLEAN_VIEWS_SQL = """
CREATE OR REPLACE VIEW daily_prices_clean AS
SELECT
    date,
    ticker,
    open,
    high,
    low,
    close,
    volume,
    vwap,
    adj_factor,
    source,
    is_suspended,
    is_imputed,
    suspicious_return
FROM daily_prices;

CREATE OR REPLACE VIEW clean_news AS
SELECT * EXCLUDE(row_num)
FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY ticker, effective_date, coalesce(content_hash, title || source)
            ORDER BY published_at DESC
        ) AS row_num
    FROM raw_news
    WHERE length(coalesce(body, summary, '')) >= 50
      AND effective_date IS NOT NULL
) t
WHERE row_num = 1;
"""


def import_duckdb():
    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'duckdb' package is required. Install it with: pip install -r requirements.txt"
        ) from exc
    return duckdb


@contextmanager
def connect(db_path: str | Path) -> Iterator[object]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb = import_duckdb()
    con = duckdb.connect(str(path))
    try:
        yield con
    finally:
        con.close()


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as con:
        con.execute(SCHEMA_SQL)
        con.execute(CLEAN_VIEWS_SQL)


def build_clean_views(db_path: str | Path) -> None:
    with connect(db_path) as con:
        con.execute(CLEAN_VIEWS_SQL)


def backup_to_parquet(db_path: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as con:
        for table in ("sp500_constituents", "daily_prices", "benchmark_daily", "raw_news"):
            target = output / table
            target.mkdir(parents=True, exist_ok=True)
            con.execute(
                f"COPY {table} TO ? (FORMAT PARQUET, PARTITION_BY (date), OVERWRITE_OR_IGNORE true)",
                [str(target)],
            )
