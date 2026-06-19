from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import pandas as pd

from .database import connect


TRADITIONAL_FACTORS = [
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "volume_change_5d",
    "rsi_14",
    "market_return_5d",
    "sector_return_5d",
    "beta",
]

NEWS_FACTORS = [
    "news_sentiment",
    "news_importance",
    "news_surprise",
    "news_risk",
    "news_revenue_impact",
    "news_margin_impact",
    "news_coverage",
    "news_ai_score",
]

ALL_FACTORS = [*TRADITIONAL_FACTORS, *NEWS_FACTORS]


@dataclass(frozen=True)
class FactorBuildResult:
    rows: int
    start_date: str | None
    end_date: str | None
    low_coverage: dict[str, float]
    high_correlation_pairs: list[tuple[str, str, float]]
    universe_mode: str = "latest_constituents"
    universe_coverage: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UniverseCoverage:
    total_rows: int
    distinct_tickers: int
    snapshot_dates: int
    rows_with_entry_date: int
    rows_with_exit_date: int
    invalid_membership_rows: int
    active_at_price_start: int
    active_at_price_end: int
    min_entry_date: str | None
    max_exit_date: str | None
    price_start_date: str | None
    price_end_date: str | None

    @property
    def dated_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return (self.rows_with_entry_date + self.rows_with_exit_date) / self.total_rows

    @property
    def has_serious_pit_history(self) -> bool:
        if self.total_rows == 0:
            return False
        has_membership_dates = self.dated_ratio >= 0.80
        has_removed_members = self.rows_with_exit_date > 0
        has_valid_spans = self.invalid_membership_rows == 0
        has_plausible_active_counts = (
            self.active_at_price_start == 0
            or 350 <= self.active_at_price_start <= 650
        ) and (
            self.active_at_price_end == 0
            or 350 <= self.active_at_price_end <= 650
        )
        has_history_beyond_current_list = self.distinct_tickers > 520 or self.rows_with_exit_date >= 10
        return has_membership_dates and has_removed_members and has_valid_spans and has_plausible_active_counts and has_history_beyond_current_list

    @property
    def rejection_reasons(self) -> list[str]:
        reasons = []
        if self.total_rows == 0:
            reasons.append("no constituent rows")
        if self.dated_ratio < 0.80:
            reasons.append("less than 80% of rows have entry_date or exit_date")
        if self.rows_with_exit_date == 0:
            reasons.append("no removed members with exit_date")
        if self.invalid_membership_rows > 0:
            reasons.append("some rows have entry_date after exit_date")
        if self.distinct_tickers <= 520 and self.rows_with_exit_date < 10:
            reasons.append("does not contain enough removed historical tickers")
        if self.active_at_price_start and not 350 <= self.active_at_price_start <= 650:
            reasons.append("active member count at price start is implausible")
        if self.active_at_price_end and not 350 <= self.active_at_price_end <= 650:
            reasons.append("active member count at price end is implausible")
        return reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "distinct_tickers": self.distinct_tickers,
            "snapshot_dates": self.snapshot_dates,
            "rows_with_entry_date": self.rows_with_entry_date,
            "rows_with_exit_date": self.rows_with_exit_date,
            "invalid_membership_rows": self.invalid_membership_rows,
            "active_at_price_start": self.active_at_price_start,
            "active_at_price_end": self.active_at_price_end,
            "dated_ratio": self.dated_ratio,
            "min_entry_date": self.min_entry_date,
            "max_exit_date": self.max_exit_date,
            "price_start_date": self.price_start_date,
            "price_end_date": self.price_end_date,
            "has_serious_pit_history": self.has_serious_pit_history,
            "rejection_reasons": self.rejection_reasons,
        }


def universe_coverage(db_path: str | Path) -> UniverseCoverage:
    with connect(db_path) as con:
        row = con.execute(
            """
            SELECT
                count(*) AS total_rows,
                count(DISTINCT ticker) AS distinct_tickers,
                count(DISTINCT date) AS snapshot_dates,
                sum(CASE WHEN entry_date IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_entry_date,
                sum(CASE WHEN exit_date IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_exit_date,
                sum(CASE WHEN entry_date IS NOT NULL AND exit_date IS NOT NULL AND entry_date > exit_date THEN 1 ELSE 0 END)
                    AS invalid_membership_rows,
                min(entry_date) AS min_entry_date,
                max(exit_date) AS max_exit_date
            FROM sp500_constituents
            """
        ).fetchone()
        price_row = con.execute(
            """
            SELECT min(date), max(date)
            FROM daily_prices_clean
            WHERE close IS NOT NULL
              AND coalesce(suspicious_return, false) = false
            """
        ).fetchone()
        active_counts = con.execute(
            """
            WITH bounds AS (
                SELECT min(date) AS start_date, max(date) AS end_date
                FROM daily_prices_clean
                WHERE close IS NOT NULL
                  AND coalesce(suspicious_return, false) = false
            )
            SELECT
                count(DISTINCT CASE
                    WHEN b.start_date IS NOT NULL
                     AND b.start_date >= coalesce(c.entry_date, c.date)
                     AND b.start_date <= coalesce(c.exit_date, DATE '9999-12-31')
                    THEN c.ticker END) AS active_at_start,
                count(DISTINCT CASE
                    WHEN b.end_date IS NOT NULL
                     AND b.end_date >= coalesce(c.entry_date, c.date)
                     AND b.end_date <= coalesce(c.exit_date, DATE '9999-12-31')
                    THEN c.ticker END) AS active_at_end
            FROM sp500_constituents c
            CROSS JOIN bounds b
            """
        ).fetchone()
    return UniverseCoverage(
        total_rows=int(row[0] or 0),
        distinct_tickers=int(row[1] or 0),
        snapshot_dates=int(row[2] or 0),
        rows_with_entry_date=int(row[3] or 0),
        rows_with_exit_date=int(row[4] or 0),
        invalid_membership_rows=int(row[5] or 0),
        active_at_price_start=int(active_counts[0] or 0),
        active_at_price_end=int(active_counts[1] or 0),
        min_entry_date=None if row[6] is None else str(row[6]),
        max_exit_date=None if row[7] is None else str(row[7]),
        price_start_date=None if price_row[0] is None else str(price_row[0]),
        price_end_date=None if price_row[1] is None else str(price_row[1]),
    )


def latest_constituent_panel_sql() -> str:
    return """
    WITH latest_constituents AS (
        SELECT ticker, company, gics_sector, gics_industry
        FROM (
            SELECT *,
                   row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM sp500_constituents
        )
        WHERE rn = 1
    )
    SELECT
        p.date,
        p.ticker,
        p.open,
        p.high,
        p.low,
        p.close,
        p.volume,
        p.suspicious_return,
        c.gics_sector
    FROM daily_prices_clean p
    LEFT JOIN latest_constituents c
      ON p.ticker = c.ticker
    WHERE p.close IS NOT NULL
      AND coalesce(p.suspicious_return, false) = false
    ORDER BY p.ticker, p.date
    """


def point_in_time_panel_sql() -> str:
    return """
    WITH point_in_time_constituents AS (
        SELECT
            p.date,
            p.ticker,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume,
            p.suspicious_return,
            c.gics_sector,
            row_number() OVER (
                PARTITION BY p.date, p.ticker
                ORDER BY c.date DESC, c.entry_date DESC NULLS LAST
            ) AS rn
        FROM daily_prices_clean p
        JOIN sp500_constituents c
          ON p.ticker = c.ticker
         AND p.date >= coalesce(c.entry_date, c.date)
         AND p.date <= coalesce(c.exit_date, DATE '9999-12-31')
        WHERE p.close IS NOT NULL
          AND coalesce(p.suspicious_return, false) = false
    )
    SELECT
        date,
        ticker,
        open,
        high,
        low,
        close,
        volume,
        suspicious_return,
        gics_sector
    FROM point_in_time_constituents
    WHERE rn = 1
    ORDER BY ticker, date
    """


def load_price_panel(
    db_path: str | Path,
    require_pit_universe: bool = False,
) -> tuple[pd.DataFrame, str, dict[str, object]]:
    coverage = universe_coverage(db_path)
    if coverage.has_serious_pit_history:
        sql = point_in_time_panel_sql()
        mode = "point_in_time_constituents"
    elif require_pit_universe:
        raise RuntimeError(
            "Point-in-time universe is required, but sp500_constituents does not look complete enough. "
            "Load a historical membership CSV with entry_date/exit_date and removed tickers first."
        )
    else:
        sql = latest_constituent_panel_sql()
        mode = "latest_constituents_fallback"
    with connect(db_path) as con:
        return con.execute(sql).fetchdf(), mode, coverage.to_dict()


def load_news_factors(db_path: str | Path) -> pd.DataFrame:
    with connect(db_path) as con:
        return con.execute("SELECT * FROM news_factors").fetchdf()


def load_benchmark(db_path: str | Path) -> pd.DataFrame:
    with connect(db_path) as con:
        return con.execute("SELECT date, spy_close FROM benchmark_daily WHERE spy_close IS NOT NULL").fetchdf()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_beta(group: pd.DataFrame) -> pd.Series:
    stock = group["daily_return"]
    market = group["market_daily_return"]
    cov = stock.rolling(window=60, min_periods=60).cov(market)
    var = market.rolling(window=60, min_periods=60).var()
    return cov / var.replace(0, pd.NA)


def add_traditional_factors(panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    by_ticker = df.groupby("ticker", group_keys=False)

    df["daily_return"] = by_ticker["close"].pct_change()
    df["momentum_5d"] = by_ticker["close"].pct_change(periods=5)
    df["momentum_20d"] = by_ticker["close"].pct_change(periods=20)
    df["momentum_60d"] = by_ticker["close"].pct_change(periods=60)
    df["volatility_20d"] = by_ticker["daily_return"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std() * sqrt(252)
    )

    avg_volume_5d = by_ticker["volume"].transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    previous_avg_volume_5d = by_ticker["volume"].transform(
        lambda s: s.shift(5).rolling(window=5, min_periods=5).mean()
    )
    df["volume_change_5d"] = avg_volume_5d / previous_avg_volume_5d.replace(0, pd.NA) - 1
    df["rsi_14"] = by_ticker["close"].transform(compute_rsi)

    spy = df[df["ticker"] == "SPY"][["date", "close"]].drop_duplicates("date").rename(columns={"close": "spy_close"})
    if spy.empty and not benchmark.empty:
        spy = benchmark.copy()
        spy["date"] = pd.to_datetime(spy["date"])

    if not spy.empty:
        spy = spy.sort_values("date")
        spy["market_return_5d"] = spy["spy_close"].pct_change(periods=5)
        spy["market_daily_return"] = spy["spy_close"].pct_change()
        df = df.merge(spy[["date", "market_return_5d", "market_daily_return"]], on="date", how="left")
    else:
        df["market_return_5d"] = pd.NA
        df["market_daily_return"] = pd.NA

    df["sector_return_5d"] = df.groupby(["date", "gics_sector"])["momentum_5d"].transform("mean")
    df["beta"] = df.groupby("ticker", group_keys=False).apply(compute_beta, include_groups=False)
    return df


def merge_news_factors(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    if news.empty:
        for col in NEWS_FACTORS:
            merged[col] = 0.0
        return merged

    news_df = news.copy()
    news_df["date"] = pd.to_datetime(news_df["date"])
    cols = ["date", "ticker", *NEWS_FACTORS]
    merged = merged.merge(news_df[cols], on=["date", "ticker"], how="left")
    for col in NEWS_FACTORS:
        merged[col] = merged[col].fillna(0.0)
    return merged


def winsorized_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(0.0, index=values.index)
    low = numeric.quantile(0.01)
    high = numeric.quantile(0.99)
    clipped = numeric.clip(lower=low, upper=high)
    std = clipped.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (clipped - clipped.mean()) / std


def add_standardized_factors(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for factor in ALL_FACTORS:
        result[f"z_{factor}"] = result.groupby("date")[factor].transform(winsorized_zscore)
    return result


def coverage_report(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {factor: 0.0 for factor in ALL_FACTORS}
    coverage = df[ALL_FACTORS].notna().mean()
    return {factor: float(value) for factor, value in coverage.items() if value < 0.80}


def high_correlation_report(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    usable = df[ALL_FACTORS].dropna(how="all")
    if len(usable) < 3:
        return []
    corr = usable.corr(numeric_only=True)
    available = [factor for factor in ALL_FACTORS if factor in corr.index and factor in corr.columns]
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(available):
        for right in available[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) > 0.95:
                pairs.append((left, right, float(value)))
    return pairs


def write_factor_table(db_path: str | Path, df: pd.DataFrame) -> None:
    output_cols = [
        "date",
        "ticker",
        "gics_sector",
        *ALL_FACTORS,
        *[f"z_{factor}" for factor in ALL_FACTORS],
    ]
    output = df[output_cols].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.date
    with connect(db_path) as con:
        con.register("factor_df", output)
        con.execute("DELETE FROM factor_table")
        con.execute(
            """
            INSERT INTO factor_table
            SELECT *, current_timestamp AS computed_at
            FROM factor_df
            """
        )
        con.unregister("factor_df")


def build_factor_table(db_path: str | Path, require_pit_universe: bool = False) -> FactorBuildResult:
    panel, universe_mode, coverage = load_price_panel(db_path, require_pit_universe=require_pit_universe)
    benchmark = load_benchmark(db_path)
    news = load_news_factors(db_path)
    factors = add_traditional_factors(panel, benchmark)
    factors = merge_news_factors(factors, news)
    factors = add_standardized_factors(factors)

    low_coverage = coverage_report(factors)
    high_corr = high_correlation_report(factors)
    write_factor_table(db_path, factors)

    if factors.empty:
        return FactorBuildResult(0, None, None, low_coverage, high_corr, universe_mode, coverage)
    return FactorBuildResult(
        rows=len(factors),
        start_date=str(pd.to_datetime(factors["date"]).min().date()),
        end_date=str(pd.to_datetime(factors["date"]).max().date()),
        low_coverage=low_coverage,
        high_correlation_pairs=high_corr,
        universe_mode=universe_mode,
        universe_coverage=coverage,
    )
