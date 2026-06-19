from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class ConstituentRecord:
    date: date
    ticker: str
    company: str
    entry_date: date | None = None
    exit_date: date | None = None
    gics_sector: str | None = None
    gics_industry: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class DailyPriceRecord:
    date: date
    ticker: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    vwap: float | None = None
    adj_factor: float | None = None
    source: str | None = None
    is_suspended: bool = False
    is_imputed: bool = False
    suspicious_return: bool = False


@dataclass(frozen=True)
class BenchmarkRecord:
    date: date
    spy_close: float | None = None
    qqq_close: float | None = None
    vix: float | None = None
    yield_2y: float | None = None
    yield_10y: float | None = None
    yield_spread: float | None = None


@dataclass(frozen=True)
class NewsRecord:
    news_id: str
    ticker: str
    title: str
    summary: str | None
    body: str | None
    source: str
    url: str | None
    published_at: datetime
    market_session: str
    effective_date: date
    is_sec_filing: bool = False
    content_hash: str | None = None
