from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .calendar import normalize_effective_date
from .database import connect
from .quality import flag_suspicious_returns
from .records import BenchmarkRecord, ConstituentRecord, DailyPriceRecord, NewsRecord


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def as_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def as_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(float(value))


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_constituents_csv(path: str | Path) -> list[ConstituentRecord]:
    rows: list[ConstituentRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                ConstituentRecord(
                    date=parse_date(row["date"]),  # type: ignore[arg-type]
                    ticker=row["ticker"].upper().strip(),
                    company=row["company"].strip(),
                    entry_date=parse_date(row.get("entry_date")),
                    exit_date=parse_date(row.get("exit_date")),
                    gics_sector=row.get("gics_sector") or None,
                    gics_industry=row.get("gics_industry") or None,
                    source=row.get("source") or "csv",
                )
            )
    return rows


def read_prices_csv(path: str | Path, threshold: float = 0.50) -> list[DailyPriceRecord]:
    rows: list[DailyPriceRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                DailyPriceRecord(
                    date=parse_date(row["date"]),  # type: ignore[arg-type]
                    ticker=row["ticker"].upper().strip(),
                    open=as_float(row.get("open")),
                    high=as_float(row.get("high")),
                    low=as_float(row.get("low")),
                    close=as_float(row.get("close")),
                    volume=as_int(row.get("volume")),
                    vwap=as_float(row.get("vwap")),
                    adj_factor=as_float(row.get("adj_factor")),
                    source=row.get("source") or "csv",
                )
            )
    return flag_suspicious_returns(rows, threshold=threshold)


def read_benchmark_csv(path: str | Path) -> list[BenchmarkRecord]:
    rows: list[BenchmarkRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            y2 = as_float(row.get("yield_2y"))
            y10 = as_float(row.get("yield_10y"))
            spread = as_float(row.get("yield_spread"))
            if spread is None and y2 is not None and y10 is not None:
                spread = y10 - y2
            rows.append(
                BenchmarkRecord(
                    date=parse_date(row["date"]),  # type: ignore[arg-type]
                    spy_close=as_float(row.get("SPY") or row.get("spy_close")),
                    qqq_close=as_float(row.get("QQQ") or row.get("qqq_close")),
                    vix=as_float(row.get("VIX") or row.get("vix")),
                    yield_2y=y2,
                    yield_10y=y10,
                    yield_spread=spread,
                )
            )
    return rows


def read_news_jsonl(path: str | Path) -> list[NewsRecord]:
    rows: list[NewsRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            published_at = parse_datetime(raw["published_at"])
            market_session, effective_date = normalize_effective_date(published_at)
            body = raw.get("body")
            summary = raw.get("summary")
            fingerprint = f"{raw.get('title', '')}|{raw.get('source', '')}|{body or summary or ''}"
            content_hash = raw.get("content_hash") or hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
            rows.append(
                NewsRecord(
                    news_id=raw.get("news_id") or content_hash,
                    ticker=raw["ticker"].upper().strip(),
                    title=raw["title"].strip(),
                    summary=summary,
                    body=body,
                    source=raw["source"].strip(),
                    url=raw.get("url"),
                    published_at=published_at,
                    market_session=raw.get("market_session") or market_session,
                    effective_date=parse_date(raw.get("effective_date")) or effective_date,  # type: ignore[arg-type]
                    is_sec_filing=bool_value(raw.get("is_sec_filing")),
                    content_hash=content_hash,
                )
            )
    return rows


def insert_constituents(db_path: str | Path, rows: Iterable[ConstituentRecord]) -> int:
    data = list(rows)
    if not data:
        return 0
    with connect(db_path) as con:
        con.executemany(
            """
            INSERT OR REPLACE INTO sp500_constituents
            (date, ticker, company, entry_date, exit_date, gics_sector, gics_industry, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row.__dict__.values()) for row in data],
        )
    return len(data)


def insert_prices(db_path: str | Path, rows: Iterable[DailyPriceRecord]) -> int:
    data = list(rows)
    if not data:
        return 0
    with connect(db_path) as con:
        con.executemany(
            """
            INSERT OR REPLACE INTO daily_prices
            (date, ticker, open, high, low, close, volume, vwap, adj_factor, source,
             is_suspended, is_imputed, suspicious_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row.__dict__.values()) for row in data],
        )
    return len(data)


def insert_benchmark(db_path: str | Path, rows: Iterable[BenchmarkRecord]) -> int:
    data = list(rows)
    if not data:
        return 0
    with connect(db_path) as con:
        con.executemany(
            """
            INSERT OR REPLACE INTO benchmark_daily
            (date, spy_close, qqq_close, vix, yield_2y, yield_10y, yield_spread)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row.__dict__.values()) for row in data],
        )
    return len(data)


def insert_news(db_path: str | Path, rows: Iterable[NewsRecord]) -> int:
    data = list(rows)
    if not data:
        return 0
    with connect(db_path) as con:
        con.executemany(
            """
            INSERT OR REPLACE INTO raw_news
            (news_id, ticker, title, summary, body, source, url, published_at,
             market_session, effective_date, is_sec_filing, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row.__dict__.values()) for row in data],
        )
    return len(data)
