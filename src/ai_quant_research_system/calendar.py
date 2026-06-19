from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def is_weekday(value: date) -> bool:
    return value.weekday() < 5


def next_trading_day(value: date) -> date:
    current = value + timedelta(days=1)
    while not is_weekday(current):
        current += timedelta(days=1)
    return current


def normalize_effective_date(published_at_utc: datetime) -> tuple[str, date]:
    if published_at_utc.tzinfo is None:
        published_at_utc = published_at_utc.replace(tzinfo=ZoneInfo("UTC"))
    published_et = published_at_utc.astimezone(ET)
    trade_date = published_et.date()

    if not is_weekday(trade_date):
        return "off_hours", next_trading_day(trade_date)

    market_open = datetime.combine(trade_date, time(9, 30), ET)
    market_close = datetime.combine(trade_date, time(16, 0), ET)
    if market_open <= published_et < market_close:
        return "regular", trade_date
    return "pre_or_post_market", next_trading_day(trade_date)
