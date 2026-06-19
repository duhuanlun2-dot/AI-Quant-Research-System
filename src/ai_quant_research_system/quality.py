from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date

from .records import DailyPriceRecord


def flag_suspicious_returns(
    rows: list[DailyPriceRecord],
    threshold: float = 0.50,
) -> list[DailyPriceRecord]:
    by_ticker: dict[str, list[DailyPriceRecord]] = defaultdict(list)
    for row in rows:
        by_ticker[row.ticker].append(row)

    result: list[DailyPriceRecord] = []
    for ticker_rows in by_ticker.values():
        previous_close: float | None = None
        for row in sorted(ticker_rows, key=lambda item: item.date):
            suspicious = row.suspicious_return
            if previous_close and row.close is not None and previous_close != 0:
                ret = row.close / previous_close - 1
                suspicious = suspicious or abs(ret) > threshold
            if row.close is not None:
                previous_close = row.close
            result.append(replace(row, suspicious_return=suspicious))
    return sorted(result, key=lambda item: (item.date, item.ticker))


def fill_suspended_days(
    rows: list[DailyPriceRecord],
    trading_days: list[date],
) -> list[DailyPriceRecord]:
    by_ticker: dict[str, dict[date, DailyPriceRecord]] = defaultdict(dict)
    for row in rows:
        by_ticker[row.ticker][row.date] = row

    filled: list[DailyPriceRecord] = []
    for ticker, ticker_rows in by_ticker.items():
        last_close: float | None = None
        for day in sorted(trading_days):
            row = ticker_rows.get(day)
            if row is None:
                if last_close is None:
                    continue
                filled.append(
                    DailyPriceRecord(
                        date=day,
                        ticker=ticker,
                        open=last_close,
                        high=last_close,
                        low=last_close,
                        close=last_close,
                        volume=0,
                        source="imputed_previous_close",
                        is_suspended=True,
                        is_imputed=True,
                    )
                )
                continue
            if row.close is not None:
                last_close = row.close
            filled.append(row)
    return sorted(filled, key=lambda item: (item.date, item.ticker))
