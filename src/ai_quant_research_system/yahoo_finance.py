from __future__ import annotations

from datetime import date
from itertools import islice

from .records import DailyPriceRecord


def chunked(values: list[str], size: int) -> list[list[str]]:
    iterator = iter(values)
    chunks: list[list[str]] = []
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        chunks.append(chunk)
    return chunks


def is_valid_number(value: object) -> bool:
    try:
        return value == value
    except Exception:
        return False


def make_record(day, ticker: str, row) -> DailyPriceRecord:
    return DailyPriceRecord(
        date=day.date(),
        ticker=ticker.upper(),
        open=float(row["Open"]) if is_valid_number(row["Open"]) else None,
        high=float(row["High"]) if is_valid_number(row["High"]) else None,
        low=float(row["Low"]) if is_valid_number(row["Low"]) else None,
        close=float(row["Close"]) if is_valid_number(row["Close"]) else None,
        volume=int(row["Volume"]) if is_valid_number(row["Volume"]) else None,
        vwap=None,
        adj_factor=1.0,
        source="yfinance",
    )


def fetch_daily_prices(
    tickers: list[str],
    start: date,
    end: date,
    batch_size: int = 25,
) -> list[DailyPriceRecord]:
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install yfinance first: pip install -r requirements.txt") from exc

    normalized = [ticker.strip().upper().replace(".", "-") for ticker in tickers if ticker.strip()]
    records: list[DailyPriceRecord] = []
    batches = chunked(normalized, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        print(
            f"Downloading Yahoo Finance batch {batch_index}/{len(batches)} "
            f"({len(batch)} tickers): {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}",
            flush=True,
        )
        data = yf.download(
            batch if len(batch) > 1 else batch[0],
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=30,
        )
        if data.empty:
            print(f"Batch {batch_index}/{len(batches)} returned no rows.", flush=True)
            continue
        before = len(records)
        if len(batch) == 1:
            ticker = batch[0]
            for index, row in data.iterrows():
                records.append(make_record(index, ticker, row))
            print(f"Batch {batch_index}/{len(batches)} loaded {len(records) - before} rows.", flush=True)
            continue
        for ticker in batch:
            if ticker not in data.columns.get_level_values(0):
                continue
            ticker_frame = data[ticker].dropna(how="all")
            for index, row in ticker_frame.iterrows():
                records.append(make_record(index, ticker, row))
        print(f"Batch {batch_index}/{len(batches)} loaded {len(records) - before} rows.", flush=True)
    return records
