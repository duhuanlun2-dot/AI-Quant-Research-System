from __future__ import annotations

from datetime import date
from io import StringIO
from urllib.request import Request, urlopen

from .records import ConstituentRecord


WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").strip().upper()


def fetch_current_sp500_constituents(as_of: date | None = None) -> list[ConstituentRecord]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pandas/lxml first: pip install -r requirements.txt") from exc

    effective_date = as_of or date.today()
    request = Request(WIKIPEDIA_SP500_URL, headers={"User-Agent": "ai-quant-research-system/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - public constituent source.
        html = response.read().decode("utf-8")

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("Could not find S&P 500 constituent table on Wikipedia.")

    frame = tables[0]
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Wikipedia S&P 500 table is missing columns: {sorted(missing)}")

    rows: list[ConstituentRecord] = []
    for _, item in frame.iterrows():
        rows.append(
            ConstituentRecord(
                date=effective_date,
                ticker=yahoo_symbol(str(item["Symbol"])),
                company=str(item["Security"]),
                entry_date=None,
                exit_date=None,
                gics_sector=str(item["GICS Sector"]),
                gics_industry=str(item["GICS Sub-Industry"]),
                source="wikipedia_current",
            )
        )
    return rows
