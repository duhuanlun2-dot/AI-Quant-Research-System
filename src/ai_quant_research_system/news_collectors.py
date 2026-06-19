from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from typing import Any
from urllib.request import Request, urlopen

from .calendar import normalize_effective_date
from .records import NewsRecord


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def content_hash(*parts: str | None) -> str:
    joined = "|".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def parse_unix_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    return datetime.now(timezone.utc)


def normalize_yahoo_news_item(ticker: str, raw: dict[str, Any]) -> NewsRecord | None:
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
    title = content.get("title") or raw.get("title")
    if not title:
        return None

    summary = content.get("summary") or content.get("description") or raw.get("summary")
    provider = content.get("provider") or raw.get("publisher") or {}
    provider_name = provider.get("displayName") if isinstance(provider, dict) else provider
    source = str(provider_name or "Yahoo Finance")

    click_url = content.get("clickThroughUrl") or content.get("canonicalUrl") or raw.get("link")
    if isinstance(click_url, dict):
        url = click_url.get("url")
    else:
        url = click_url

    published_raw = content.get("pubDate") or content.get("providerPublishTime") or raw.get("providerPublishTime")
    if isinstance(published_raw, str) and not published_raw.isdigit():
        published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        published_at = parse_unix_timestamp(published_raw)

    market_session, effective_date = normalize_effective_date(published_at)
    fingerprint = content_hash(ticker, title, summary, source, url)
    return NewsRecord(
        news_id=f"yf_{fingerprint}",
        ticker=ticker.upper(),
        title=str(title),
        summary=str(summary or title),
        body=str(summary or title),
        source=source,
        url=str(url) if url else None,
        published_at=published_at,
        market_session=market_session,
        effective_date=effective_date,
        is_sec_filing=False,
        content_hash=fingerprint,
    )


def fetch_yahoo_finance_news(tickers: list[str], limit_per_ticker: int = 10) -> list[NewsRecord]:
    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install yfinance first: pip install -r requirements.txt") from exc

    records: list[NewsRecord] = []
    seen: set[str] = set()
    for ticker in [item.strip().upper().replace(".", "-") for item in tickers if item.strip()]:
        try:
            items = yf.Ticker(ticker).news or []
        except Exception:
            items = []
        for raw in items[:limit_per_ticker]:
            normalized = normalize_yahoo_news_item(ticker, raw)
            if normalized is None or normalized.news_id in seen:
                continue
            seen.add(normalized.news_id)
            records.append(normalized)
    return records


def sec_get_json(url: str, user_agent: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - official SEC public API.
        return json.loads(response.read().decode("utf-8"))


def fetch_sec_ticker_map(user_agent: str) -> dict[str, int]:
    payload = sec_get_json(SEC_COMPANY_TICKERS_URL, user_agent)
    result: dict[str, int] = {}
    for item in payload.values():
        ticker = str(item["ticker"]).upper().replace(".", "-")
        result[ticker] = int(item["cik_str"])
    return result


def fetch_sec_filings_news(
    tickers: list[str],
    user_agent: str,
    forms: set[str] | None = None,
    since: date | None = None,
    limit_per_ticker: int = 20,
) -> list[NewsRecord]:
    allowed_forms = forms or {"8-K", "10-Q", "10-K"}
    ticker_map = fetch_sec_ticker_map(user_agent)
    records: list[NewsRecord] = []

    for ticker in [item.strip().upper().replace(".", "-") for item in tickers if item.strip()]:
        cik = ticker_map.get(ticker)
        if cik is None:
            continue
        payload = sec_get_json(SEC_SUBMISSIONS_URL.format(cik=cik), user_agent)
        recent = payload.get("filings", {}).get("recent", {})
        forms_list = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        added = 0
        for form, filing_date, accession, doc in zip(forms_list, dates, accessions, docs):
            if form not in allowed_forms:
                continue
            filed = date.fromisoformat(filing_date)
            if since is not None and filed < since:
                continue
            published_at = datetime.combine(filed, time(12, 0), tzinfo=timezone.utc)
            market_session, effective_date = normalize_effective_date(published_at)
            accession_compact = str(accession).replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_compact}/{doc}"
            title = f"{ticker} SEC filing: {form} filed {filing_date}"
            summary = f"{ticker} filed Form {form} with the SEC on {filing_date}. Accession number: {accession}."
            fingerprint = content_hash(ticker, form, filing_date, accession, url)
            records.append(
                NewsRecord(
                    news_id=f"sec_{fingerprint}",
                    ticker=ticker,
                    title=title,
                    summary=summary,
                    body=summary,
                    source="SEC EDGAR",
                    url=url,
                    published_at=published_at,
                    market_session=market_session,
                    effective_date=effective_date,
                    is_sec_filing=True,
                    content_hash=fingerprint,
                )
            )
            added += 1
            if added >= limit_per_ticker:
                break
    return records
