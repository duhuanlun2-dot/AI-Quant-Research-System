from __future__ import annotations

import asyncio
import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from time import monotonic
from typing import Any

from .config import LlmConfig
from .database import connect


FACTOR_WEIGHTS = {
    "sentiment": 0.35,
    "importance": 0.20,
    "surprise": 0.15,
    "risk": -0.20,
    "revenue_impact": 0.10,
}


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    ticker: str
    effective_date: date
    title: str
    summary: str | None
    market_session: str | None
    is_sec_filing: bool


@dataclass(frozen=True)
class NewsScore:
    sentiment: float
    importance: float
    surprise: float
    risk: float
    revenue_impact: float
    margin_impact: float
    confidence: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RateLimiter:
    def __init__(self, rate_limit_per_minute: int):
        self.min_interval = 60.0 / max(1, rate_limit_per_minute)
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = monotonic()
            delay = self.min_interval - (now - self._last_call)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = monotonic()


class HeuristicNewsScorer:
    """Local scorer for pipeline testing when no paid LLM key is configured."""

    positive_words = {
        "beat",
        "beats",
        "raise",
        "raises",
        "growth",
        "record",
        "upgrade",
        "surge",
        "profit",
        "strong",
        "launch",
        "approval",
        "partnership",
        "announces",
        "expands",
    }
    negative_words = {
        "miss",
        "cuts",
        "cut",
        "downgrade",
        "lawsuit",
        "probe",
        "fall",
        "falls",
        "weak",
        "recall",
        "risk",
        "warning",
        "loss",
        "decline",
        "delay",
    }
    risk_words = {"lawsuit", "probe", "recall", "warning", "regulatory", "fraud", "breach", "default"}
    surprise_words = {"unexpected", "surprise", "beats", "misses", "guidance", "upgrade", "downgrade"}
    revenue_words = {"revenue", "sales", "orders", "demand", "growth", "launch", "partnership"}
    margin_words = {"margin", "cost", "pricing", "profit", "efficiency", "layoff", "expense"}

    async def score(self, item: NewsItem) -> NewsScore:
        text = f"{item.title} {item.summary or ''}".lower()
        words = {word.strip(".,:;!?()[]{}\"'") for word in text.split()}
        pos = len(words & self.positive_words)
        neg = len(words & self.negative_words)
        risk_hits = len(words & self.risk_words)
        surprise_hits = len(words & self.surprise_words)
        revenue_hits = len(words & self.revenue_words)
        margin_hits = len(words & self.margin_words)
        sec_boost = 0.15 if item.is_sec_filing else 0.0
        length = len(text)

        sentiment = clamp((pos - neg) / 4.0, -1.0, 1.0)
        importance = clamp(0.25 + length / 800.0 + sec_boost + 0.05 * (pos + neg), 0.0, 1.0)
        surprise = clamp(0.10 + 0.20 * surprise_hits + (0.10 if item.market_session != "regular" else 0.0), 0.0, 1.0)
        risk = clamp(0.05 + 0.25 * risk_hits + 0.10 * neg, 0.0, 1.0)
        revenue_impact = clamp(sentiment * 0.60 + 0.10 * revenue_hits, -1.0, 1.0)
        margin_impact = clamp(sentiment * 0.50 + 0.10 * margin_hits - 0.10 * risk_hits, -1.0, 1.0)
        confidence = clamp(0.55 + 0.10 * min(pos + neg + risk_hits + surprise_hits, 3), 0.0, 0.9)
        return NewsScore(sentiment, importance, surprise, risk, revenue_impact, margin_impact, confidence)


class OpenAICompatibleNewsScorer:
    def __init__(self, config: LlmConfig):
        if not config.api_key:
            raise RuntimeError(f"LLM provider '{config.provider}' requires an API key.")
        self.config = config

    async def score(self, item: NewsItem) -> NewsScore:
        return await asyncio.to_thread(self._score_sync, item)

    def _score_sync(self, item: NewsItem) -> NewsScore:
        system_prompt = (
            "You are a financial news analyst. Read the news title and summary. "
            "Return only valid JSON with these numeric fields: sentiment, importance, surprise, "
            "risk, revenue_impact, margin_impact, confidence. "
            "sentiment/revenue_impact/margin_impact are in [-1, 1]. Others are in [0, 1]."
        )
        user_prompt = f"ticker: {item.ticker}\ntitle: {item.title}\nsummary: {item.summary or ''}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:  # noqa: S310 - configured API endpoint.
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)
        return NewsScore(
            sentiment=clamp(float(data["sentiment"]), -1.0, 1.0),
            importance=clamp(float(data["importance"]), 0.0, 1.0),
            surprise=clamp(float(data["surprise"]), 0.0, 1.0),
            risk=clamp(float(data["risk"]), 0.0, 1.0),
            revenue_impact=clamp(float(data["revenue_impact"]), -1.0, 1.0),
            margin_impact=clamp(float(data["margin_impact"]), -1.0, 1.0),
            confidence=clamp(float(data["confidence"]), 0.0, 1.0),
        )


def make_scorer(config: LlmConfig):
    if config.provider.lower() in {"heuristic", "local", "local-heuristic"}:
        return HeuristicNewsScorer()
    if config.provider.lower() in {"openai", "openai_compatible", "external_api"}:
        return OpenAICompatibleNewsScorer(config)
    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")


def fetch_pending_news(
    db_path: str,
    prompt_version: str,
    limit: int,
    retry_failed: bool = False,
) -> list[NewsItem]:
    status_filter = "s.score_status IS NULL"
    if retry_failed:
        status_filter = "(s.score_status IS NULL OR s.score_status = 'failed')"

    sql = f"""
    SELECT
        n.news_id,
        n.ticker,
        n.effective_date,
        n.title,
        n.summary,
        n.market_session,
        n.is_sec_filing
    FROM clean_news n
    LEFT JOIN news_scores s
      ON n.news_id = s.news_id
     AND s.prompt_version = ?
    WHERE {status_filter}
    ORDER BY n.effective_date, n.ticker, n.published_at
    LIMIT ?
    """
    with connect(db_path) as con:
        rows = con.execute(sql, [prompt_version, limit]).fetchall()
    return [
        NewsItem(
            news_id=row[0],
            ticker=row[1],
            effective_date=row[2],
            title=row[3],
            summary=row[4],
            market_session=row[5],
            is_sec_filing=bool(row[6]),
        )
        for row in rows
    ]


def write_score_success(db_path: str, item: NewsItem, score: NewsScore, prompt_version: str) -> None:
    with connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO news_scores
            (news_id, ticker, effective_date, sentiment, importance, surprise, risk,
             revenue_impact, margin_impact, confidence, score_status, prompt_version,
             error_message, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'success', ?, NULL, current_timestamp)
            """,
            [
                item.news_id,
                item.ticker,
                item.effective_date,
                score.sentiment,
                score.importance,
                score.surprise,
                score.risk,
                score.revenue_impact,
                score.margin_impact,
                score.confidence,
                prompt_version,
            ],
        )


def write_score_failure(db_path: str, item: NewsItem, prompt_version: str, error_message: str) -> None:
    with connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO news_scores
            (news_id, ticker, effective_date, score_status, prompt_version, error_message, scored_at)
            VALUES (?, ?, ?, 'failed', ?, ?, current_timestamp)
            """,
            [item.news_id, item.ticker, item.effective_date, prompt_version, error_message[:1000]],
        )


async def score_one(
    db_path: str,
    item: NewsItem,
    scorer,
    config: LlmConfig,
    limiter: RateLimiter,
) -> str:
    last_error = ""
    for attempt in range(1, config.max_retries + 1):
        try:
            await limiter.wait()
            score = await asyncio.wait_for(scorer.score(item), timeout=config.timeout_seconds)
            write_score_success(db_path, item, score, config.prompt_version)
            return "success"
        except Exception as exc:  # noqa: BLE001 - batch pipeline records errors per item.
            last_error = f"attempt {attempt}: {exc}"
            await asyncio.sleep(min(2**attempt, 10))
    write_score_failure(db_path, item, config.prompt_version, last_error or "unknown scoring error")
    return "failed"


async def score_news_batch_async(
    db_path: str,
    config: LlmConfig,
    limit: int = 100,
    retry_failed: bool = False,
) -> dict[str, int]:
    items = fetch_pending_news(db_path, config.prompt_version, limit, retry_failed=retry_failed)
    scorer = make_scorer(config)
    limiter = RateLimiter(config.rate_limit_per_minute)
    semaphore = asyncio.Semaphore(config.concurrency)

    async def guarded(item: NewsItem) -> str:
        async with semaphore:
            return await score_one(db_path, item, scorer, config, limiter)

    results = await asyncio.gather(*(guarded(item) for item in items))
    return {
        "pending": len(items),
        "success": sum(1 for result in results if result == "success"),
        "failed": sum(1 for result in results if result == "failed"),
    }


def score_news_batch(
    db_path: str,
    config: LlmConfig,
    limit: int = 100,
    retry_failed: bool = False,
) -> dict[str, int]:
    return asyncio.run(score_news_batch_async(db_path, config, limit=limit, retry_failed=retry_failed))


def aggregate_news_factors(db_path: str) -> int:
    with connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO news_factors_raw
            SELECT
                effective_date AS date,
                ticker,
                CASE WHEN sum(importance * confidence) = 0 THEN avg(sentiment)
                     ELSE sum(sentiment * importance * confidence) / sum(importance * confidence) END AS news_sentiment,
                avg(importance) AS news_importance,
                max(surprise) AS news_surprise,
                max(risk) AS news_risk,
                CASE WHEN sum(importance * confidence) = 0 THEN avg(revenue_impact)
                     ELSE sum(revenue_impact * importance * confidence) / sum(importance * confidence) END AS news_revenue_impact,
                CASE WHEN sum(importance * confidence) = 0 THEN avg(margin_impact)
                     ELSE sum(margin_impact * importance * confidence) / sum(importance * confidence) END AS news_margin_impact,
                count(news_id) AS news_coverage,
                current_timestamp AS computed_at
            FROM news_scores
            WHERE score_status = 'success'
            GROUP BY effective_date, ticker
            """
        )
        con.execute(
            """
            INSERT OR REPLACE INTO news_factors
            SELECT
                date,
                ticker,
                news_sentiment,
                news_importance,
                news_surprise,
                news_risk,
                news_revenue_impact,
                news_margin_impact,
                news_coverage,
                0.35 * news_sentiment
                  + 0.20 * news_importance
                  + 0.15 * news_surprise
                  - 0.20 * news_risk
                  + 0.10 * news_revenue_impact AS news_ai_score,
                current_timestamp AS computed_at
            FROM news_factors_raw
            """
        )
        return con.execute("SELECT count(*) FROM news_factors").fetchone()[0]


def pending_news_count(db_path: str, prompt_version: str) -> int:
    with connect(db_path) as con:
        return con.execute(
            """
            SELECT count(*)
            FROM clean_news n
            LEFT JOIN news_scores s
              ON n.news_id = s.news_id
             AND s.prompt_version = ?
            WHERE s.score_status IS NULL
            """,
            [prompt_version],
        ).fetchone()[0]
