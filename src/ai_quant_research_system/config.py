from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
import os


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path
    parquet_backup_dir: Path


@dataclass(frozen=True)
class ApiKeys:
    polygon: str = ""
    fred: str = ""
    benzinga: str = ""


@dataclass(frozen=True)
class IngestionConfig:
    price_run_time_et: str = "17:00"
    news_interval_minutes: int = 15
    suspicious_return_threshold: float = 0.50


@dataclass(frozen=True)
class LlmConfig:
    provider: str = "heuristic"
    model: str = "local-heuristic-v1"
    prompt_version: str = "news_score_v1"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    concurrency: int = 5
    rate_limit_per_minute: int = 50
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass(frozen=True)
class SourcesConfig:
    news_primary: str = "yfinance"
    sec_user_agent: str = "AI Quant Research System contact@example.com"


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    api_keys: ApiKeys
    ingestion: IngestionConfig
    llm: LlmConfig
    sources: SourcesConfig
    timezone: str = "America/New_York"


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    base_dir = config_path.parent
    db = raw.get("database", {})
    keys = raw.get("api_keys", {})
    ingestion = raw.get("ingestion", {})
    calendar = raw.get("calendar", {})
    llm = raw.get("llm", {})
    sources = raw.get("sources", {})

    db_path = Path(db.get("path", "data/ai_quant.duckdb"))
    backup_dir = Path(db.get("parquet_backup_dir", "data/parquet"))
    if not db_path.is_absolute():
        db_path = base_dir / db_path
    if not backup_dir.is_absolute():
        backup_dir = base_dir / backup_dir

    return AppConfig(
        database=DatabaseConfig(path=db_path, parquet_backup_dir=backup_dir),
        api_keys=ApiKeys(
            polygon=keys.get("polygon", ""),
            fred=keys.get("fred", ""),
            benzinga=keys.get("benzinga", ""),
        ),
        ingestion=IngestionConfig(
            price_run_time_et=ingestion.get("price_run_time_et", "17:00"),
            news_interval_minutes=int(ingestion.get("news_interval_minutes", 15)),
            suspicious_return_threshold=float(ingestion.get("suspicious_return_threshold", 0.50)),
        ),
        llm=LlmConfig(
            provider=llm.get("provider", "heuristic"),
            model=llm.get("model", "local-heuristic-v1"),
            prompt_version=llm.get("prompt_version", "news_score_v1"),
            api_key=llm.get("api_key", "") or os.environ.get(llm.get("api_key_env", "OPENAI_API_KEY"), ""),
            api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"),
            base_url=llm.get("base_url", "https://api.openai.com/v1").rstrip("/"),
            concurrency=int(llm.get("concurrency", 5)),
            rate_limit_per_minute=int(llm.get("rate_limit_per_minute", 50)),
            max_retries=int(llm.get("max_retries", 3)),
            timeout_seconds=int(llm.get("timeout_seconds", 30)),
        ),
        sources=SourcesConfig(
            news_primary=sources.get("news_primary", "yfinance"),
            sec_user_agent=sources.get("sec_user_agent", "AI Quant Research System contact@example.com"),
        ),
        timezone=calendar.get("timezone", "America/New_York"),
    )
