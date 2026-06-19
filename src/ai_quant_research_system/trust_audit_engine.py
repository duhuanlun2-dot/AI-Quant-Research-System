from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .database import connect
from .factor_engine import universe_coverage
from .portfolio_engine import latest_model_version, model_info


@dataclass(frozen=True)
class TrustAuditResult:
    model_version: str
    artifact_dir: str
    report_path: str
    summary: dict[str, Any]


def run_trust_audit(
    db_path: str | Path,
    artifact_root: str | Path,
    model_version: str | None = None,
) -> TrustAuditResult:
    resolved_version = model_version or latest_model_version(db_path)
    info = model_info(db_path, resolved_version)
    artifact_dir = Path(info.get("artifact_dir") or Path(artifact_root) / resolved_version)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    constituents = constituent_bias_report(db_path, artifact_dir)
    exposures = exposure_report(db_path, resolved_version, artifact_dir)
    liquidity = liquidity_report(db_path, resolved_version, artifact_dir)
    sensitivity = sensitivity_report(artifact_dir)
    windows = multi_window_report(artifact_dir)

    summary = {
        "model_version": resolved_version,
        "constituent_bias": constituents,
        "exposure": exposures,
        "liquidity": liquidity,
        "sensitivity": sensitivity,
        "multi_window": windows,
        "caveat": (
            "This is a research audit. Current S&P 500 constituents are not a historical "
            "point-in-time universe unless entry/exit dates are populated from a licensed PIT source."
        ),
    }
    summary_path = artifact_dir / "trust_audit_summary.json"
    report_path = artifact_dir / "trust_audit_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return TrustAuditResult(
        model_version=resolved_version,
        artifact_dir=str(artifact_dir),
        report_path=str(report_path),
        summary=summary,
    )


def constituent_bias_report(db_path: str | Path, artifact_dir: Path) -> dict[str, Any]:
    coverage = universe_coverage(db_path)
    with connect(db_path) as con:
        factor = con.execute(
            """
            SELECT count(DISTINCT ticker), min(date), max(date)
            FROM factor_table
            """
        ).fetchone()
    if coverage.has_serious_pit_history:
        status = "point_in_time_universe_available"
        interpretation = (
            "The constituent table has historical membership metadata. Factor construction can use "
            "point-in-time membership filtering, but you should still validate the source license and coverage."
        )
    elif coverage.rows_with_entry_date or coverage.rows_with_exit_date:
        status = "partial_point_in_time_metadata_not_enough"
        interpretation = (
            "Some entry/exit metadata exists, but it does not look like a complete historical S&P 500 "
            "membership table. Current-only sources with date-added fields still leave survivorship bias."
        )
    else:
        status = "high_survivorship_bias_risk"
        interpretation = (
            "Universe appears to be based on current constituents; historical delisted/removed names "
            "are likely missing, so backtest results may be overstated."
        )
    report = {
        "status": status,
        "constituent_rows": coverage.total_rows,
        "constituent_tickers": coverage.distinct_tickers,
        "snapshot_dates": coverage.snapshot_dates,
        "rows_with_entry_date": coverage.rows_with_entry_date,
        "rows_with_exit_date": coverage.rows_with_exit_date,
        "dated_ratio": coverage.dated_ratio,
        "has_serious_pit_history": coverage.has_serious_pit_history,
        "constituent_membership_date_range": [coverage.min_entry_date, coverage.max_exit_date],
        "factor_tickers": int(factor[0] or 0),
        "factor_date_range": [str(factor[1]), str(factor[2])],
        "interpretation": interpretation,
    }
    pd.DataFrame([report]).to_csv(artifact_dir / "constituent_bias_report.csv", index=False)
    return report


def exposure_report(db_path: str | Path, model_version: str, artifact_dir: Path) -> dict[str, Any]:
    with connect(db_path) as con:
        latest_date = con.execute(
            "SELECT max(date) FROM portfolio_positions WHERE model_version = ?",
            [model_version],
        ).fetchone()[0]
        if latest_date is None:
            return {"status": "no_positions"}
        exposures = con.execute(
            """
            SELECT
                p.date,
                coalesce(f.gics_sector, 'Unknown') AS gics_sector,
                sum(p.target_weight) AS weight,
                sum(p.target_weight * coalesce(f.beta, 1.0)) AS beta_contribution
            FROM portfolio_positions p
            LEFT JOIN factor_table f
              ON p.date = f.date AND p.ticker = f.ticker
            WHERE p.model_version = ?
              AND p.date = ?
            GROUP BY p.date, coalesce(f.gics_sector, 'Unknown')
            ORDER BY weight DESC
            """,
            [model_version, latest_date],
        ).fetchdf()
        beta = con.execute(
            """
            SELECT sum(p.target_weight * coalesce(f.beta, 1.0)) AS portfolio_beta
            FROM portfolio_positions p
            LEFT JOIN factor_table f
              ON p.date = f.date AND p.ticker = f.ticker
            WHERE p.model_version = ?
              AND p.date = ?
            """,
            [model_version, latest_date],
        ).fetchone()[0]
    exposures.to_csv(artifact_dir / "sector_beta_exposure.csv", index=False)
    top_sector = exposures.iloc[0].to_dict() if not exposures.empty else {}
    return {
        "latest_date": str(latest_date),
        "portfolio_beta": None if beta is None else float(beta),
        "top_sector": top_sector,
        "sector_count": int(len(exposures)),
    }


def liquidity_report(db_path: str | Path, model_version: str, artifact_dir: Path) -> dict[str, Any]:
    with connect(db_path) as con:
        latest_date = con.execute(
            "SELECT max(date) FROM portfolio_positions WHERE model_version = ?",
            [model_version],
        ).fetchone()[0]
        if latest_date is None:
            return {"status": "no_positions"}
        frame = con.execute(
            """
            WITH adv AS (
                SELECT
                    ticker,
                    avg(close * volume) AS avg_dollar_volume_20d
                FROM (
                    SELECT *,
                           row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM daily_prices_clean
                    WHERE date <= ?
                      AND close IS NOT NULL
                      AND volume IS NOT NULL
                )
                WHERE rn <= 20
                GROUP BY ticker
            )
            SELECT p.ticker, p.target_weight, adv.avg_dollar_volume_20d
            FROM portfolio_positions p
            LEFT JOIN adv ON p.ticker = adv.ticker
            WHERE p.model_version = ?
              AND p.date = ?
            ORDER BY p.target_weight DESC
            """,
            [latest_date, model_version, latest_date],
        ).fetchdf()
    frame.to_csv(artifact_dir / "liquidity_report.csv", index=False)
    if frame.empty:
        return {"status": "no_positions"}
    low_liquidity = frame[frame["avg_dollar_volume_20d"].fillna(0) < 50_000_000]
    return {
        "latest_date": str(latest_date),
        "holding_count": int(len(frame)),
        "low_liquidity_count_under_50m_adv": int(len(low_liquidity)),
        "min_avg_dollar_volume_20d": None
        if frame["avg_dollar_volume_20d"].dropna().empty
        else float(frame["avg_dollar_volume_20d"].min()),
        "median_avg_dollar_volume_20d": None
        if frame["avg_dollar_volume_20d"].dropna().empty
        else float(frame["avg_dollar_volume_20d"].median()),
    }


def sensitivity_report(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "optimization_report.csv"
    if not path.exists():
        return {"status": "missing_optimization_report"}
    frame = pd.read_csv(path)
    frame.to_csv(artifact_dir / "parameter_sensitivity.csv", index=False)
    feasible = frame[frame.get("feasible", False) == True] if "feasible" in frame else pd.DataFrame()
    source = feasible if not feasible.empty else frame
    if source.empty:
        return {"status": "empty_optimization_report"}
    top = source.sort_values(["sharpe", "total_return"], ascending=[False, False]).head(5)
    return {
        "candidate_count": int(len(frame)),
        "feasible_count": int(len(feasible)),
        "top_candidates": top.to_dict(orient="records"),
    }


def multi_window_report(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "backtest_results.csv"
    if not path.exists():
        return {"status": "missing_backtest_results"}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"status": "empty_backtest_results"}
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    rows = []
    for window in (21, 42, 63, 126):
        if len(frame) < window:
            continue
        subset = frame.tail(window)
        rows.append(window_metrics(subset, f"last_{window}_trading_days"))
    monthly = frame.set_index("date").resample("ME").apply(
        {
            "portfolio_value": "last",
            "benchmark_value": "last",
            "strategy_return": lambda s: (1 + s.fillna(0)).prod() - 1,
            "benchmark_return": lambda s: (1 + s.fillna(0)).prod() - 1,
        }
    )
    if not monthly.empty:
        monthly.to_csv(artifact_dir / "monthly_returns.csv")
    out = pd.DataFrame(rows)
    out.to_csv(artifact_dir / "multi_window_backtest.csv", index=False)
    return {"windows": rows}


def window_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    strategy_return = frame["portfolio_value"].iloc[-1] / frame["portfolio_value"].iloc[0] - 1
    benchmark_return = frame["benchmark_value"].iloc[-1] / frame["benchmark_value"].iloc[0] - 1
    return {
        "window": label,
        "start": str(frame["date"].iloc[0].date()),
        "end": str(frame["date"].iloc[-1].date()),
        "strategy_return": float(strategy_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(strategy_return - benchmark_return),
        "max_drawdown": float(frame["drawdown"].min()),
        "avg_turnover": float(frame["turnover"].mean()),
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Trust Audit Report",
        "",
        f"Model version: `{summary['model_version']}`",
        "",
        "## Universe Bias",
        "",
        f"- Status: `{summary['constituent_bias'].get('status')}`",
        f"- Interpretation: {summary['constituent_bias'].get('interpretation')}",
        "",
        "## Exposure",
        "",
        f"- Portfolio beta: {summary['exposure'].get('portfolio_beta')}",
        f"- Top sector: {summary['exposure'].get('top_sector')}",
        "",
        "## Liquidity",
        "",
        f"- Holdings under $50m ADV: {summary['liquidity'].get('low_liquidity_count_under_50m_adv')}",
        f"- Median 20d ADV: {summary['liquidity'].get('median_avg_dollar_volume_20d')}",
        "",
        "## Sensitivity",
        "",
        f"- Candidates: {summary['sensitivity'].get('candidate_count')}",
        f"- Feasible: {summary['sensitivity'].get('feasible_count')}",
        "",
        "## Multi-Window Backtest",
        "",
    ]
    for row in summary["multi_window"].get("windows", []):
        lines.append(
            f"- {row['window']}: strategy {row['strategy_return']:.2%}, "
            f"SPY {row['benchmark_return']:.2%}, max DD {row['max_drawdown']:.2%}"
        )
    lines.extend(["", f"Note: {summary['caveat']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
