from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from .database import connect


@dataclass(frozen=True)
class BacktestResult:
    model_version: str
    start_date: str | None
    end_date: str | None
    signal_rows: int
    position_rows: int
    trade_rows: int
    result_rows: int
    artifact_dir: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    model_version: str
    candidate_count: int
    feasible_count: int
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    artifact_dir: str


def latest_model_version(db_path: str | Path) -> str:
    with connect(db_path) as con:
        row = con.execute(
            """
            SELECT model_version
            FROM model_registry
            ORDER BY trained_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        raise RuntimeError("No trained model found. Run train-models first.")
    return str(row[0])


def model_info(db_path: str | Path, model_version: str) -> dict[str, Any]:
    with connect(db_path) as con:
        row = con.execute(
            """
            SELECT model_version, test_start, test_end, artifact_dir, metrics_json
            FROM model_registry
            WHERE model_version = ?
            """,
            [model_version],
        ).fetchone()
    if not row:
        raise RuntimeError(f"Model version not found: {model_version}")
    metrics = json.loads(row[4] or "{}")
    final_holdout = (metrics.get("split_summary") or {}).get("final_holdout") or {}
    final_start = final_holdout.get("start")
    final_end = final_holdout.get("end")
    use_final_holdout = bool(final_start and final_end and (final_holdout.get("rows") or 0) > 0)
    return {
        "model_version": row[0],
        "test_start": final_start if use_final_holdout else row[1],
        "test_end": final_end if use_final_holdout else row[2],
        "artifact_dir": row[3],
        "oos_window_type": "final_holdout" if use_final_holdout else "test",
    }


def load_predictions(
    db_path: str | Path,
    model_version: str,
    exclude_tickers: list[str] | None = None,
    min_avg_dollar_volume: float = 0.0,
) -> pd.DataFrame:
    info = model_info(db_path, model_version)
    excluded = [ticker.upper() for ticker in (exclude_tickers or [])]
    with connect(db_path) as con:
        frame = con.execute(
            """
            SELECT date, ticker, predicted_excess_return_5d
            FROM model_predictions
            WHERE model_version = ?
              AND predicted_excess_return_5d IS NOT NULL
              AND ticker NOT IN (SELECT * FROM unnest(?))
              AND (? IS NULL OR date >= ?)
              AND (? IS NULL OR date <= ?)
            ORDER BY date, predicted_excess_return_5d DESC
            """,
            [model_version, excluded, info["test_start"], info["test_start"], info["test_end"], info["test_end"]],
        ).fetchdf()
    if frame.empty:
        raise RuntimeError("No model predictions available for the model test window.")
    frame["date"] = pd.to_datetime(frame["date"])
    if min_avg_dollar_volume > 0:
        with connect(db_path) as con:
            liquidity = con.execute(
                """
                SELECT
                    date,
                    ticker,
                    avg(close * volume) OVER (
                        PARTITION BY ticker
                        ORDER BY date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS avg_dollar_volume_20d
                FROM daily_prices_clean
                WHERE close IS NOT NULL
                  AND volume IS NOT NULL
                """
            ).fetchdf()
        liquidity["date"] = pd.to_datetime(liquidity["date"])
        frame = frame.merge(liquidity, on=["date", "ticker"], how="left")
        frame = frame[frame["avg_dollar_volume_20d"].fillna(0) >= min_avg_dollar_volume].copy()
        if frame.empty:
            raise RuntimeError("Liquidity filter removed all predictions. Lower min_avg_dollar_volume.")
    return frame


def build_signals(
    predictions: pd.DataFrame,
    model_version: str,
    top_pct: float = 0.10,
    max_weight: float = 0.05,
    min_weight: float = 0.005,
    target_exposure: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_pct = min(max(top_pct, 0.01), 1.0)
    max_weight = min(max(max_weight, 0.001), 1.0)

    for day, group in predictions.groupby("date"):
        ranked = group.sort_values("predicted_excess_return_5d", ascending=False).reset_index(drop=True)
        selected_count = max(1, int(len(ranked) * top_pct))
        selected_weights = score_weighted_targets(
            ranked.head(selected_count)["predicted_excess_return_5d"],
            max_weight=max_weight,
            min_weight=min_weight,
            target_exposure=target_exposure,
        )
        for idx, row in ranked.iterrows():
            target_weight = selected_weights[idx] if idx < selected_count else 0.0
            is_long = target_weight > 0.0
            rows.append(
                {
                    "date": day.date(),
                    "ticker": row["ticker"],
                    "predicted_excess_return_5d": float(row["predicted_excess_return_5d"]),
                    "prediction_rank": int(idx + 1),
                    "signal": "long" if is_long else "flat",
                    "target_weight": float(target_weight),
                    "model_version": model_version,
                }
            )
    return pd.DataFrame(rows)


def score_weighted_targets(
    scores: pd.Series,
    max_weight: float = 0.05,
    min_weight: float = 0.005,
    target_exposure: float = 1.0,
) -> list[float]:
    if scores.empty:
        return []
    max_weight = min(max(max_weight, 0.001), 1.0)
    min_weight = min(max(min_weight, 0.0), max_weight)
    target_exposure = min(max(target_exposure, 0.0), 1.0)
    clean = pd.to_numeric(scores, errors="coerce").fillna(scores.mean() if scores.notna().any() else 0.0)
    shifted = clean - clean.min()
    shifted = shifted + max(float(shifted.max()), 1.0) * 1e-6
    if float(shifted.sum()) <= 0:
        raw = pd.Series([1.0 / len(clean)] * len(clean), index=clean.index)
    else:
        raw = shifted / shifted.sum()
    capped = cap_and_renormalize(raw.tolist(), max_weight=max_weight, target_exposure=target_exposure)
    if min_weight <= 0:
        return capped

    keep = [idx for idx, weight in enumerate(capped) if weight >= min_weight]
    if not keep:
        keep = [int(raw.reset_index(drop=True).idxmax())]
    filtered_raw = [float(raw.iloc[idx]) if idx in keep else 0.0 for idx in range(len(raw))]
    return cap_and_renormalize(filtered_raw, max_weight=max_weight, target_exposure=target_exposure)


def cap_and_renormalize(weights: list[float], max_weight: float, target_exposure: float = 1.0) -> list[float]:
    if not weights:
        return []
    total_capacity = len(weights) * max_weight
    target_total = min(max(target_exposure, 0.0), 1.0, total_capacity)
    capped = [0.0] * len(weights)
    active = set(range(len(weights)))
    raw = [max(0.0, float(weight)) for weight in weights]

    while active:
        active_total = sum(raw[idx] for idx in active)
        remaining = target_total - sum(capped)
        if remaining <= 1e-12:
            break
        if active_total <= 1e-12:
            equal = remaining / len(active)
            for idx in list(active):
                capped[idx] = min(max_weight, equal)
            break

        newly_capped: set[int] = set()
        for idx in active:
            proposed = remaining * raw[idx] / active_total
            if proposed >= max_weight:
                capped[idx] = max_weight
                newly_capped.add(idx)
        if not newly_capped:
            for idx in active:
                capped[idx] = remaining * raw[idx] / active_total
            break
        active -= newly_capped

    return capped


def rebalance_positions(signals: pd.DataFrame, rebalance_days: int = 5) -> pd.DataFrame:
    long_signals = signals[signals["signal"] == "long"].copy()
    if long_signals.empty:
        return pd.DataFrame(columns=["date", "ticker", "target_weight", "model_version"])

    dates = sorted(pd.to_datetime(long_signals["date"]).drop_duplicates())
    rebalance_days = max(1, int(rebalance_days))
    rebalance_dates = set(dates[::rebalance_days])
    positions = long_signals[pd.to_datetime(long_signals["date"]).isin(rebalance_dates)][
        ["date", "ticker", "target_weight", "model_version"]
    ].copy()
    positions["date"] = pd.to_datetime(positions["date"]).dt.date
    return positions.reset_index(drop=True)


def load_daily_returns(db_path: str | Path, tickers: list[str], start_date: Any, end_date: Any) -> pd.DataFrame:
    with connect(db_path) as con:
        prices = con.execute(
            """
            SELECT date, ticker, close
            FROM daily_prices_clean
            WHERE ticker IN (SELECT * FROM unnest(?))
              AND date >= ?
              AND date <= ?
              AND close IS NOT NULL
            ORDER BY ticker, date
            """,
            [tickers, start_date, end_date],
        ).fetchdf()
    if prices.empty:
        raise RuntimeError("No prices found for backtest tickers.")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    return prices


def load_execution_price_panel(db_path: str | Path, tickers: list[str], start_date: Any, end_date: Any) -> pd.DataFrame:
    with connect(db_path) as con:
        prices = con.execute(
            """
            SELECT date, ticker, open, high, low, close, volume, vwap
            FROM daily_prices_clean
            WHERE ticker IN (SELECT * FROM unnest(?))
              AND date >= ?
              AND date <= ?
              AND close IS NOT NULL
            ORDER BY ticker, date
            """,
            [tickers, start_date, end_date],
        ).fetchdf()
    if prices.empty:
        raise RuntimeError("No execution prices found for backtest tickers.")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["close_return"] = prices.groupby("ticker")["close"].pct_change()
    prices["execution_vwap"] = prices["vwap"].fillna((prices["high"].fillna(prices["close"]) + prices["low"].fillna(prices["close"]) + prices["close"]) / 3.0)
    return prices


def execution_price_for(price_panel: pd.DataFrame, day: Any, ticker: str, execution_price_model: str) -> float:
    rows = price_panel[
        (pd.to_datetime(price_panel["date"]).dt.date == pd.to_datetime(day).date())
        & (price_panel["ticker"] == ticker)
    ]
    if rows.empty:
        return 0.0
    row = rows.iloc[0]
    model = execution_price_model.lower()
    if model == "vwap":
        value = row.get("execution_vwap")
    elif model == "close":
        value = row.get("close")
    else:
        value = row.get("open")
    if pd.isna(value) or float(value) <= 0:
        value = row.get("close")
    return 0.0 if pd.isna(value) else float(value)


def build_execution_price_lookup(price_panel: pd.DataFrame) -> dict[tuple[Any, str, str], float]:
    lookup: dict[tuple[Any, str, str], float] = {}
    for _, row in price_panel.iterrows():
        day = pd.to_datetime(row["date"]).date()
        ticker = row["ticker"]
        close = 0.0 if pd.isna(row.get("close")) else float(row.get("close"))
        open_price = row.get("open")
        vwap = row.get("execution_vwap")
        lookup[(day, ticker, "close")] = close
        lookup[(day, ticker, "open")] = close if pd.isna(open_price) or float(open_price) <= 0 else float(open_price)
        lookup[(day, ticker, "vwap")] = close if pd.isna(vwap) or float(vwap) <= 0 else float(vwap)
    return lookup


def lookup_execution_price(
    lookup: dict[tuple[Any, str, str], float],
    day: Any,
    ticker: str,
    execution_price_model: str,
) -> float:
    model = execution_price_model.lower()
    date_key = pd.to_datetime(day).date()
    return lookup.get((date_key, ticker, model)) or lookup.get((date_key, ticker, "close")) or 0.0


def run_portfolio_backtest(
    db_path: str | Path,
    artifact_root: str | Path,
    model_version: str | None = None,
    top_pct: float = 0.10,
    max_weight: float = 0.05,
    min_weight: float = 0.005,
    rebalance_days: int = 5,
    transaction_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
    exclude_tickers: list[str] | None = None,
    target_exposure: float = 1.0,
    min_avg_dollar_volume: float = 0.0,
    execution_price_model: str = "open",
) -> BacktestResult:
    resolved_version = model_version or latest_model_version(db_path)
    excluded = exclude_tickers if exclude_tickers is not None else ["SPY", "QQQ"]
    predictions = load_predictions(
        db_path,
        resolved_version,
        exclude_tickers=excluded,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )
    result = simulate_portfolio_backtest(
        db_path,
        predictions,
        resolved_version,
        top_pct=top_pct,
        max_weight=max_weight,
        min_weight=min_weight,
        rebalance_days=rebalance_days,
        transaction_bps=transaction_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        target_exposure=target_exposure,
        execution_price_model=execution_price_model,
    )
    result["metrics"]["excluded_tickers"] = excluded

    artifact_dir = Path(artifact_root) / resolved_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        artifact_dir,
        result["results"],
        result["trades"],
        result["positions"],
        result["metrics"],
        orders=result["orders"],
        fills=result["fills"],
        current_positions=result["current_positions"],
    )
    write_backtest_tables(
        db_path,
        resolved_version,
        result["signals"],
        result["positions"],
        result["results"],
        result["trades"],
        result["orders"],
        result["fills"],
        result["current_positions"],
    )

    return BacktestResult(
        model_version=resolved_version,
        start_date=result["start_date"],
        end_date=result["end_date"],
        signal_rows=len(result["signals"]),
        position_rows=len(result["positions"]),
        trade_rows=len(result["trades"]),
        result_rows=len(result["results"]),
        artifact_dir=str(artifact_dir),
        metrics=result["metrics"],
    )


def run_portfolio_optimization(
    db_path: str | Path,
    artifact_root: str | Path,
    model_version: str | None = None,
    max_drawdown_limit: float = 0.10,
    transaction_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
    exclude_tickers: list[str] | None = None,
    min_avg_dollar_volume: float = 0.0,
    execution_price_model: str = "open",
) -> OptimizationResult:
    resolved_version = model_version or latest_model_version(db_path)
    excluded = exclude_tickers if exclude_tickers is not None else ["SPY", "QQQ"]
    predictions = load_predictions(
        db_path,
        resolved_version,
        exclude_tickers=excluded,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )
    unique_dates = sorted(pd.to_datetime(predictions["date"]).drop_duplicates())
    if len(unique_dates) >= 20:
        holdout_cut = max(1, int(len(unique_dates) * 0.70))
        optimization_dates = set(unique_dates[:holdout_cut])
        final_holdout_dates = set(unique_dates[holdout_cut:])
        optimization_predictions = predictions[pd.to_datetime(predictions["date"]).isin(optimization_dates)].copy()
        final_holdout_predictions = predictions[pd.to_datetime(predictions["date"]).isin(final_holdout_dates)].copy()
    else:
        optimization_predictions = predictions.copy()
        final_holdout_predictions = pd.DataFrame(columns=predictions.columns)
    artifact_dir = Path(artifact_root) / resolved_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.to_datetime(optimization_predictions["date"]).min().date()
    end_date = pd.to_datetime(optimization_predictions["date"]).max().date()
    cached_tickers = sorted(set(predictions["ticker"].tolist()) | {"SPY"})
    cached_price_panel = load_execution_price_panel(db_path, cached_tickers, start_date, end_date)
    cached_return_pivot = cached_price_panel.pivot(index="date", columns="ticker", values="close_return").sort_index().fillna(0.0)

    candidates = []
    for max_weight in (0.03, 0.04, 0.05):
        for target_exposure in (0.70, 0.80, 0.85, 0.90, 0.95, 1.00):
            candidates.append(
                {
                    "top_pct": 0.10,
                    "max_weight": max_weight,
                    "min_weight": 0.005,
                    "target_exposure": target_exposure,
                    "rebalance_days": 5,
                }
            )

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    dd_threshold = -abs(max_drawdown_limit)
    for params in candidates:
        try:
            simulated = simulate_portfolio_backtest(
                db_path,
                optimization_predictions,
                resolved_version,
                transaction_bps=transaction_bps,
                slippage_bps=slippage_bps,
                initial_capital=initial_capital,
                return_pivot=cached_return_pivot,
                price_panel=cached_price_panel,
                execution_price_model=execution_price_model,
                **params,
            )
        except Exception as exc:
            rows.append({**params, "error": str(exc), "feasible": False})
            continue
        metrics = simulated["metrics"]
        feasible = metrics.get("max_drawdown") is not None and metrics["max_drawdown"] >= dd_threshold
        row = {
            **params,
            "total_return": metrics.get("total_return"),
            "benchmark_return": metrics.get("benchmark_return"),
            "excess_return": metrics.get("excess_return"),
            "cagr": metrics.get("cagr"),
            "sharpe": metrics.get("sharpe"),
            "max_drawdown": metrics.get("max_drawdown"),
            "avg_turnover": metrics.get("avg_turnover"),
            "feasible": feasible,
        }
        rows.append(row)
        score = metrics.get("sharpe")
        if score is None:
            continue
        if feasible:
            candidate_key = (1, float(score), float(metrics.get("total_return") or 0.0))
        else:
            dd_gap = abs(float(metrics.get("max_drawdown") or 0.0)) - abs(max_drawdown_limit)
            candidate_key = (0, -dd_gap, float(score))
        if best is None or candidate_key > best["key"]:
            best = {"key": candidate_key, "params": params, "simulated": simulated}

    if best is None:
        raise RuntimeError("Portfolio optimization failed: no valid candidate portfolios were generated.")

    best_simulated = best["simulated"]
    best_params = dict(best["params"])
    full_simulated = simulate_portfolio_backtest(
        db_path,
        predictions,
        resolved_version,
        transaction_bps=transaction_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        execution_price_model=execution_price_model,
        **best_params,
    )
    final_holdout_metrics: dict[str, Any] | None = None
    if not final_holdout_predictions.empty:
        final_simulated = simulate_portfolio_backtest(
            db_path,
            final_holdout_predictions,
            resolved_version,
            transaction_bps=transaction_bps,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
            execution_price_model=execution_price_model,
            **best_params,
        )
        final_holdout_metrics = final_simulated["metrics"]
        final_simulated["results"].to_csv(artifact_dir / "final_holdout_results.csv", index=False)
    best_metrics = full_simulated["metrics"]
    best_metrics.update(
        {
            "optimization_objective": "Maximize out-of-sample Sharpe subject to max drawdown <= limit.",
            "max_drawdown_limit": max_drawdown_limit,
            "excluded_tickers": excluded,
            "min_avg_dollar_volume": min_avg_dollar_volume,
            "optimized": True,
            "optimization_window_start": str(pd.to_datetime(optimization_predictions["date"]).min().date()),
            "optimization_window_end": str(pd.to_datetime(optimization_predictions["date"]).max().date()),
            "final_holdout_window_start": None
            if final_holdout_predictions.empty
            else str(pd.to_datetime(final_holdout_predictions["date"]).min().date()),
            "final_holdout_window_end": None
            if final_holdout_predictions.empty
            else str(pd.to_datetime(final_holdout_predictions["date"]).max().date()),
            "final_holdout_metrics": final_holdout_metrics,
        }
    )
    best_metrics.update(best_params)

    report = pd.DataFrame(rows)
    report = report.sort_values(["feasible", "sharpe", "total_return"], ascending=[False, False, False])
    report.to_csv(artifact_dir / "optimization_report.csv", index=False)
    (artifact_dir / "optimization_summary.json").write_text(
        json.dumps(
            {
                "model_version": resolved_version,
                "candidate_count": len(candidates),
                "feasible_count": int(report["feasible"].fillna(False).sum()) if "feasible" in report else 0,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "optimization_metrics": best_simulated["metrics"],
                "final_holdout_metrics": final_holdout_metrics,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_outputs(
        artifact_dir,
        full_simulated["results"],
        full_simulated["trades"],
        full_simulated["positions"],
        best_metrics,
        orders=full_simulated["orders"],
        fills=full_simulated["fills"],
        current_positions=full_simulated["current_positions"],
    )
    write_optimized_weights(artifact_dir, full_simulated["positions"], best_params, best_metrics)
    write_backtest_tables(
        db_path,
        resolved_version,
        full_simulated["signals"],
        full_simulated["positions"],
        full_simulated["results"],
        full_simulated["trades"],
        full_simulated["orders"],
        full_simulated["fills"],
        full_simulated["current_positions"],
    )
    return OptimizationResult(
        model_version=resolved_version,
        candidate_count=len(candidates),
        feasible_count=int(report["feasible"].fillna(False).sum()) if "feasible" in report else 0,
        best_params=best_params,
        best_metrics=best_metrics,
        artifact_dir=str(artifact_dir),
    )


def simulate_portfolio_backtest(
    db_path: str | Path,
    predictions: pd.DataFrame,
    model_version: str,
    top_pct: float = 0.10,
    max_weight: float = 0.05,
    min_weight: float = 0.005,
    rebalance_days: int = 5,
    transaction_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
    target_exposure: float = 1.0,
    return_pivot: pd.DataFrame | None = None,
    price_panel: pd.DataFrame | None = None,
    execution_price_model: str = "open",
) -> dict[str, Any]:
    signals = build_signals(
        predictions,
        model_version,
        top_pct=top_pct,
        max_weight=max_weight,
        min_weight=min_weight,
        target_exposure=target_exposure,
    )
    positions = rebalance_positions(signals, rebalance_days=rebalance_days)
    if positions.empty:
        raise RuntimeError("No portfolio positions were generated.")

    start_date = pd.to_datetime(predictions["date"]).min().date()
    end_date = pd.to_datetime(predictions["date"]).max().date()
    if price_panel is None:
        tickers = sorted(set(positions["ticker"].tolist()) | {"SPY"})
        price_panel = load_execution_price_panel(db_path, tickers, start_date, end_date)
    if return_pivot is None:
        return_pivot = price_panel.pivot(index="date", columns="ticker", values="close_return").sort_index().fillna(0.0)
    price_lookup = build_execution_price_lookup(price_panel)
    dates = [d for d in return_pivot.index if start_date <= d.date() <= end_date]
    if not dates:
        raise RuntimeError("No overlapping return dates found for backtest.")

    position_map = {
        pd.to_datetime(day).date(): group.set_index("ticker")["target_weight"].astype(float).to_dict()
        for day, group in positions.groupby("date")
    }
    cost_rate = (transaction_bps + slippage_bps) / 10000.0
    weights: dict[str, float] = {}
    shares: dict[str, float] = {}
    pending_target: dict[str, float] | None = None
    pending_signal_date: Any | None = None
    value = float(initial_capital)
    benchmark_value = float(initial_capital)
    peak = value
    result_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    current_position_rows: list[dict[str, Any]] = []

    for idx, ts in enumerate(dates):
        day = ts.date()
        prev_value = value
        row_returns = return_pivot.loc[ts]
        gross_strategy_return = sum(weight * float(row_returns.get(ticker, 0.0)) for ticker, weight in weights.items())
        if idx == 0:
            gross_strategy_return = 0.0
        value_after_return = value * (1.0 + gross_strategy_return)

        benchmark_return = float(row_returns.get("SPY", 0.0))
        if idx == 0:
            benchmark_return = 0.0
        benchmark_value *= 1.0 + benchmark_return

        turnover = 0.0
        cost = 0.0
        if pending_target is not None and pending_signal_date != day:
            target = pending_target
            all_tickers = sorted(set(weights) | set(target))
            turnover = sum(abs(target.get(ticker, 0.0) - weights.get(ticker, 0.0)) for ticker in all_tickers)
            cost = value_after_return * turnover * cost_rate
            for ticker in all_tickers:
                from_weight = weights.get(ticker, 0.0)
                to_weight = target.get(ticker, 0.0)
                trade_weight = to_weight - from_weight
                if abs(trade_weight) > 1e-10:
                    order_id = f"{model_version}_{pending_signal_date}_{day}_{ticker}"
                    side = "BUY" if trade_weight > 0 else "SELL"
                    fill_price = lookup_execution_price(price_lookup, day, ticker, execution_price_model)
                    fill_value = value_after_return * abs(trade_weight)
                    fill_shares = fill_value / fill_price if fill_price and fill_price > 0 else 0.0
                    if side == "SELL":
                        fill_shares *= -1.0
                    shares[ticker] = shares.get(ticker, 0.0) + fill_shares
                    order_rows.append(
                        {
                            "order_id": order_id,
                            "signal_date": pending_signal_date,
                            "order_date": day,
                            "ticker": ticker,
                            "side": side,
                            "target_weight": float(to_weight),
                            "current_weight": float(from_weight),
                            "order_weight": float(trade_weight),
                            "estimated_order_value": float(value_after_return * trade_weight),
                            "model_version": model_version,
                            "status": "filled",
                        }
                    )
                    fill_rows.append(
                        {
                            "fill_id": f"{order_id}_fill",
                            "order_id": order_id,
                            "fill_date": day,
                            "ticker": ticker,
                            "side": side,
                            "fill_price": float(fill_price),
                            "fill_shares": float(fill_shares),
                            "fill_value": float(fill_value if side == "BUY" else -fill_value),
                            "transaction_cost": float(value_after_return * abs(trade_weight) * cost_rate),
                            "execution_price_model": execution_price_model,
                            "model_version": model_version,
                        }
                    )
                    trade_rows.append(
                        {
                            "date": day,
                            "ticker": ticker,
                            "from_weight": float(from_weight),
                            "to_weight": float(to_weight),
                            "trade_weight": float(trade_weight),
                            "trade_value": float(value_after_return * trade_weight),
                            "transaction_cost": float(value_after_return * abs(trade_weight) * cost_rate),
                            "model_version": model_version,
                        }
                    )
            weights = target.copy()
            pending_target = None
            pending_signal_date = None

        if day in position_map:
            pending_target = position_map[day]
            pending_signal_date = day

        value = value_after_return - cost
        for ticker, weight in weights.items():
            market_price = lookup_execution_price(price_lookup, day, ticker, "close")
            current_position_rows.append(
                {
                    "as_of_date": day,
                    "ticker": ticker,
                    "shares": float(shares.get(ticker, 0.0)),
                    "market_price": float(market_price),
                    "market_value": float(value * weight),
                    "target_weight": float(weight),
                    "current_weight": float(weight),
                    "unrealized_pnl": 0.0,
                    "model_version": model_version,
                }
            )
        peak = max(peak, value)
        drawdown = value / peak - 1.0 if peak else 0.0
        result_rows.append(
            {
                "date": day,
                "portfolio_value": float(value),
                "strategy_return": float(value / prev_value - 1.0 if prev_value else 0.0),
                "benchmark_value": float(benchmark_value),
                "benchmark_return": benchmark_return,
                "drawdown": float(drawdown),
                "gross_exposure": float(sum(abs(weight) for weight in weights.values())),
                "net_exposure": float(sum(weights.values())),
                "turnover": float(turnover),
                "model_version": model_version,
            }
        )

    result = pd.DataFrame(result_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = compute_backtest_metrics(result)
    metrics.update(
        {
            "top_pct": top_pct,
            "max_weight": max_weight,
            "min_weight": min_weight,
            "target_exposure": target_exposure,
            "rebalance_days": rebalance_days,
            "transaction_bps": transaction_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "weighting_method": "score_weighted_with_single_name_cap_and_renormalization",
            "execution_assumption": "Signals rebalance after the close; new weights earn returns from the next trading day.",
            "execution_price_model": execution_price_model,
        }
    )
    return {
        "signals": signals,
        "positions": positions,
        "results": result,
        "trades": trades,
        "orders": pd.DataFrame(order_rows),
        "fills": pd.DataFrame(fill_rows),
        "current_positions": pd.DataFrame(current_position_rows),
        "metrics": metrics,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }


def compute_backtest_metrics(result: pd.DataFrame) -> dict[str, Any]:
    if result.empty:
        return {}
    days = max(1, len(result))
    total_return = result["portfolio_value"].iloc[-1] / result["portfolio_value"].iloc[0] - 1.0
    benchmark_return = result["benchmark_value"].iloc[-1] / result["benchmark_value"].iloc[0] - 1.0
    daily = result["strategy_return"].fillna(0.0)
    annual_vol = float(daily.std() * sqrt(252)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / daily.std() * sqrt(252)) if len(daily) > 1 and daily.std() > 0 else None
    cagr = float((1.0 + total_return) ** (252.0 / days) - 1.0)
    benchmark_cagr = float((1.0 + benchmark_return) ** (252.0 / days) - 1.0)
    return {
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(total_return - benchmark_return),
        "cagr": cagr,
        "benchmark_cagr": benchmark_cagr,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(result["drawdown"].min()),
        "avg_turnover": float(result["turnover"].mean()),
        "win_rate": float((daily > 0).mean()),
        "final_portfolio_value": float(result["portfolio_value"].iloc[-1]),
        "final_benchmark_value": float(result["benchmark_value"].iloc[-1]),
    }


def write_outputs(
    artifact_dir: Path,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: dict[str, Any],
    orders: pd.DataFrame | None = None,
    fills: pd.DataFrame | None = None,
    current_positions: pd.DataFrame | None = None,
) -> None:
    result.to_csv(artifact_dir / "backtest_results.csv", index=False)
    trades.to_csv(artifact_dir / "backtest_trades.csv", index=False)
    positions.to_csv(artifact_dir / "portfolio_positions.csv", index=False)
    if orders is not None:
        orders.to_csv(artifact_dir / "execution_orders.csv", index=False)
    if fills is not None:
        fills.to_csv(artifact_dir / "execution_fills.csv", index=False)
    if current_positions is not None:
        current_positions.to_csv(artifact_dir / "current_positions.csv", index=False)
        write_latest_current_positions(artifact_dir, current_positions)
    (artifact_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_latest_holdings(artifact_dir, positions)
    write_backtest_plot(artifact_dir, result, metrics)


def write_latest_holdings(artifact_dir: Path, positions: pd.DataFrame) -> None:
    if positions.empty:
        return
    latest_date = pd.to_datetime(positions["date"]).max()
    latest = positions[pd.to_datetime(positions["date"]) == latest_date].sort_values("target_weight", ascending=False)
    summary = {
        "latest_rebalance_date": str(latest_date.date()),
        "holding_count": int(len(latest)),
        "invested_weight": float(latest["target_weight"].sum()),
        "cash_weight": float(max(0.0, 1.0 - latest["target_weight"].sum())),
        "holdings": [
            {"ticker": row["ticker"], "target_weight": float(row["target_weight"])}
            for _, row in latest.head(50).iterrows()
        ],
    }
    latest.to_csv(artifact_dir / "latest_holdings.csv", index=False)
    (artifact_dir / "latest_holdings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_optimized_weights(
    artifact_dir: Path,
    positions: pd.DataFrame,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if positions.empty:
        return
    latest_date = pd.to_datetime(positions["date"]).max()
    latest = positions[pd.to_datetime(positions["date"]) == latest_date].sort_values("target_weight", ascending=False).copy()
    invested = float(latest["target_weight"].sum())
    latest["target_weight_pct"] = latest["target_weight"].astype(float) * 100.0
    latest.to_csv(artifact_dir / "optimized_weights.csv", index=False)
    summary = {
        "latest_rebalance_date": str(latest_date.date()),
        "holding_count": int(len(latest)),
        "invested_weight": invested,
        "cash_weight": float(max(0.0, 1.0 - invested)),
        "best_params": params,
        "best_metrics": metrics,
        "weights": [
            {
                "ticker": row["ticker"],
                "target_weight": float(row["target_weight"]),
                "target_weight_pct": float(row["target_weight_pct"]),
            }
            for _, row in latest.iterrows()
        ],
    }
    (artifact_dir / "optimized_weights.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def write_latest_current_positions(artifact_dir: Path, current_positions: pd.DataFrame) -> None:
    if current_positions.empty:
        return
    latest_date = pd.to_datetime(current_positions["as_of_date"]).max()
    latest = current_positions[pd.to_datetime(current_positions["as_of_date"]) == latest_date].copy()
    latest = latest.sort_values("market_value", ascending=False)
    latest.to_csv(artifact_dir / "latest_current_positions.csv", index=False)
    summary = {
        "as_of_date": str(latest_date.date()),
        "position_count": int(len(latest)),
        "gross_market_value": float(latest["market_value"].sum()),
        "positions": [
            {
                "ticker": row["ticker"],
                "shares": float(row["shares"]),
                "market_price": float(row["market_price"]),
                "market_value": float(row["market_value"]),
                "current_weight": float(row["current_weight"]),
            }
            for _, row in latest.iterrows()
        ],
    }
    (artifact_dir / "latest_current_positions.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def write_backtest_plot(artifact_dir: Path, result: pd.DataFrame, metrics: dict[str, Any]) -> str | None:
    if result.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    plot_path = artifact_dir / "backtest_report.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=130, sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].plot(result["date"], result["portfolio_value"], label="Strategy", linewidth=2)
    axes[0].plot(result["date"], result["benchmark_value"], label="S&P 500 / SPY", linewidth=1.7)
    axes[0].set_title("Out-of-sample backtest equity curve")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(result["date"], result["drawdown"], 0, color="#D55E00", alpha=0.35)
    axes[1].set_title("Strategy drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)

    axes[2].bar(result["date"], result["turnover"], width=1.0, color="#0072B2", alpha=0.8)
    axes[2].set_title("Rebalance turnover")
    axes[2].set_ylabel("Turnover")
    axes[2].grid(alpha=0.25)

    title = (
        f"Return={metrics.get('total_return', 0):.2%} | "
        f"SPY={metrics.get('benchmark_return', 0):.2%} | "
        f"Sharpe={metrics.get('sharpe') if metrics.get('sharpe') is not None else 'n/a'} | "
        f"Max DD={metrics.get('max_drawdown', 0):.2%}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return str(plot_path)


def write_backtest_tables(
    db_path: str | Path,
    model_version: str,
    signals: pd.DataFrame,
    positions: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    orders: pd.DataFrame | None = None,
    fills: pd.DataFrame | None = None,
    current_positions: pd.DataFrame | None = None,
) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM trading_signals WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM portfolio_positions WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM backtest_results WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM backtest_trades WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM execution_orders WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM execution_fills WHERE model_version = ?", [model_version])
        con.execute("DELETE FROM current_positions WHERE model_version = ?", [model_version])
        con.register("signals_df", signals)
        con.register("positions_df", positions)
        con.register("results_df", result)
        con.register("trades_df", trades)
        con.execute("INSERT INTO trading_signals SELECT *, current_timestamp AS created_at FROM signals_df")
        con.execute("INSERT INTO portfolio_positions SELECT *, current_timestamp AS created_at FROM positions_df")
        con.execute("INSERT INTO backtest_results SELECT *, current_timestamp AS created_at FROM results_df")
        if not trades.empty:
            con.execute("INSERT INTO backtest_trades SELECT *, current_timestamp AS created_at FROM trades_df")
        if orders is not None and not orders.empty:
            con.register("orders_df", orders)
            con.execute("INSERT INTO execution_orders SELECT *, current_timestamp AS created_at FROM orders_df")
            con.unregister("orders_df")
        if fills is not None and not fills.empty:
            con.register("fills_df", fills)
            con.execute("INSERT INTO execution_fills SELECT *, current_timestamp AS created_at FROM fills_df")
            con.unregister("fills_df")
        if current_positions is not None and not current_positions.empty:
            con.register("current_positions_df", current_positions)
            con.execute("INSERT INTO current_positions SELECT *, current_timestamp AS updated_at FROM current_positions_df")
            con.unregister("current_positions_df")
        con.unregister("signals_df")
        con.unregister("positions_df")
        con.unregister("results_df")
        con.unregister("trades_df")
