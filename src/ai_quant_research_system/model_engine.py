from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from math import sqrt
from typing import Any

import pandas as pd

from .database import connect


FEATURE_COLUMNS = [
    "z_momentum_5d",
    "z_momentum_20d",
    "z_momentum_60d",
    "z_volatility_20d",
    "z_volume_change_5d",
    "z_rsi_14",
    "z_market_return_5d",
    "z_sector_return_5d",
    "z_beta",
    "z_news_sentiment",
    "z_news_importance",
    "z_news_surprise",
    "z_news_risk",
    "z_news_revenue_impact",
    "z_news_margin_impact",
    "z_news_ai_score",
]

LABEL_HORIZONS = [1, 5, 10, 20]


@dataclass(frozen=True)
class ModelRunResult:
    model_version: str
    dataset_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    prediction_rows: int
    artifact_dir: str
    metrics: dict[str, Any]


def load_factor_dataset(db_path: str | Path) -> pd.DataFrame:
    with connect(db_path) as con:
        factors = con.execute("SELECT * FROM factor_table").fetchdf()
        prices = con.execute(
            """
            SELECT date, ticker, close
            FROM daily_prices_clean
            WHERE close IS NOT NULL
            """
        ).fetchdf()

    if factors.empty or prices.empty:
        return pd.DataFrame()

    factors["date"] = pd.to_datetime(factors["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    for horizon in LABEL_HORIZONS:
        prices[f"future_close_{horizon}d"] = prices.groupby("ticker")["close"].shift(-horizon)
        prices[f"stock_return_{horizon}d"] = prices[f"future_close_{horizon}d"] / prices["close"] - 1

    spy = prices[prices["ticker"] == "SPY"][["date", "close"]].drop_duplicates("date").sort_values("date")
    for horizon in LABEL_HORIZONS:
        spy[f"future_spy_close_{horizon}d"] = spy["close"].shift(-horizon)
        spy[f"spy_return_{horizon}d"] = spy[f"future_spy_close_{horizon}d"] / spy["close"] - 1

    price_label_cols = ["date", "ticker", *[f"stock_return_{horizon}d" for horizon in LABEL_HORIZONS]]
    spy_label_cols = ["date", *[f"spy_return_{horizon}d" for horizon in LABEL_HORIZONS]]
    labels = prices[price_label_cols].merge(
        spy[spy_label_cols],
        on="date",
        how="left",
    )
    for horizon in LABEL_HORIZONS:
        labels[f"future_excess_return_{horizon}d"] = labels[f"stock_return_{horizon}d"] - labels[f"spy_return_{horizon}d"]

    dataset = factors.merge(
        labels[
            [
                "date",
                "ticker",
                *[f"stock_return_{horizon}d" for horizon in LABEL_HORIZONS],
                *[f"spy_return_{horizon}d" for horizon in LABEL_HORIZONS],
                *[f"future_excess_return_{horizon}d" for horizon in LABEL_HORIZONS],
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    return dataset


def prepare_features(dataset: pd.DataFrame) -> pd.DataFrame:
    prepared = dataset.copy()
    feature_coverage = prepared[FEATURE_COLUMNS].notna().mean(axis=1)
    prepared = prepared[feature_coverage >= 0.80].copy()

    for col in FEATURE_COLUMNS:
        by_date_mean = prepared.groupby("date")[col].transform("mean")
        prepared[col] = prepared[col].fillna(by_date_mean).fillna(0.0)
    return prepared


def save_dataset(dataset: pd.DataFrame, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "dataset.parquet"
    dataset.to_parquet(path, index=False)
    return path


def date_splits(
    dataset: pd.DataFrame,
    validation_pct: float = 0.15,
    test_pct: float = 0.15,
    final_holdout_pct: float = 0.10,
    embargo_days: int = 5,
) -> dict[str, pd.DataFrame]:
    labeled = dataset.dropna(subset=["future_excess_return_5d"]).copy()
    if labeled.empty:
        return {"train": labeled, "validation": labeled, "test": labeled, "final_holdout": labeled}

    unique_dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates())
    n = len(unique_dates)
    if n < 30:
        train_cut = max(1, int(n * 0.60))
        val_cut = max(train_cut + 1, int(n * 0.80))
        val_start = train_cut
        val_end = val_cut
        test_start = val_cut
        test_end = n
        final_start = n
        gap = 0
    else:
        final_size = max(1, int(n * final_holdout_pct))
        test_size = max(1, int(n * test_pct))
        val_size = max(1, int(n * validation_pct))
        final_start = n - final_size
        test_end = max(1, final_start - embargo_days)
        test_start = max(1, test_end - test_size)
        val_end = max(1, test_start - embargo_days)
        val_start = max(1, val_end - val_size)
        train_cut = max(1, val_start - embargo_days)
        gap = embargo_days

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[val_start:val_end])
    test_dates = set(unique_dates[test_start:test_end])
    final_holdout_dates = set(unique_dates[final_start:])

    return {
        "train": labeled[labeled["date"].isin(train_dates)].copy(),
        "validation": labeled[labeled["date"].isin(val_dates)].copy(),
        "test": labeled[labeled["date"].isin(test_dates)].copy(),
        "final_holdout": labeled[labeled["date"].isin(final_holdout_dates)].copy(),
    }


def split_summary(splits: dict[str, pd.DataFrame], embargo_days: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"embargo_trading_days": embargo_days}
    for name, frame in splits.items():
        start, end = date_bounds(frame)
        summary[name] = {
            "rows": int(len(frame)),
            "start": None if start is None else str(start),
            "end": None if end is None else str(end),
            "unique_dates": int(pd.to_datetime(frame["date"]).nunique()) if not frame.empty else 0,
        }
    return summary


def write_split_report(summary: dict[str, Any], artifact_dir: Path) -> Path:
    path = artifact_dir / "oos_split_report.md"
    lines = [
        "# OOS Split Report",
        "",
        "Policy: purged chronological split with final untouched holdout.",
        f"Embargo: {summary.get('embargo_trading_days')} trading days between adjacent segments.",
        "",
        "| Segment | Purpose | Rows | Dates | Unique Dates |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    purposes = {
        "train": "Fit model parameters only.",
        "validation": "Tune ensemble weights / model-selection diagnostics.",
        "test": "Development OOS diagnostics.",
        "final_holdout": "Primary untouched OOS report; not used for tuning.",
    }
    for name in ("train", "validation", "test", "final_holdout"):
        row = summary.get(name, {})
        lines.append(
            f"| {name} | {purposes[name]} | {row.get('rows')} | "
            f"{row.get('start')}..{row.get('end')} | {row.get('unique_dates')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Validation metrics can guide model selection, but they are not final performance.",
            "- Test metrics are useful diagnostics after model selection.",
            "- Final holdout metrics are the strictest single-split OOS estimate in `train-models`.",
            "- For the closest live-trading simulation, use `train-walk-forward-models`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_models(selected_models: list[str] | None = None) -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install model dependencies first: pip install -r requirements.txt") from exc

    selected = set(selected_models or ["lightgbm", "xgboost", "rf"])
    models: dict[str, Any] = {}
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        if "lightgbm" in selected:
            models["lightgbm"] = LGBMRegressor(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=500,
                random_state=42,
            )
    except ModuleNotFoundError:
        if "lightgbm" in selected:
            models["lightgbm"] = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_leaf_nodes=31,
                max_iter=300,
                random_state=42,
            )

    try:
        from xgboost import XGBRegressor  # type: ignore

        if "xgboost" in selected:
            models["xgboost"] = XGBRegressor(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=500,
                objective="reg:squarederror",
                random_state=42,
            )
    except ModuleNotFoundError:
        if "xgboost" in selected:
            models["xgboost"] = GradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=300,
                random_state=42,
            )

    if "rf" in selected:
        models["rf"] = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    if not models:
        raise RuntimeError("No valid models selected. Choose from: lightgbm, xgboost, rf.")
    return models


def rank_ic(pred: pd.Series, actual: pd.Series) -> float | None:
    usable = pred.notna() & actual.notna()
    if usable.sum() < 3 or pred[usable].nunique() < 2 or actual[usable].nunique() < 2:
        return None
    value = pred.rank().corr(actual.rank(), method="pearson")
    return None if pd.isna(value) else float(value)


def pearson_ic(pred: pd.Series, actual: pd.Series) -> float | None:
    usable = pred.notna() & actual.notna()
    if usable.sum() < 3 or pred[usable].nunique() < 2 or actual[usable].nunique() < 2:
        return None
    value = pred.corr(actual, method="pearson")
    return None if pd.isna(value) else float(value)


def hit_rate(pred: pd.Series, actual: pd.Series) -> float | None:
    usable = pred.notna() & actual.notna()
    if usable.sum() == 0:
        return None
    return float(((pred[usable] >= 0) == (actual[usable] >= 0)).mean())


def evaluate_predictions(frame: pd.DataFrame, pred_col: str) -> dict[str, float | None]:
    try:
        from sklearn.metrics import mean_squared_error
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install model dependencies first: pip install -r requirements.txt") from exc

    if frame.empty:
        return {"mse": None, "ic": None, "rank_ic": None, "hit_rate": None}
    actual = frame["future_excess_return_5d"]
    pred = frame[pred_col]
    mse = float(mean_squared_error(actual, pred))
    return {
        "mse": mse,
        "rmse": sqrt(mse),
        "ic": pearson_ic(pred, actual),
        "rank_ic": rank_ic(pred, actual),
        "hit_rate": hit_rate(pred, actual),
    }


def factor_ic_report(dataset: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in LABEL_HORIZONS:
        label = f"future_excess_return_{horizon}d"
        if label not in dataset.columns:
            continue
        for feature in FEATURE_COLUMNS:
            daily_rows = []
            for day, group in dataset.groupby("date"):
                clean = group[[feature, label]].dropna()
                if len(clean) < 10:
                    continue
                daily_rows.append(
                    {
                        "date": day,
                        "ic": pearson_ic(clean[feature], clean[label]),
                        "rank_ic": rank_ic(clean[feature], clean[label]),
                    }
                )
            daily = pd.DataFrame(daily_rows)
            if daily.empty:
                rows.append(
                    {
                        "horizon": horizon,
                        "feature": feature,
                        "mean_ic": None,
                        "mean_rank_ic": None,
                        "ic_ir": None,
                        "positive_rank_ic_rate": None,
                        "observations": 0,
                    }
                )
                continue
            rank_values = daily["rank_ic"].dropna()
            ic_values = daily["ic"].dropna()
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    "mean_ic": None if ic_values.empty else float(ic_values.mean()),
                    "mean_rank_ic": None if rank_values.empty else float(rank_values.mean()),
                    "ic_ir": None if rank_values.std() in (0, None) or pd.isna(rank_values.std()) else float(rank_values.mean() / rank_values.std()),
                    "positive_rank_ic_rate": None if rank_values.empty else float((rank_values > 0).mean()),
                    "observations": int(len(daily)),
                }
            )
    report = pd.DataFrame(rows).sort_values(["horizon", "mean_rank_ic"], ascending=[True, False])
    report.to_csv(artifact_dir / "factor_ic_report.csv", index=False)
    return report


def ensemble_weights_from_validation(
    validation: pd.DataFrame,
    models: dict[str, Any],
) -> dict[str, float]:
    if validation.empty:
        return {name: 1.0 / len(models) for name in models}
    raw_scores: dict[str, float] = {}
    x_val = validation[FEATURE_COLUMNS]
    for name, model in models.items():
        pred = pd.Series(model.predict(x_val), index=validation.index)
        score = rank_ic(pred, validation["future_excess_return_5d"])
        raw_scores[name] = max(0.0, float(score or 0.0))
    total = sum(raw_scores.values())
    if total <= 0:
        return {name: 1.0 / len(models) for name in models}
    return {name: score / total for name, score in raw_scores.items()}


def write_feature_importance_report(
    models: dict[str, Any],
    validation: pd.DataFrame,
    artifact_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            for feature, importance in zip(FEATURE_COLUMNS, importances):
                rows.append({"model": name, "feature": feature, "importance": float(importance), "method": "model_native"})
            continue
        for feature in FEATURE_COLUMNS:
            rows.append({"model": name, "feature": feature, "importance": None, "method": "not_available"})
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["model", "importance"], ascending=[True, False])
    report.to_csv(artifact_dir / "feature_importance.csv", index=False)
    return report


def oos_return_report(frame: pd.DataFrame, selection_quantile: float = 0.10) -> tuple[pd.DataFrame, dict[str, float | None]]:
    if frame.empty:
        return pd.DataFrame(), {
            "long_short_total_return": None,
        "long_only_total_return": None,
        "sp500_total_return": None,
        "long_minus_sp500_total_return": None,
            "short_only_total_return": None,
            "mean_daily_rank_ic": None,
        }

    rows: list[dict[str, Any]] = []
    for day, group in frame.groupby("date"):
        clean = group.dropna(subset=["predicted_excess_return_5d", "future_excess_return_5d"])
        if len(clean) < 10:
            continue
        q = max(1, int(len(clean) * selection_quantile))
        ranked = clean.sort_values("predicted_excess_return_5d", ascending=False)
        top = ranked.head(q)
        bottom = ranked.tail(q)
        long_return = top["stock_return_5d"].mean()
        benchmark_return = clean["spy_return_5d"].dropna().mean()
        rows.append(
            {
                "date": day,
                "long_return": long_return,
                "benchmark_return": benchmark_return,
                "long_excess_return": long_return - benchmark_return,
                "short_return": -bottom["stock_return_5d"].mean(),
                "long_short_return": top["stock_return_5d"].mean() - bottom["stock_return_5d"].mean(),
                "rank_ic": rank_ic(clean["predicted_excess_return_5d"], clean["future_excess_return_5d"]),
                "n": len(clean),
                "selected_count": q,
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report, {
            "long_short_total_return": None,
            "long_only_total_return": None,
            "sp500_total_return": None,
            "long_minus_sp500_total_return": None,
            "short_only_total_return": None,
            "mean_daily_rank_ic": None,
        }
    report = report.sort_values("date")
    report["long_cum_return"] = (1 + report["long_return"].fillna(0)).cumprod() - 1
    report["benchmark_cum_return"] = (1 + report["benchmark_return"].fillna(0)).cumprod() - 1
    report["long_excess_cum_return"] = (1 + report["long_excess_return"].fillna(0)).cumprod() - 1
    report["short_cum_return"] = (1 + report["short_return"].fillna(0)).cumprod() - 1
    report["long_short_cum_return"] = (1 + report["long_short_return"].fillna(0)).cumprod() - 1
    metrics = {
        "long_short_total_return": float(report["long_short_cum_return"].iloc[-1]),
        "long_only_total_return": float(report["long_cum_return"].iloc[-1]),
        "sp500_total_return": float(report["benchmark_cum_return"].iloc[-1]),
        "long_minus_sp500_total_return": float(report["long_cum_return"].iloc[-1] - report["benchmark_cum_return"].iloc[-1]),
        "short_only_total_return": float(report["short_cum_return"].iloc[-1]),
        "mean_daily_rank_ic": None if report["rank_ic"].dropna().empty else float(report["rank_ic"].mean()),
        "selection_quantile": selection_quantile,
    }
    return report, metrics


def write_stock_selection_guide(
    scored: pd.DataFrame,
    artifact_dir: Path,
    top_n: int = 30,
    selection_quantile: float = 0.10,
) -> dict[str, str]:
    latest_date = pd.to_datetime(scored["date"]).max()
    latest = scored[pd.to_datetime(scored["date"]) == latest_date].copy()
    latest = latest.sort_values("predicted_excess_return_5d", ascending=False)
    selected_count = max(1, int(len(latest) * selection_quantile))
    equal_weight = 1.0 / selected_count
    columns = ["date", "ticker", "predicted_excess_return_5d", "lightgbm_pred", "xgboost_pred", "rf_pred"]
    picks = latest.head(top_n)[columns]
    avoids = latest.tail(top_n)[columns].sort_values("predicted_excess_return_5d")
    picks_path = artifact_dir / "stock_selection_guide.csv"
    md_path = artifact_dir / "stock_selection_guide.md"
    summary_path = artifact_dir / "selection_summary.json"
    combined = pd.concat(
        [picks.assign(action="candidate_long"), avoids.assign(action="candidate_avoid_or_short")],
        ignore_index=True,
    )
    combined.to_csv(picks_path, index=False)
    lines = [
        "# Stock Selection Guide",
        "",
        f"Latest factor date: {latest_date.date()}",
        f"Buy ratio: top {selection_quantile:.0%} of ranked stocks",
        f"Approximate holdings: {selected_count}",
        f"Equal-weight allocation per holding: {equal_weight:.2%}",
        "",
        "This guide ranks stocks by predicted 5-day excess return. It is a research output, not trading advice.",
        "",
        "## Top long candidates",
        "",
    ]
    for _, row in picks.iterrows():
        lines.append(f"- {row['ticker']}: predicted excess return {row['predicted_excess_return_5d']:.4f}")
    lines.extend(["", "## Avoid / short candidates", ""])
    for _, row in avoids.iterrows():
        lines.append(f"- {row['ticker']}: predicted excess return {row['predicted_excess_return_5d']:.4f}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "latest_date": str(latest_date.date()),
        "buy_ratio": selection_quantile,
        "selected_count": selected_count,
        "equal_weight_per_holding": equal_weight,
        "top_tickers": picks["ticker"].head(12).tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"csv": str(picks_path), "markdown": str(md_path), "summary": str(summary_path)}


def write_oos_plot(report: pd.DataFrame, metrics: dict[str, Any], artifact_dir: Path) -> str | None:
    if report.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    plot_path = artifact_dir / "out_of_sample_report.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=140, sharex=True)
    axes[0].plot(report["date"], report["long_short_cum_return"], label="Long-short", linewidth=2)
    axes[0].plot(report["date"], report["long_cum_return"], label="Long top decile", linewidth=1.5)
    axes[0].plot(report["date"], report["benchmark_cum_return"], label="S&P 500 / SPY", linewidth=1.5)
    axes[0].plot(report["date"], report["short_cum_return"], label="Short bottom decile", linewidth=1.5)
    axes[0].axhline(0, color="#8a8f98", linewidth=0.8)
    axes[0].set_title("Out-of-sample cumulative return proxy")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].bar(report["date"], report["rank_ic"].fillna(0), width=1.0, color="#4C78A8")
    axes[1].axhline(0, color="#8a8f98", linewidth=0.8)
    axes[1].set_title("Daily Rank IC")
    axes[1].set_ylabel("Rank IC")
    axes[1].grid(alpha=0.25)

    summary = (
        f"Test IC={metrics.get('ic'):.4f} | "
        f"Rank IC={metrics.get('rank_ic'):.4f} | "
        f"RMSE={metrics.get('rmse'):.4f} | "
        f"Hit Rate={metrics.get('hit_rate'):.2%}"
    )
    fig.suptitle(summary, fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return str(plot_path)


def fit_models(
    splits: dict[str, pd.DataFrame],
    artifact_dir: Path,
    selected_models: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install model dependencies first: pip install -r requirements.txt") from exc

    train = splits["train"]
    if train.empty:
        raise RuntimeError("No labeled training rows are available. Need at least 5 future trading days for labels.")

    models = make_models(selected_models)
    x_train = train[FEATURE_COLUMNS]
    y_train = train["future_excess_return_5d"]
    metrics: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump(model, artifact_dir / f"{name}.joblib")
        metrics[name] = {"params": getattr(model, "get_params", lambda: {})()}

    return models, metrics


def add_model_predictions(
    dataset: pd.DataFrame,
    models: dict[str, Any],
    ensemble_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored = dataset.copy()
    x_all = scored[FEATURE_COLUMNS]
    prediction_cols: list[str] = []
    weighted_predictions = []
    for name, col in (("lightgbm", "lightgbm_pred"), ("xgboost", "xgboost_pred"), ("rf", "rf_pred")):
        if name in models:
            scored[col] = models[name].predict(x_all)
            prediction_cols.append(col)
            weight = (ensemble_weights or {}).get(name)
            if weight is not None:
                weighted_predictions.append(scored[col].astype(float) * float(weight))
        else:
            scored[col] = pd.NA
    if weighted_predictions:
        scored["predicted_excess_return_5d"] = sum(weighted_predictions)
    else:
        scored["predicted_excess_return_5d"] = scored[prediction_cols].mean(axis=1)
    return scored


def date_bounds(frame: pd.DataFrame) -> tuple[Any, Any]:
    if frame.empty:
        return None, None
    dates = pd.to_datetime(frame["date"])
    return dates.min().date(), dates.max().date()


def write_predictions(db_path: str | Path, predictions: pd.DataFrame, model_version: str) -> int:
    output = predictions[
        ["date", "ticker", "predicted_excess_return_5d", "lightgbm_pred", "xgboost_pred", "rf_pred"]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.date
    output["model_version"] = model_version
    output = output[
        ["date", "ticker", "predicted_excess_return_5d", "model_version", "lightgbm_pred", "xgboost_pred", "rf_pred"]
    ]

    with connect(db_path) as con:
        con.register("prediction_df", output)
        con.execute("DELETE FROM model_predictions WHERE model_version = ?", [model_version])
        con.execute(
            """
            INSERT INTO model_predictions
            SELECT *, current_timestamp AS predicted_at
            FROM prediction_df
            """
        )
        con.unregister("prediction_df")
    return len(output)


def write_registry(
    db_path: str | Path,
    model_version: str,
    splits: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> None:
    train_start, train_end = date_bounds(splits["train"])
    val_start, val_end = date_bounds(splits["validation"])
    test_start, test_end = date_bounds(splits["test"])
    with connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO model_registry
            (model_version, trained_at, train_start, train_end, validation_start, validation_end,
             test_start, test_end, feature_columns, model_params_json, metrics_json, artifact_dir, notes)
            VALUES (?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_version,
                train_start,
                train_end,
                val_start,
                val_end,
                test_start,
                test_end,
                json.dumps(FEATURE_COLUMNS),
                json.dumps({k: v.get("params", {}) for k, v in metrics.items()}, default=str),
                json.dumps(metrics, default=str),
                str(artifact_dir),
                "Purged chronological split with validation, development test, and untouched final holdout. LightGBM/XGBoost fall back to sklearn models if optional packages are unavailable.",
            ],
        )


def train_prediction_models(
    db_path: str | Path,
    artifact_root: str | Path,
    selected_models: list[str] | None = None,
) -> ModelRunResult:
    model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_dir = Path(artifact_root) / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_features(load_factor_dataset(db_path))
    if dataset.empty:
        raise RuntimeError("factor_table is empty or has insufficient feature coverage. Run build-factor-table first.")
    save_dataset(dataset, artifact_dir)
    factor_ic_report(dataset, artifact_dir)

    splits = date_splits(dataset)
    metrics: dict[str, Any]
    models, metrics = fit_models(splits, artifact_dir, selected_models=selected_models)
    metrics["split_policy"] = {
        "type": "purged_chronological_holdout",
        "validation_usage": "Validation is used only for ensemble weighting and model selection diagnostics.",
        "test_usage": "Test is a development OOS segment for diagnostics.",
        "final_holdout_usage": "Final holdout is untouched by tuning and is the primary reported OOS segment.",
        "label_horizon_days": 5,
        "embargo_trading_days": 5,
    }
    metrics["split_summary"] = split_summary(splits, embargo_days=5)
    (artifact_dir / "split_summary.json").write_text(json.dumps(metrics["split_summary"], indent=2), encoding="utf-8")
    split_report_path = write_split_report(metrics["split_summary"], artifact_dir)
    ensemble_weights = ensemble_weights_from_validation(splits["validation"], models)
    metrics["ensemble_weights"] = ensemble_weights
    write_feature_importance_report(models, splits["validation"], artifact_dir)
    (artifact_dir / "ensemble_weights.json").write_text(json.dumps(ensemble_weights, indent=2), encoding="utf-8")
    scored = add_model_predictions(dataset, models, ensemble_weights=ensemble_weights)

    for split_name, split in splits.items():
        if split.empty:
            metrics[f"{split_name}_ensemble"] = {"mse": None, "rmse": None, "ic": None, "rank_ic": None, "hit_rate": None}
            continue
        split_scored = scored.loc[split.index]
        metrics[f"{split_name}_ensemble"] = evaluate_predictions(split_scored, "predicted_excess_return_5d")

    final_holdout_scored = scored.loc[splits["final_holdout"].index] if not splits["final_holdout"].empty else pd.DataFrame()
    test_scored = scored.loc[splits["test"].index] if not splits["test"].empty else pd.DataFrame()
    oos_source = final_holdout_scored if not final_holdout_scored.empty else test_scored
    oos_report, return_metrics = oos_return_report(oos_source)
    if not oos_report.empty:
        oos_report.to_csv(artifact_dir / "out_of_sample_returns.csv", index=False)
    metrics["final_holdout_return_proxy"] = return_metrics if not final_holdout_scored.empty else {}
    metrics["test_return_proxy"] = oos_return_report(test_scored)[1] if not test_scored.empty else {}
    primary_oos_metrics = metrics.get("final_holdout_ensemble") or metrics.get("test_ensemble", {})
    plot_path = write_oos_plot(oos_report, primary_oos_metrics, artifact_dir)
    guide_paths = write_stock_selection_guide(scored, artifact_dir)
    metrics["artifacts"] = {
        "dataset": str(artifact_dir / "dataset.parquet"),
        "out_of_sample_plot": plot_path,
        "out_of_sample_returns": str(artifact_dir / "out_of_sample_returns.csv") if not oos_report.empty else None,
        "stock_selection_guide_csv": guide_paths["csv"],
        "stock_selection_guide_markdown": guide_paths["markdown"],
        "factor_ic_report": str(artifact_dir / "factor_ic_report.csv"),
        "feature_importance": str(artifact_dir / "feature_importance.csv"),
        "ensemble_weights": str(artifact_dir / "ensemble_weights.json"),
        "split_summary": str(artifact_dir / "split_summary.json"),
        "oos_split_report": str(split_report_path),
    }

    prediction_rows = write_predictions(db_path, scored, model_version)
    write_registry(db_path, model_version, splits, metrics, artifact_dir)

    return ModelRunResult(
        model_version=model_version,
        dataset_rows=len(dataset),
        train_rows=len(splits["train"]),
        validation_rows=len(splits["validation"]),
        test_rows=len(splits["test"]),
        prediction_rows=prediction_rows,
        artifact_dir=str(artifact_dir),
        metrics=metrics,
    )


def train_walk_forward_models(
    db_path: str | Path,
    artifact_root: str | Path,
    selected_models: list[str] | None = None,
    min_train_days: int = 252,
    step_days: int = 21,
    final_holdout_pct: float = 0.20,
) -> ModelRunResult:
    model_version = f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_dir = Path(artifact_root) / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_features(load_factor_dataset(db_path))
    if dataset.empty:
        raise RuntimeError("factor_table is empty or has insufficient feature coverage. Run build-factor-table first.")
    labeled = dataset.dropna(subset=["future_excess_return_5d"]).copy()
    if labeled.empty:
        raise RuntimeError("No labeled rows are available for walk-forward training.")
    save_dataset(dataset, artifact_dir)

    dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates())
    min_train_days = min(max(30, min_train_days), max(1, len(dates) - 2))
    step_days = max(1, step_days)
    holdout_cut = max(min_train_days + 1, int(len(dates) * (1.0 - final_holdout_pct)))
    holdout_dates = set(dates[holdout_cut:]) if holdout_cut < len(dates) else set()

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    cursor = min_train_days
    fold = 1
    while cursor < len(dates):
        train_dates = set(dates[:cursor])
        predict_dates = set(dates[cursor : min(cursor + step_days, len(dates))])
        train = labeled[pd.to_datetime(labeled["date"]).isin(train_dates)].copy()
        predict = dataset[pd.to_datetime(dataset["date"]).isin(predict_dates)].copy()
        if train.empty or predict.empty:
            cursor += step_days
            continue
        models, _ = fit_models({"train": train}, artifact_dir, selected_models=selected_models)
        train_dates_sorted = sorted(pd.to_datetime(train["date"]).drop_duplicates())
        val_tail = set(train_dates_sorted[-min(21, len(train_dates_sorted)) :])
        validation_tail = train[pd.to_datetime(train["date"]).isin(val_tail)].copy()
        ensemble_weights = ensemble_weights_from_validation(validation_tail, models)
        scored = add_model_predictions(predict, models, ensemble_weights=ensemble_weights)
        scored["walk_forward_fold"] = fold
        prediction_frames.append(scored)
        fold_rows.append(
            {
                "fold": fold,
                "train_start": str(min(train_dates).date()),
                "train_end": str(max(train_dates).date()),
                "predict_start": str(min(predict_dates).date()),
                "predict_end": str(max(predict_dates).date()),
                "train_rows": len(train),
                "prediction_rows": len(scored),
                "ensemble_weights": json.dumps(ensemble_weights),
            }
        )
        cursor += step_days
        fold += 1

    if not prediction_frames:
        raise RuntimeError("Walk-forward training produced no prediction folds.")
    scored_all = pd.concat(prediction_frames, ignore_index=True).sort_values(["date", "ticker"])
    scored_all.to_csv(artifact_dir / "walk_forward_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(artifact_dir / "walk_forward_folds.csv", index=False)

    test_scored = scored_all.dropna(subset=["future_excess_return_5d"]).copy()
    final_holdout = test_scored[pd.to_datetime(test_scored["date"]).isin(holdout_dates)].copy() if holdout_dates else pd.DataFrame()
    metrics: dict[str, Any] = {
        "walk_forward_ensemble": evaluate_predictions(test_scored, "predicted_excess_return_5d"),
        "final_holdout_ensemble": evaluate_predictions(final_holdout, "predicted_excess_return_5d") if not final_holdout.empty else {},
        "walk_forward_folds": fold_rows,
        "final_holdout_start": None if not holdout_dates else str(min(holdout_dates).date()),
        "final_holdout_end": None if not holdout_dates else str(max(holdout_dates).date()),
    }
    oos_report, return_metrics = oos_return_report(final_holdout if not final_holdout.empty else test_scored)
    if not oos_report.empty:
        oos_report.to_csv(artifact_dir / "walk_forward_oos_returns.csv", index=False)
    metrics["walk_forward_return_proxy"] = return_metrics
    plot_path = write_oos_plot(oos_report, metrics.get("final_holdout_ensemble") or metrics["walk_forward_ensemble"], artifact_dir)
    metrics["artifacts"] = {
        "dataset": str(artifact_dir / "dataset.parquet"),
        "out_of_sample_plot": plot_path,
        "walk_forward_predictions": str(artifact_dir / "walk_forward_predictions.csv"),
        "walk_forward_folds": str(artifact_dir / "walk_forward_folds.csv"),
    }

    prediction_rows = write_predictions(db_path, scored_all, model_version)
    splits = {
        "train": labeled[pd.to_datetime(labeled["date"]).isin(set(dates[:min_train_days]))],
        "validation": test_scored[~pd.to_datetime(test_scored["date"]).isin(holdout_dates)] if holdout_dates else pd.DataFrame(),
        "test": final_holdout if not final_holdout.empty else test_scored,
    }
    write_registry(db_path, model_version, splits, metrics, artifact_dir)

    return ModelRunResult(
        model_version=model_version,
        dataset_rows=len(dataset),
        train_rows=len(splits["train"]),
        validation_rows=len(splits["validation"]),
        test_rows=len(splits["test"]),
        prediction_rows=prediction_rows,
        artifact_dir=str(artifact_dir),
        metrics=metrics,
    )
