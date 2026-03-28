from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_SCHEDULE = {
    "jobs": ["2026-04-03"],
    "cpi": ["2026-04-10"],
    "ppi": ["2026-04-14"],
}

STOOQ_SYMBOLS = ("soxx.us", "igv.us", "qqq.us", "spy.us")
FRED_SERIES = ("DGS10", "DGS2", "VIXCLS")


@dataclass(slots=True)
class RunConfig:
    horizon: int = 15
    n_splits: int = 6
    quote_width: float = 0.35
    output_dir: Path = Path("outputs")
    use_xgboost: bool = False
    top_k_ensemble: int = 2


def load_stooq_close(symbol: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    frame = pd.read_csv(url)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"Unexpected Stooq schema for {symbol}: {frame.columns.tolist()}")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    return frame["Close"].astype(float).rename(symbol.upper())


def load_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(url)
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    if date_column not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"Unexpected FRED schema for {series_id}: {frame.columns.tolist()}")
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame.set_index(date_column)[series_id].rename(series_id)


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods)


def safe_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
    covariance = asset_returns.rolling(window).cov(benchmark_returns)
    variance = benchmark_returns.rolling(window).var().replace(0.0, np.nan)
    return covariance / variance


def build_event_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    flags = pd.DataFrame(index=index)
    normalized_index = index.normalize()
    for event_name, dates in EVENT_SCHEDULE.items():
        event_days = pd.to_datetime(dates)
        window_days: set[pd.Timestamp] = set()
        for event_day in event_days:
            for day in pd.bdate_range(event_day - pd.Timedelta(days=3), event_day + pd.Timedelta(days=3)):
                window_days.add(pd.Timestamp(day))
        flags[f"{event_name}_day"] = normalized_index.isin(event_days).astype(int)
        flags[f"{event_name}_window"] = normalized_index.isin(pd.DatetimeIndex(sorted(window_days))).astype(int)
    window_columns = [column for column in flags.columns if column.endswith("_window")]
    flags["macro_event_window"] = flags[window_columns].max(axis=1).astype(int)
    return flags


def load_market_data() -> pd.DataFrame:
    series = [load_stooq_close(symbol) for symbol in STOOQ_SYMBOLS]
    prices = pd.concat(series, axis=1).dropna()
    for series_id in FRED_SERIES:
        prices = prices.join(load_fred_series(series_id), how="left")
    return prices.ffill()


def build_feature_frame(config: RunConfig) -> pd.DataFrame:
    prices = load_market_data()
    prices["S"] = prices["SOXX.US"] / 4.0 - prices["IGV.US"]

    returns = pd.DataFrame(index=prices.index)
    for column in ("SOXX.US", "IGV.US", "QQQ.US", "SPY.US"):
        returns[column] = pct_change(prices[column], 1)

    features = pd.DataFrame(index=prices.index)
    features["S"] = prices["S"]
    for window in (1, 3, 5, 10, 20):
        features[f"S_diff_{window}"] = prices["S"].diff(window)
    for window in (10, 20, 60):
        features[f"S_z{window}"] = safe_zscore(prices["S"], window)
        features[f"spread_vol_{window}"] = prices["S"].diff().rolling(window).std()

    features["spread_vol_ratio_20_60"] = features["spread_vol_20"] / features["spread_vol_60"].replace(0.0, np.nan)
    features["spread_range_20"] = prices["S"].rolling(20).max() - prices["S"].rolling(20).min()
    features["spread_trend_20"] = prices["S"].diff(20)

    for window in (1, 5, 10, 20):
        features[f"SOXX_ret_{window}"] = pct_change(prices["SOXX.US"], window)
        features[f"IGV_ret_{window}"] = pct_change(prices["IGV.US"], window)
        features[f"QQQ_ret_{window}"] = pct_change(prices["QQQ.US"], window)
        features[f"SPY_ret_{window}"] = pct_change(prices["SPY.US"], window)
        features[f"rel_ret_{window}"] = features[f"SOXX_ret_{window}"] - features[f"IGV_ret_{window}"]
        features[f"soxx_excess_qqq_{window}"] = features[f"SOXX_ret_{window}"] - features[f"QQQ_ret_{window}"]
        features[f"igv_excess_qqq_{window}"] = features[f"IGV_ret_{window}"] - features[f"QQQ_ret_{window}"]

    features["corr_20"] = returns["SOXX.US"].rolling(20).corr(returns["IGV.US"])
    features["corr_60"] = returns["SOXX.US"].rolling(60).corr(returns["IGV.US"])
    features["corr_gap_20_60"] = features["corr_20"] - features["corr_60"]
    features["soxx_beta_qqq_20"] = rolling_beta(returns["SOXX.US"], returns["QQQ.US"], 20)
    features["igv_beta_qqq_20"] = rolling_beta(returns["IGV.US"], returns["QQQ.US"], 20)
    features["beta_gap_qqq_20"] = features["soxx_beta_qqq_20"] - features["igv_beta_qqq_20"]

    features["DGS10"] = prices["DGS10"]
    features["DGS2"] = prices["DGS2"]
    features["curve_slope"] = prices["DGS10"] - prices["DGS2"]
    features["curve_slope_chg_5"] = features["curve_slope"].diff(5)
    features["dgs10_chg_1"] = prices["DGS10"].diff(1)
    features["dgs10_chg_5"] = prices["DGS10"].diff(5)
    features["dgs2_chg_1"] = prices["DGS2"].diff(1)
    features["dgs2_chg_5"] = prices["DGS2"].diff(5)
    features["vix_level"] = prices["VIXCLS"]
    features["vix_chg_1"] = prices["VIXCLS"].diff(1)
    features["vix_z20"] = safe_zscore(prices["VIXCLS"], 20)

    features = features.join(build_event_flags(features.index))
    features["break_flag"] = (
        (features["spread_vol_ratio_20_60"] > 1.35)
        | (features["corr_20"] < 0.55)
        | (features["vix_z20"] > 1.25)
    ).astype(int)

    label = (prices["S"].shift(-config.horizon) - prices["S"]).rename("dS_H")
    dataset = features.join(label).dropna().copy()
    dataset["S_now"] = prices.loc[dataset.index, "S"]
    return dataset


def make_elastic_net(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000)),
        ]
    )


def make_xgboost(**kwargs: Any) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed. Run: pip install -e .[xgboost]") from exc
    return XGBRegressor(objective="reg:squarederror", random_state=7, **kwargs)


def model_factories(config: RunConfig) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "elastic_stable": lambda: make_elastic_net(alpha=0.08, l1_ratio=0.25),
        "elastic_balanced": lambda: make_elastic_net(alpha=0.03, l1_ratio=0.50),
        "elastic_sparse": lambda: make_elastic_net(alpha=0.015, l1_ratio=0.80),
    }
    if config.use_xgboost:
        factories.update(
            {
                "xgboost_shallow": lambda: make_xgboost(
                    n_estimators=250,
                    max_depth=2,
                    learning_rate=0.04,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=0.3,
                    reg_lambda=2.0,
                    min_child_weight=4,
                ),
                "xgboost_regularized": lambda: make_xgboost(
                    n_estimators=400,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=3.0,
                    min_child_weight=5,
                    gamma=0.1,
                ),
            }
        )
    return factories


def summarize_predictions(y_true: pd.Series, y_pred: pd.Series, context: pd.DataFrame, horizon: int) -> dict[str, float]:
    residual = y_true - y_pred
    residual_sigma = float(residual.std()) if len(residual) else 0.0
    edge_threshold = max(residual_sigma * 0.30, float(y_true.abs().median() * 0.15))

    size_multiplier = pd.Series(1.0, index=context.index)
    size_multiplier = size_multiplier.where(context["macro_event_window"] == 0, 0.75)
    size_multiplier = size_multiplier.where(context["break_flag"] == 0, size_multiplier * 0.50)

    direction = np.sign(y_pred)
    trade_mask = y_pred.abs() > edge_threshold
    weighted_signal = direction * size_multiplier * trade_mask.astype(float)
    realized_pnl = weighted_signal * y_true
    traded_pnl = realized_pnl[trade_mask]

    traded_hit_rate = float((np.sign(y_pred[trade_mask]) == np.sign(y_true[trade_mask])).mean()) if trade_mask.any() else 0.0
    pnl_std = float(traded_pnl.std()) if len(traded_pnl) > 1 else 0.0
    annualization = float(np.sqrt(252 / max(horizon, 1)))
    sharpe_proxy = float(traded_pnl.mean() / pnl_std * annualization) if pnl_std > 0 else 0.0
    selection_score = float(sharpe_proxy * np.sqrt(max(float(trade_mask.mean()), 1e-8)))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "hit_rate": float((np.sign(y_pred) == np.sign(y_true)).mean()),
        "traded_hit_rate": traded_hit_rate,
        "trade_rate": float(trade_mask.mean()),
        "avg_daily_pnl_proxy": float(realized_pnl.mean()),
        "avg_traded_pnl_proxy": float(traded_pnl.mean()) if len(traded_pnl) else 0.0,
        "cum_pnl_proxy": float(realized_pnl.sum()),
        "sharpe_proxy": sharpe_proxy,
        "selection_score": selection_score,
        "residual_sigma": residual_sigma,
        "edge_threshold": edge_threshold,
    }


def walk_forward_backtest(
    model_builder: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    context: pd.DataFrame,
    n_splits: int,
    horizon: int,
) -> tuple[pd.Series, dict[str, float]]:
    effective_splits = max(2, min(n_splits, len(X) - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    predictions = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in splitter.split(X):
        model = model_builder()
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        model.fit(X_train, y_train)
        predictions.iloc[test_idx] = model.predict(X_test)

    valid_index = predictions.dropna().index
    valid_pred = predictions.loc[valid_index]
    valid_y = y.loc[valid_index]
    valid_context = context.loc[valid_index]
    return predictions, summarize_predictions(valid_y, valid_pred, valid_context, horizon)


def summarize_feature_importance(model: Any, feature_names: list[str], top_n: int = 10) -> list[dict[str, Any]]:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float))
    elif hasattr(estimator, "feature_importances_"):
        values = np.abs(np.asarray(estimator.feature_importances_, dtype=float))
    else:
        return []

    importance = pd.Series(values, index=feature_names).sort_values(ascending=False).head(top_n)
    return [{"feature": str(name), "importance": float(value)} for name, value in importance.items()]


def build_quote_snapshot(
    config: RunConfig,
    latest_row: pd.Series,
    metrics: dict[str, float],
    final_prediction: float,
) -> dict[str, Any]:
    current_spread = float(latest_row["S_now"])
    fair_value = current_spread + final_prediction

    break_alarm = bool(latest_row["break_flag"])
    event_window = bool(latest_row["macro_event_window"])
    width_multiplier = 1.0
    size_multiplier = 1.0
    if break_alarm:
        width_multiplier *= 1.8
        size_multiplier *= 0.5
    if event_window:
        width_multiplier *= 1.4
        size_multiplier *= 0.75

    sigma = metrics["residual_sigma"]
    half_width = config.quote_width * sigma * width_multiplier
    signal_to_noise = abs(final_prediction) / sigma if sigma > 0 else 0.0

    return {
        "spread_now": current_spread,
        "predicted_delta_spread": float(final_prediction),
        "fair_value": float(fair_value),
        "bid": float(fair_value - half_width),
        "ask": float(fair_value + half_width),
        "residual_sigma": float(sigma),
        "edge_threshold": float(metrics["edge_threshold"]),
        "signal_to_noise": float(signal_to_noise),
        "macro_event_window": event_window,
        "break_alarm": break_alarm,
        "recommended_size_multiplier": float(size_multiplier),
    }


def evaluate_models(config: RunConfig, dataset: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    X = dataset.drop(columns=["dS_H", "S_now"])
    y = dataset["dS_H"]
    context = dataset[["macro_event_window", "break_flag", "S_now"]].copy()

    prediction_table = pd.DataFrame(index=dataset.index)
    prediction_table["actual_dS_H"] = y

    evaluated: dict[str, Any] = {}
    for model_name, builder in model_factories(config).items():
        predictions, metrics = walk_forward_backtest(builder, X, y, context, config.n_splits, config.horizon)
        fitted_model = builder()
        fitted_model.fit(X, y)
        final_prediction = float(fitted_model.predict(X.iloc[[-1]])[0])
        evaluated[model_name] = {
            "predictions": predictions,
            "metrics": metrics,
            "final_prediction": final_prediction,
            "feature_importance": summarize_feature_importance(fitted_model, list(X.columns)),
        }
        prediction_table[f"{model_name}_pred"] = predictions

    ranked_names = sorted(
        evaluated,
        key=lambda name: (
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
        reverse=True,
    )

    ensemble_members = ranked_names[: max(1, config.top_k_ensemble)]
    ensemble_prediction_frame = pd.concat(
        [evaluated[name]["predictions"].rename(name) for name in ensemble_members],
        axis=1,
    )
    ensemble_predictions = ensemble_prediction_frame.mean(axis=1)
    ensemble_valid_index = ensemble_predictions.dropna().index
    ensemble_metrics = summarize_predictions(
        y.loc[ensemble_valid_index],
        ensemble_predictions.loc[ensemble_valid_index],
        context.loc[ensemble_valid_index],
        config.horizon,
    )
    ensemble_final_prediction = float(np.mean([evaluated[name]["final_prediction"] for name in ensemble_members]))

    evaluated["ensemble_top"] = {
        "predictions": ensemble_predictions,
        "metrics": ensemble_metrics,
        "final_prediction": ensemble_final_prediction,
        "members": ensemble_members,
        "feature_importance": [],
    }
    prediction_table["ensemble_top_pred"] = ensemble_predictions

    champion_name = max(
        evaluated,
        key=lambda name: (
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
    )
    prediction_table["champion_pred"] = evaluated[champion_name]["predictions"]
    return {"evaluated": evaluated, "champion_name": champion_name}, prediction_table


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    dataset = build_feature_frame(config)
    evaluation, prediction_table = evaluate_models(config, dataset)
    latest_row = dataset.iloc[-1]

    model_results: dict[str, Any] = {}
    for model_name, payload in evaluation["evaluated"].items():
        model_results[model_name] = {
            "metrics": payload["metrics"],
            "live": build_quote_snapshot(config, latest_row, payload["metrics"], payload["final_prediction"]),
            "feature_importance": payload["feature_importance"],
        }
        if "members" in payload:
            model_results[model_name]["members"] = payload["members"]

    leaderboard = [
        {
            "model": model_name,
            "selection_score": float(payload["metrics"]["selection_score"]),
            "cum_pnl_proxy": float(payload["metrics"]["cum_pnl_proxy"]),
            "rmse": float(payload["metrics"]["rmse"]),
            "traded_hit_rate": float(payload["metrics"]["traded_hit_rate"]),
        }
        for model_name, payload in sorted(
            evaluation["evaluated"].items(),
            key=lambda item: (
                item[1]["metrics"]["selection_score"],
                item[1]["metrics"]["cum_pnl_proxy"],
                -item[1]["metrics"]["rmse"],
            ),
            reverse=True,
        )
    ]

    output = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "dataset": {
            "rows": int(len(dataset)),
            "start": str(dataset.index.min().date()),
            "end": str(dataset.index.max().date()),
        },
        "champion_model": evaluation["champion_name"],
        "leaderboard": leaderboard,
        "models": model_results,
    }
    write_outputs(config.output_dir, output, prediction_table)
    return output


def write_outputs(output_dir: Path, output: dict[str, Any], predictions: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_run.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "latest_predictions.csv", index_label="date")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run the spread model pipeline for SOXX/4 - IGV.")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in trading days.")
    parser.add_argument("--n-splits", type=int, default=6, help="Number of walk-forward splits.")
    parser.add_argument("--quote-width", type=float, default=0.35, help="Base quote-width factor applied to residual sigma.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for JSON and CSV outputs.")
    parser.add_argument("--use-xgboost", action="store_true", help="Evaluate the optional XGBoost models.")
    parser.add_argument("--top-k-ensemble", type=int, default=2, help="Number of top models to average in the ensemble.")
    args = parser.parse_args()
    return RunConfig(
        horizon=args.horizon,
        n_splits=args.n_splits,
        quote_width=args.quote_width,
        output_dir=args.output_dir,
        use_xgboost=args.use_xgboost,
        top_k_ensemble=args.top_k_ensemble,
    )


def main() -> None:
    config = parse_args()
    results = run_pipeline(config)
    print(f"Champion model: {results['champion_model']}")
    for entry in results["leaderboard"]:
        print(
            f"Leaderboard {entry['model']}:",
            f"Score={entry['selection_score']:.4f}",
            f"PnLProxy={entry['cum_pnl_proxy']:.4f}",
            f"RMSE={entry['rmse']:.4f}",
            f"TradedHitRate={entry['traded_hit_rate']:.3f}",
        )

    champion_payload = results["models"][results["champion_model"]]
    metrics = champion_payload["metrics"]
    live = champion_payload["live"]
    print(
        "Champion backtest:",
        f"MAE={metrics['mae']:.4f}",
        f"RMSE={metrics['rmse']:.4f}",
        f"HitRate={metrics['hit_rate']:.3f}",
        f"TradeRate={metrics['trade_rate']:.3f}",
        f"SharpeProxy={metrics['sharpe_proxy']:.3f}",
        f"PnLProxy={metrics['cum_pnl_proxy']:.4f}",
    )
    print(
        "Champion live:",
        f"S={live['spread_now']:.4f}",
        f"dS={live['predicted_delta_spread']:.4f}",
        f"Fair={live['fair_value']:.4f}",
        f"Bid={live['bid']:.4f}",
        f"Ask={live['ask']:.4f}",
        f"SNR={live['signal_to_noise']:.3f}",
        f"BreakAlarm={live['break_alarm']}",
        f"EventWindow={live['macro_event_window']}",
        f"Size={live['recommended_size_multiplier']:.2f}",
    )