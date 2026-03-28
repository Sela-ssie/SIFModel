from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
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
    use_lightgbm: bool = False
    top_k_ensemble: int = 3
    uncertainty_gate: float = 0.25
    break_threshold_mult: float = 0.25
    event_threshold_mult: float = 0.10
    stability_penalty_weight: float = 0.60
    normalize_label: bool = False
    sign_error_penalty: float = 2.5
    use_direction_filter: bool = False
    direction_gate_prob: float = 0.54


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
    for window in (1, 3, 5, 10, 20, 40, 60):
        features[f"S_diff_{window}"] = prices["S"].diff(window)
    for window in (10, 20, 60, 120):
        features[f"S_z{window}"] = safe_zscore(prices["S"], window)
        features[f"spread_vol_{window}"] = prices["S"].diff().rolling(window).std()

    features["spread_vol_ratio_20_60"] = features["spread_vol_20"] / features["spread_vol_60"].replace(0.0, np.nan)
    features["spread_vol_ratio_20_120"] = features["spread_vol_20"] / features["spread_vol_120"].replace(0.0, np.nan)
    features["spread_range_20"] = prices["S"].rolling(20).max() - prices["S"].rolling(20).min()
    features["spread_range_60"] = prices["S"].rolling(60).max() - prices["S"].rolling(60).min()
    features["spread_trend_20"] = prices["S"].diff(20)
    features["spread_trend_60"] = prices["S"].diff(60)

    for window in (1, 5, 10, 20, 40, 60):
        features[f"SOXX_ret_{window}"] = pct_change(prices["SOXX.US"], window)
        features[f"IGV_ret_{window}"] = pct_change(prices["IGV.US"], window)
        features[f"QQQ_ret_{window}"] = pct_change(prices["QQQ.US"], window)
        features[f"SPY_ret_{window}"] = pct_change(prices["SPY.US"], window)
        features[f"rel_ret_{window}"] = features[f"SOXX_ret_{window}"] - features[f"IGV_ret_{window}"]
        features[f"soxx_excess_qqq_{window}"] = features[f"SOXX_ret_{window}"] - features[f"QQQ_ret_{window}"]
        features[f"igv_excess_qqq_{window}"] = features[f"IGV_ret_{window}"] - features[f"QQQ_ret_{window}"]

    features["corr_20"] = returns["SOXX.US"].rolling(20).corr(returns["IGV.US"])
    features["corr_60"] = returns["SOXX.US"].rolling(60).corr(returns["IGV.US"])
    features["corr_120"] = returns["SOXX.US"].rolling(120).corr(returns["IGV.US"])
    features["corr_gap_20_60"] = features["corr_20"] - features["corr_60"]
    features["corr_gap_20_120"] = features["corr_20"] - features["corr_120"]
    features["soxx_beta_qqq_20"] = rolling_beta(returns["SOXX.US"], returns["QQQ.US"], 20)
    features["igv_beta_qqq_20"] = rolling_beta(returns["IGV.US"], returns["QQQ.US"], 20)
    features["beta_gap_qqq_20"] = features["soxx_beta_qqq_20"] - features["igv_beta_qqq_20"]
    features["soxx_beta_qqq_60"] = rolling_beta(returns["SOXX.US"], returns["QQQ.US"], 60)
    features["igv_beta_qqq_60"] = rolling_beta(returns["IGV.US"], returns["QQQ.US"], 60)
    features["beta_gap_qqq_60"] = features["soxx_beta_qqq_60"] - features["igv_beta_qqq_60"]

    features["DGS10"] = prices["DGS10"]
    features["DGS2"] = prices["DGS2"]
    features["curve_slope"] = prices["DGS10"] - prices["DGS2"]
    features["curve_slope_chg_5"] = features["curve_slope"].diff(5)
    features["curve_slope_chg_10"] = features["curve_slope"].diff(10)
    features["dgs10_chg_1"] = prices["DGS10"].diff(1)
    features["dgs10_chg_5"] = prices["DGS10"].diff(5)
    features["dgs10_chg_10"] = prices["DGS10"].diff(10)
    features["dgs2_chg_1"] = prices["DGS2"].diff(1)
    features["dgs2_chg_5"] = prices["DGS2"].diff(5)
    features["dgs2_chg_10"] = prices["DGS2"].diff(10)
    features["vix_level"] = prices["VIXCLS"]
    features["vix_chg_1"] = prices["VIXCLS"].diff(1)
    features["vix_chg_5"] = prices["VIXCLS"].diff(5)
    features["vix_z20"] = safe_zscore(prices["VIXCLS"], 20)
    features["vix_z60"] = safe_zscore(prices["VIXCLS"], 60)

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


def make_direction_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )


def make_xgboost(sign_error_penalty: float = 1.0, **kwargs: Any) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed. Run: pip install -e .[xgboost]") from exc
    if sign_error_penalty > 1.0:
        def _objective(labels: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            residual = preds - labels
            wrong = (np.sign(preds) != np.sign(labels)).astype(float)
            mult = 1.0 + (sign_error_penalty - 1.0) * wrong
            return residual * mult, mult
        objective: Any = _objective
    else:
        objective = "reg:squarederror"
    return XGBRegressor(objective=objective, random_state=7, **kwargs)


def make_lightgbm(**kwargs: Any) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("LightGBM is not installed. Run: pip install lightgbm") from exc
    return LGBMRegressor(objective="regression", random_state=7, verbose=-1, **kwargs)


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
                    sign_error_penalty=config.sign_error_penalty,
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
                    sign_error_penalty=config.sign_error_penalty,
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
    if config.use_lightgbm:
        factories.update(
            {
                "lightgbm_shallow": lambda: make_lightgbm(
                    n_estimators=300,
                    learning_rate=0.04,
                    num_leaves=15,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.2,
                    reg_lambda=2.5,
                    min_child_samples=30,
                ),
                "lightgbm_regularized": lambda: make_lightgbm(
                    n_estimators=450,
                    learning_rate=0.03,
                    num_leaves=23,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.4,
                    reg_lambda=3.5,
                    min_child_samples=40,
                ),
            }
        )
    return factories


def summarize_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    context: pd.DataFrame,
    horizon: int,
    uncertainty_gate: float,
    break_threshold_mult: float,
    event_threshold_mult: float,
    direction_prob_positive: pd.Series | None = None,
    direction_gate_prob: float = 0.50,
) -> dict[str, float]:
    residual = y_true - y_pred
    residual_sigma = float(residual.std()) if len(residual) else 0.0
    edge_threshold = max(residual_sigma * 0.30, float(y_true.abs().median() * 0.15))

    size_multiplier = pd.Series(1.0, index=context.index)
    size_multiplier = size_multiplier.where(context["macro_event_window"] == 0, 0.75)
    size_multiplier = size_multiplier.where(context["break_flag"] == 0, size_multiplier * 0.50)

    adaptive_edge = edge_threshold * (
        1.0
        + break_threshold_mult * context["break_flag"].astype(float)
        + event_threshold_mult * context["macro_event_window"].astype(float)
    )

    direction = np.sign(y_pred)
    normalized_confidence = y_pred.abs() / max(residual_sigma, 1e-8)
    trade_mask = (y_pred.abs() > adaptive_edge) & (normalized_confidence > uncertainty_gate)
    if direction_prob_positive is not None:
        direction_prob_positive = direction_prob_positive.reindex(y_pred.index)
        classifier_direction = np.where(direction_prob_positive >= 0.5, 1.0, -1.0)
        classifier_confidence = np.maximum(direction_prob_positive, 1.0 - direction_prob_positive)
        agreement = classifier_direction == np.where(direction >= 0.0, 1.0, -1.0)
        trade_mask = trade_mask & agreement & (classifier_confidence >= direction_gate_prob)
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
        "raw_selection_score": selection_score,
        "residual_sigma": residual_sigma,
        "edge_threshold": edge_threshold,
    }


def walk_forward_direction_filter(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> tuple[pd.Series, Pipeline]:
    effective_splits = max(2, min(n_splits, len(X) - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    direction_prob = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in splitter.split(X):
        classifier = make_direction_classifier()
        y_train_direction = (y.iloc[train_idx] > 0.0).astype(int)
        classifier.fit(X.iloc[train_idx], y_train_direction)
        direction_prob.iloc[test_idx] = classifier.predict_proba(X.iloc[test_idx])[:, 1]

    full_classifier = make_direction_classifier()
    full_classifier.fit(X, (y > 0.0).astype(int))
    return direction_prob, full_classifier


def walk_forward_backtest(
    model_builder: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    context: pd.DataFrame,
    n_splits: int,
    horizon: int,
    uncertainty_gate: float,
    break_threshold_mult: float,
    event_threshold_mult: float,
    stability_penalty_weight: float,
    vol_scale: pd.Series | None = None,
    direction_prob_positive: pd.Series | None = None,
    direction_gate_prob: float = 0.50,
) -> tuple[pd.Series, dict[str, float]]:
    effective_splits = max(2, min(n_splits, len(X) - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    predictions = pd.Series(index=X.index, dtype=float)
    fold_pnls: list[float] = []

    for train_idx, test_idx in splitter.split(X):
        model = model_builder()
        X_train = X.iloc[train_idx]
        y_train_raw = y.iloc[train_idx]
        y_train_fit = y_train_raw / vol_scale.iloc[train_idx] if vol_scale is not None else y_train_raw
        X_test = X.iloc[test_idx]
        model.fit(X_train, y_train_fit)
        fold_pred_norm = pd.Series(model.predict(X_test), index=X_test.index, dtype=float)
        fold_pred = fold_pred_norm * vol_scale.iloc[test_idx] if vol_scale is not None else fold_pred_norm
        predictions.iloc[test_idx] = fold_pred
        fold_metrics = summarize_predictions(
            y.iloc[test_idx],
            fold_pred,
            context.iloc[test_idx],
            horizon,
            uncertainty_gate,
            break_threshold_mult,
            event_threshold_mult,
            direction_prob_positive.iloc[test_idx] if direction_prob_positive is not None else None,
            direction_gate_prob,
        )
        fold_pnls.append(float(fold_metrics["cum_pnl_proxy"]))

    valid_index = predictions.dropna().index
    valid_pred = predictions.loc[valid_index]
    valid_y = y.loc[valid_index]
    valid_context = context.loc[valid_index]
    metrics = summarize_predictions(
        valid_y,
        valid_pred,
        valid_context,
        horizon,
        uncertainty_gate,
        break_threshold_mult,
        event_threshold_mult,
        direction_prob_positive.loc[valid_index] if direction_prob_positive is not None else None,
        direction_gate_prob,
    )

    mean_fold_pnl = float(np.mean(fold_pnls)) if fold_pnls else 0.0
    worst_fold_pnl = float(np.min(fold_pnls)) if fold_pnls else 0.0
    downside = abs(min(worst_fold_pnl, 0.0))
    baseline = abs(mean_fold_pnl) + 1e-8
    fold_drawdown_ratio = downside / baseline
    stability_penalty = float(1.0 / (1.0 + stability_penalty_weight * fold_drawdown_ratio))

    metrics["raw_selection_score"] = float(metrics["selection_score"])
    metrics["selection_score"] = float(metrics["selection_score"] * stability_penalty)
    metrics["stability_penalty"] = stability_penalty
    metrics["worst_fold_pnl"] = worst_fold_pnl
    metrics["mean_fold_pnl"] = mean_fold_pnl
    metrics["fold_pnl_std"] = float(np.std(fold_pnls)) if fold_pnls else 0.0
    metrics["positive_fold_rate"] = float(np.mean(np.asarray(fold_pnls) > 0.0)) if fold_pnls else 0.0

    return predictions, metrics


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
    vol_scale: pd.Series | None = dataset["spread_vol_20"].clip(lower=1e-8) if config.normalize_label else None
    context = dataset[["macro_event_window", "break_flag", "S_now"]].copy()
    direction_prob_positive: pd.Series | None = None
    if config.use_direction_filter:
        direction_prob_positive, _ = walk_forward_direction_filter(X, y, config.n_splits)

    prediction_table = pd.DataFrame(index=dataset.index)
    prediction_table["actual_dS_H"] = y
    if direction_prob_positive is not None:
        prediction_table["direction_prob_positive"] = direction_prob_positive

    evaluated: dict[str, Any] = {}
    for model_name, builder in model_factories(config).items():
        predictions, metrics = walk_forward_backtest(
            builder,
            X,
            y,
            context,
            config.n_splits,
            config.horizon,
            config.uncertainty_gate,
            config.break_threshold_mult,
            config.event_threshold_mult,
            config.stability_penalty_weight,
            vol_scale=vol_scale,
            direction_prob_positive=direction_prob_positive,
            direction_gate_prob=config.direction_gate_prob,
        )
        fitted_model = builder()
        if vol_scale is not None:
            fitted_model.fit(X, y / vol_scale)
            final_prediction = float(fitted_model.predict(X.iloc[[-1]])[0]) * float(vol_scale.iloc[-1])
        else:
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
    member_scores = np.asarray(
        [max(0.0, evaluated[name]["metrics"]["selection_score"]) for name in ensemble_members],
        dtype=float,
    )
    if float(member_scores.sum()) <= 0.0:
        member_weights = np.full(shape=len(ensemble_members), fill_value=1.0 / len(ensemble_members), dtype=float)
    else:
        member_weights = member_scores / member_scores.sum()
    weighted_frame = ensemble_prediction_frame.mul(member_weights, axis=1)
    ensemble_predictions = weighted_frame.sum(axis=1)
    ensemble_valid_index = ensemble_predictions.dropna().index
    ensemble_metrics = summarize_predictions(
        y.loc[ensemble_valid_index],
        ensemble_predictions.loc[ensemble_valid_index],
        context.loc[ensemble_valid_index],
        config.horizon,
        config.uncertainty_gate,
        config.break_threshold_mult,
        config.event_threshold_mult,
        direction_prob_positive.loc[ensemble_valid_index] if direction_prob_positive is not None else None,
        config.direction_gate_prob,
    )
    ensemble_final_prediction = float(
        np.dot(member_weights, np.asarray([evaluated[name]["final_prediction"] for name in ensemble_members], dtype=float))
    )

    evaluated["ensemble_top"] = {
        "predictions": ensemble_predictions,
        "metrics": ensemble_metrics,
        "final_prediction": ensemble_final_prediction,
        "members": ensemble_members,
        "member_weights": {
            name: float(weight) for name, weight in zip(ensemble_members, member_weights, strict=False)
        },
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
        if "member_weights" in payload:
            model_results[model_name]["member_weights"] = payload["member_weights"]

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
    parser.add_argument("--use-lightgbm", action="store_true", help="Evaluate optional LightGBM challenger models.")
    parser.add_argument("--top-k-ensemble", type=int, default=3, help="Number of top models to average in the ensemble.")
    parser.add_argument(
        "--uncertainty-gate",
        type=float,
        default=0.25,
        help="Minimum normalized confidence (|pred|/sigma) required to trade.",
    )
    parser.add_argument(
        "--break-threshold-mult",
        type=float,
        default=0.25,
        help="Additional threshold multiplier applied during break-like regimes.",
    )
    parser.add_argument(
        "--event-threshold-mult",
        type=float,
        default=0.10,
        help="Additional threshold multiplier applied during macro event windows.",
    )
    parser.add_argument(
        "--stability-penalty-weight",
        type=float,
        default=0.60,
        help="Penalty weight for unstable fold PnL when ranking models.",
    )
    parser.add_argument(
        "--direction-gate-prob",
        type=float,
        default=0.54,
        help="Minimum classifier confidence required when confirming signal direction.",
    )
    parser.add_argument(
        "--use-direction-filter",
        action="store_true",
        help="Enable experimental classifier-based direction confirmation before trading.",
    )
    parser.add_argument(
        "--normalize-label",
        action="store_true",
        help="Enable volatility-normalized label training (off by default; harms linear models).",
    )
    parser.add_argument(
        "--sign-error-penalty",
        type=float,
        default=2.5,
        help="XGBoost sign-error loss multiplier (>1 penalises wrong-direction predictions).",
    )
    args = parser.parse_args()
    return RunConfig(
        horizon=args.horizon,
        n_splits=args.n_splits,
        quote_width=args.quote_width,
        output_dir=args.output_dir,
        use_xgboost=args.use_xgboost,
        use_lightgbm=args.use_lightgbm,
        top_k_ensemble=args.top_k_ensemble,
        uncertainty_gate=args.uncertainty_gate,
        break_threshold_mult=args.break_threshold_mult,
        event_threshold_mult=args.event_threshold_mult,
        stability_penalty_weight=args.stability_penalty_weight,
        normalize_label=args.normalize_label,
        sign_error_penalty=args.sign_error_penalty,
        use_direction_filter=args.use_direction_filter,
        direction_gate_prob=args.direction_gate_prob,
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