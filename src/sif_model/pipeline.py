from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.request
import urllib.parse
from dataclasses import asdict, dataclass
from itertools import product
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
YAHOO_SYMBOL_MAP = {
    "soxx.us": "SOXX",
    "igv.us": "IGV",
    "qqq.us": "QQQ",
    "spy.us": "SPY",
}
FRED_SERIES = ("DGS10", "DGS2", "VIXCLS")


@dataclass(slots=True)
class RunConfig:
    horizon: int = 15
    n_splits: int = 6
    quote_width: float = 0.35
    output_dir: Path = Path("outputs")
    use_xgboost: bool = False
    use_lightgbm: bool = False
    top_k_ensemble: int = 5
    uncertainty_gate: float = 0.25
    break_threshold_mult: float = 0.35
    event_threshold_mult: float = 0.10
    stability_penalty_weight: float = 0.40
    normalize_label: bool = False
    sign_error_penalty: float = 3.5
    use_direction_filter: bool = False
    direction_gate_prob: float = 0.54
    use_regime_router: bool = True
    min_size_multiplier: float = 1.0
    max_size_multiplier: float = 1.0
    size_sigmoid_k: float = 3.0
    size_conf_mid: float = 0.60
    use_purged_cv: bool = True
    holdout_fraction: float = 0.15
    use_meta_label_filter: bool = False
    meta_label_gate_prob: float = 0.54
    run_objective_tuning: bool = False
    tuning_trials: int = 24


def clone_config(config: RunConfig, **updates: Any) -> RunConfig:
    cfg = asdict(config)
    cfg.update(updates)
    output_dir_value = cfg.get("output_dir", Path("outputs"))
    cfg["output_dir"] = output_dir_value if isinstance(output_dir_value, Path) else Path(str(output_dir_value))
    return RunConfig(**cfg)


def load_stooq_close(symbol: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    last_error: Exception | None = None
    frame: pd.DataFrame | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read().decode("utf-8", errors="ignore")
            if not payload.strip():
                raise ValueError(f"Empty response for {symbol}")
            frame = pd.read_csv(io.StringIO(payload))
            if len(frame.columns) == 0:
                raise ValueError(f"No columns returned for {symbol}")
            break
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.0 + attempt)
    if frame is None:
        yahoo_symbol = YAHOO_SYMBOL_MAP.get(symbol.lower())
        if yahoo_symbol is not None:
            return load_yahoo_close(yahoo_symbol).rename(symbol.upper())
        raise ValueError(f"Failed to fetch Stooq data for {symbol}: {last_error}")
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"Unexpected Stooq schema for {symbol}: {frame.columns.tolist()}")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    return frame["Close"].astype(float).rename(symbol.upper())


def load_yahoo_close(symbol: str) -> pd.Series:
    period1 = int(pd.Timestamp("2010-01-01").timestamp())
    period2 = int(pd.Timestamp.utcnow().timestamp())
    params = urllib.parse.urlencode({"period1": period1, "period2": period2, "interval": "1d"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    body = json.loads(payload)
    result = body.get("chart", {}).get("result", [])
    if not result:
        raise ValueError(f"Unexpected Yahoo response for {symbol}: {body}")
    timestamps = result[0].get("timestamp", [])
    closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
    if not timestamps or not closes:
        raise ValueError(f"Yahoo chart response missing data for {symbol}")
    index = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    close_series = pd.to_numeric(pd.Series(closes), errors="coerce")
    frame = pd.DataFrame({"close": close_series.values}, index=index)
    return frame["close"].dropna().astype(float).rename(symbol)


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
                "lightgbm_trend": lambda: make_lightgbm(
                    n_estimators=650,
                    learning_rate=0.02,
                    num_leaves=19,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    reg_alpha=0.15,
                    reg_lambda=2.0,
                    min_child_samples=20,
                ),
                "lightgbm_highcap": lambda: make_lightgbm(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=2.5,
                    min_child_samples=16,
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
    meta_label_prob: pd.Series | None = None,
    meta_label_gate_prob: float = 0.56,
    min_size_multiplier: float = 0.35,
    max_size_multiplier: float = 1.25,
    size_sigmoid_k: float = 3.0,
    size_conf_mid: float = 0.60,
    trade_rate_cap: float | None = 0.25,
    reference_sigma: float | None = None,
) -> dict[str, float]:
    residual = y_true - y_pred
    residual_sigma = float(residual.std()) if len(residual) else 0.0
    min_edge = float(y_true.abs().median() * 0.15)

    # Use reference_sigma (from CV calibration) when provided so holdout
    # evaluation uses the same threshold scale as CV — avoids look-ahead bias
    # where a shorter/quieter holdout window would produce a smaller sigma and
    # therefore a lower filter bar, artificially inflating the holdout trade rate.
    effective_sigma = reference_sigma if (reference_sigma is not None and reference_sigma > 0) else residual_sigma
    _base_edge = max(effective_sigma * 0.30, min_edge)
    local_sigma = pd.Series(max(effective_sigma, 1e-8), index=y_pred.index)
    edge_threshold_series = pd.Series(_base_edge, index=y_pred.index)

    size_multiplier = pd.Series(1.0, index=context.index)
    size_multiplier = size_multiplier.where(context["macro_event_window"] == 0, 0.75)
    size_multiplier = size_multiplier.where(context["break_flag"] == 0, size_multiplier * 0.50)

    adaptive_edge = edge_threshold_series * (
        1.0
        + break_threshold_mult * context["break_flag"].astype(float)
        + event_threshold_mult * context["macro_event_window"].astype(float)
    )

    direction = np.sign(y_pred)
    normalized_confidence = y_pred.abs() / local_sigma
    trade_mask = (y_pred.abs() > adaptive_edge) & (normalized_confidence > uncertainty_gate)
    if direction_prob_positive is not None:
        direction_prob_positive = direction_prob_positive.reindex(y_pred.index)
        classifier_direction = np.where(direction_prob_positive >= 0.5, 1.0, -1.0)
        classifier_confidence = np.maximum(direction_prob_positive, 1.0 - direction_prob_positive)
        agreement = classifier_direction == np.where(direction >= 0.0, 1.0, -1.0)
        trade_mask = trade_mask & agreement & (classifier_confidence >= direction_gate_prob)
    if meta_label_prob is not None:
        trade_mask = trade_mask & (meta_label_prob.reindex(y_pred.index).fillna(0.0) >= meta_label_gate_prob)

    edge_excess = (y_pred.abs() - adaptive_edge) / local_sigma

    if trade_rate_cap is not None and 0.0 < trade_rate_cap < float(trade_mask.mean()):
        eligible_index = edge_excess[trade_mask].sort_values(ascending=False).index
        max_trades = max(1, int(np.floor(trade_rate_cap * len(trade_mask))))
        selected_index = eligible_index[:max_trades]
        capped_trade_mask = pd.Series(False, index=trade_mask.index)
        capped_trade_mask.loc[selected_index] = True
        trade_mask = capped_trade_mask

    confidence_curve = 1.0 / (1.0 + np.exp(-size_sigmoid_k * (edge_excess - size_conf_mid)))
    dynamic_size = min_size_multiplier + (max_size_multiplier - min_size_multiplier) * confidence_curve
    effective_size_multiplier = size_multiplier * dynamic_size

    weighted_signal = direction * effective_size_multiplier * trade_mask.astype(float)
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
        "avg_size_multiplier": float(effective_size_multiplier.where(trade_mask, 0.0).mean()),
        "avg_traded_size_multiplier": float(effective_size_multiplier[trade_mask].mean()) if trade_mask.any() else 0.0,
        "sharpe_proxy": sharpe_proxy,
        "selection_score": selection_score,
        "raw_selection_score": selection_score,
        "residual_sigma": residual_sigma,
        "edge_threshold": float(edge_threshold_series.iloc[-1]) if len(edge_threshold_series) else min_edge,
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


def fold_meta_label_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_model: Any,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Estimate fold-local trade-quality probability for a base model signal."""
    train_pred = pd.Series(base_model.predict(X_train), index=X_train.index, dtype=float)
    train_edge = float(max(1e-8, y_train.abs().median() * 0.10))
    quality_label = ((np.sign(train_pred) == np.sign(y_train)) & (train_pred.abs() >= train_edge)).astype(int)

    if quality_label.nunique() < 2:
        default_prob = float(quality_label.iloc[-1]) if len(quality_label) else 0.5
        return pd.Series(default_prob, index=X_test.index, dtype=float)

    quality_clf = make_direction_classifier()
    quality_clf.fit(X_train, quality_label)
    return pd.Series(quality_clf.predict_proba(X_test)[:, 1], index=X_test.index, dtype=float)


def purged_walk_forward_split(
    X: pd.DataFrame,
    n_splits: int,
    horizon: int,
    holdout_fraction: float = 0.15,
    holdout_size: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Create purged walk-forward splits with lookahead bias removal and a locked holdout block.
    
    For each fold, remove training data within `horizon` periods before the test set starts
    to prevent lookahead bias.
    
    Returns:
        - list of (train_idx, test_idx) tuples for CV folds
        - holdout_idx for final locked holdout evaluation
    """
    n_total = len(X)
    if holdout_size is None:
        holdout_size = max(1, int(n_total * holdout_fraction))
    else:
        holdout_size = max(1, int(holdout_size))
    holdout_size = min(holdout_size, max(1, n_total - 3))
    holdout_start = n_total - holdout_size
    holdout_idx = np.arange(holdout_start, n_total)
    
    # Create time series splits on the non-holdout portion
    cv_X = X.iloc[:holdout_start]
    n_cv = len(cv_X)
    effective_splits = max(2, min(n_splits, n_cv - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    
    purged_splits = []
    for train_idx_relative, test_idx_relative in splitter.split(cv_X):
        # Purge by sample index gap (trading periods), not calendar days.
        test_start_pos = int(test_idx_relative[0])
        purge_until = max(0, test_start_pos - max(1, horizon))
        purged_train_idx = train_idx_relative[train_idx_relative < purge_until]
        if len(purged_train_idx) > 10:
            purged_splits.append((purged_train_idx, test_idx_relative))

    if not purged_splits:
        fallback_splitter = TimeSeriesSplit(n_splits=max(2, min(n_splits, max(3, n_cv - 1))))
        purged_splits = list(fallback_splitter.split(cv_X))
    
    return purged_splits, holdout_idx


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
    use_meta_label_filter: bool = False,
    meta_label_gate_prob: float = 0.56,
    min_size_multiplier: float = 0.35,
    max_size_multiplier: float = 1.25,
    size_sigmoid_k: float = 3.0,
    size_conf_mid: float = 0.60,
    use_purged_cv: bool = True,
    holdout_fraction: float = 0.15,
    holdout_size: int | None = None,
) -> tuple[pd.Series, dict[str, float], dict[str, Any]]:
    """
    Walk-forward backtest with optional purged CV and locked holdout evaluation.

    Returns:
        predictions: Series of predictions on non-holdout data
        metrics: Dictionary of fold-aggregated metrics
        holdout_results: Dictionary containing holdout test results
    """
    predictions = pd.Series(index=X.index, dtype=float)
    meta_label_prob = pd.Series(index=X.index, dtype=float)
    fold_pnls: list[float] = []
    
    if use_purged_cv:
        splits, holdout_idx = purged_walk_forward_split(X, n_splits, horizon, holdout_fraction, holdout_size)
    else:
        effective_splits = max(2, min(n_splits, len(X) - 1))
        splitter = TimeSeriesSplit(n_splits=effective_splits)
        splits = list(splitter.split(X))
        resolved_holdout_size = holdout_size if holdout_size is not None else max(1, int(len(X) * holdout_fraction))
        resolved_holdout_size = min(max(1, int(resolved_holdout_size)), max(1, len(X) - 3))
        holdout_idx = np.arange(len(X) - resolved_holdout_size, len(X))

    for train_idx, test_idx in splits:
        model = model_builder()
        X_train = X.iloc[train_idx]
        y_train_raw = y.iloc[train_idx]
        y_train_fit = y_train_raw / vol_scale.iloc[train_idx] if vol_scale is not None else y_train_raw
        X_test = X.iloc[test_idx]
        model.fit(X_train, y_train_fit)
        fold_pred_norm = pd.Series(model.predict(X_test), index=X_test.index, dtype=float)
        fold_pred = fold_pred_norm * vol_scale.iloc[test_idx] if vol_scale is not None else fold_pred_norm
        predictions.iloc[test_idx] = fold_pred
        if use_meta_label_filter:
            meta_label_prob.iloc[test_idx] = fold_meta_label_probability(X_train, y_train_raw, model, X_test)
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
            meta_label_prob.iloc[test_idx] if use_meta_label_filter else None,
            meta_label_gate_prob,
            min_size_multiplier,
            max_size_multiplier,
            size_sigmoid_k,
            size_conf_mid,
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
        meta_label_prob.loc[valid_index] if use_meta_label_filter else None,
        meta_label_gate_prob,
        min_size_multiplier,
        max_size_multiplier,
        size_sigmoid_k,
        size_conf_mid,
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

    # Evaluate on locked holdout
    X_holdout = X.iloc[holdout_idx]
    y_holdout = y.iloc[holdout_idx]
    context_holdout = context.iloc[holdout_idx]
    direction_holdout = None
    if direction_prob_positive is not None:
        direction_holdout = direction_prob_positive.iloc[holdout_idx]
    
    # Train on all non-holdout data for holdout evaluation
    train_end = len(X) - len(holdout_idx)
    y_train_raw = y.iloc[:train_end] if len(holdout_idx) > 0 else y
    vol_holdout = vol_scale.iloc[holdout_idx] if vol_scale is not None else None
    y_train_fit = y_train_raw / vol_scale.iloc[:train_end] if vol_scale is not None else y_train_raw
    
    model_for_holdout = model_builder()
    model_for_holdout.fit(X.iloc[:train_end], y_train_fit)
    holdout_pred_norm = model_for_holdout.predict(X_holdout)
    holdout_pred = holdout_pred_norm * vol_holdout if vol_scale is not None else holdout_pred_norm
    holdout_pred = pd.Series(holdout_pred, index=X_holdout.index, dtype=float)
    holdout_meta_prob = None
    if use_meta_label_filter:
        holdout_meta_prob = fold_meta_label_probability(
            X.iloc[:train_end],
            y.iloc[:train_end],
            model_for_holdout,
            X_holdout,
        )
    
    holdout_metrics = summarize_predictions(
        y_holdout,
        holdout_pred,
        context_holdout,
        horizon,
        uncertainty_gate,
        break_threshold_mult,
        event_threshold_mult,
        direction_holdout,
        direction_gate_prob,
        holdout_meta_prob,
        meta_label_gate_prob,
        min_size_multiplier,
        max_size_multiplier,
        size_sigmoid_k,
        size_conf_mid,
        reference_sigma=metrics["residual_sigma"],
    )
    
    holdout_results = {
        "holdout_predictions": holdout_pred,
        "holdout_metrics": holdout_metrics,
        "holdout_size": len(holdout_idx),
    }

    return predictions, metrics, holdout_results


def apply_prediction_stability_penalty(
    predictions: pd.Series,
    metrics: dict[str, float],
    y: pd.Series,
    context: pd.DataFrame,
    config: RunConfig,
    direction_prob_positive: pd.Series | None = None,
) -> dict[str, float]:
    valid_predictions = predictions.dropna()
    if len(valid_predictions) < max(50, config.n_splits * 10):
        return metrics

    dummy_X = pd.DataFrame(index=valid_predictions.index)
    effective_splits = max(2, min(config.n_splits, len(dummy_X) - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    fold_pnls: list[float] = []

    for _, test_idx in splitter.split(dummy_X):
        fold_index = valid_predictions.index[test_idx]
        fold_metrics = summarize_predictions(
            y.loc[fold_index],
            valid_predictions.loc[fold_index],
            context.loc[fold_index],
            horizon=config.horizon,
            uncertainty_gate=config.uncertainty_gate,
            break_threshold_mult=config.break_threshold_mult,
            event_threshold_mult=config.event_threshold_mult,
            direction_prob_positive=direction_prob_positive.loc[fold_index] if direction_prob_positive is not None else None,
            direction_gate_prob=config.direction_gate_prob,
            meta_label_prob=None,
            meta_label_gate_prob=config.meta_label_gate_prob,
            min_size_multiplier=config.min_size_multiplier,
            max_size_multiplier=config.max_size_multiplier,
            size_sigmoid_k=config.size_sigmoid_k,
            size_conf_mid=config.size_conf_mid,
        )
        fold_pnls.append(float(fold_metrics["cum_pnl_proxy"]))

    if not fold_pnls:
        return metrics

    mean_fold_pnl = float(np.mean(fold_pnls))
    worst_fold_pnl = float(np.min(fold_pnls))
    downside = abs(min(worst_fold_pnl, 0.0))
    baseline = abs(mean_fold_pnl) + 1e-8
    fold_drawdown_ratio = downside / baseline
    stability_penalty = float(1.0 / (1.0 + config.stability_penalty_weight * fold_drawdown_ratio))

    adjusted_metrics = dict(metrics)
    adjusted_metrics["raw_selection_score"] = float(metrics.get("raw_selection_score", metrics["selection_score"]))
    adjusted_metrics["selection_score"] = float(adjusted_metrics["raw_selection_score"] * stability_penalty)
    adjusted_metrics["stability_penalty"] = stability_penalty
    adjusted_metrics["worst_fold_pnl"] = worst_fold_pnl
    adjusted_metrics["mean_fold_pnl"] = mean_fold_pnl
    adjusted_metrics["fold_pnl_std"] = float(np.std(fold_pnls))
    adjusted_metrics["positive_fold_rate"] = float(np.mean(np.asarray(fold_pnls) > 0.0))
    return adjusted_metrics


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


def build_regime_router_candidate(
    evaluated: dict[str, Any],
    y: pd.Series,
    context: pd.DataFrame,
    config: RunConfig,
    direction_prob_positive: pd.Series | None,
) -> dict[str, Any] | None:
    base_names = list(evaluated.keys())
    if not base_names:
        return None

    ranked_base_names = sorted(
        base_names,
        key=lambda name: (
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
        reverse=True,
    )
    candidate_pool_size = max(3, min(len(ranked_base_names), config.top_k_ensemble + 2))
    base_names = ranked_base_names[:candidate_pool_size]

    calm_mask = (context["break_flag"] == 0) & (context["macro_event_window"] == 0)
    stress_mask = ~calm_mask

    def best_for_mask(mask: pd.Series) -> tuple[str, dict[str, float]] | None:
        best_name: str | None = None
        best_metrics: dict[str, float] | None = None
        for name in base_names:
            pred = evaluated[name]["predictions"].dropna()
            if pred.empty:
                continue
            idx = pred.index[mask.reindex(pred.index).fillna(False)]
            if len(idx) < 200:
                continue
            metrics = summarize_predictions(
                y.loc[idx],
                pred.loc[idx],
                context.loc[idx],
                horizon=config.horizon,
                uncertainty_gate=config.uncertainty_gate,
                break_threshold_mult=config.break_threshold_mult,
                event_threshold_mult=config.event_threshold_mult,
                direction_prob_positive=direction_prob_positive.loc[idx] if direction_prob_positive is not None else None,
                direction_gate_prob=config.direction_gate_prob,
                meta_label_prob=None,
                meta_label_gate_prob=config.meta_label_gate_prob,
                min_size_multiplier=config.min_size_multiplier,
                max_size_multiplier=config.max_size_multiplier,
                size_sigmoid_k=config.size_sigmoid_k,
                size_conf_mid=config.size_conf_mid,
            )
            if best_metrics is None or (
                metrics["selection_score"],
                metrics["cum_pnl_proxy"],
                -metrics["rmse"],
            ) > (
                best_metrics["selection_score"],
                best_metrics["cum_pnl_proxy"],
                -best_metrics["rmse"],
            ):
                best_name = name
                best_metrics = metrics

        if best_name is None or best_metrics is None:
            return None
        return best_name, best_metrics

    calm_choice = best_for_mask(calm_mask)
    stress_choice = best_for_mask(stress_mask)
    if calm_choice is None or stress_choice is None:
        return None

    calm_name, _ = calm_choice
    stress_name, _ = stress_choice
    calm_pred = evaluated[calm_name]["predictions"]
    stress_pred = evaluated[stress_name]["predictions"]

    router_pred = pd.Series(index=y.index, dtype=float)
    calm_idx = router_pred.index[calm_mask.reindex(router_pred.index).fillna(False)]
    stress_idx = router_pred.index[stress_mask.reindex(router_pred.index).fillna(False)]
    router_pred.loc[calm_idx] = calm_pred.reindex(calm_idx)
    router_pred.loc[stress_idx] = stress_pred.reindex(stress_idx)

    valid_idx = router_pred.dropna().index
    if len(valid_idx) < 400:
        return None

    metrics = summarize_predictions(
        y.loc[valid_idx],
        router_pred.loc[valid_idx],
        context.loc[valid_idx],
        horizon=config.horizon,
        uncertainty_gate=config.uncertainty_gate,
        break_threshold_mult=config.break_threshold_mult,
        event_threshold_mult=config.event_threshold_mult,
        direction_prob_positive=direction_prob_positive.loc[valid_idx] if direction_prob_positive is not None else None,
        direction_gate_prob=config.direction_gate_prob,
        meta_label_prob=None,
        meta_label_gate_prob=config.meta_label_gate_prob,
        min_size_multiplier=config.min_size_multiplier,
        max_size_multiplier=config.max_size_multiplier,
        size_sigmoid_k=config.size_sigmoid_k,
        size_conf_mid=config.size_conf_mid,
    )
    metrics = apply_prediction_stability_penalty(
        router_pred.loc[valid_idx],
        metrics,
        y,
        context,
        config,
        direction_prob_positive,
    )

    router_holdout_results: dict[str, Any] | None = None
    calm_holdout = evaluated[calm_name].get("holdout_results", {}).get("holdout_predictions")
    stress_holdout = evaluated[stress_name].get("holdout_results", {}).get("holdout_predictions")
    if calm_holdout is not None and stress_holdout is not None:
        holdout_index = calm_holdout.index.union(stress_holdout.index)
        router_holdout_pred = pd.Series(index=holdout_index, dtype=float)
        calm_holdout_idx = holdout_index[calm_mask.reindex(holdout_index).fillna(False)]
        stress_holdout_idx = holdout_index[stress_mask.reindex(holdout_index).fillna(False)]
        router_holdout_pred.loc[calm_holdout_idx] = calm_holdout.reindex(calm_holdout_idx)
        router_holdout_pred.loc[stress_holdout_idx] = stress_holdout.reindex(stress_holdout_idx)

        valid_holdout_idx = router_holdout_pred.dropna().index
        if len(valid_holdout_idx) > 0:
            router_holdout_metrics = summarize_predictions(
                y.loc[valid_holdout_idx],
                router_holdout_pred.loc[valid_holdout_idx],
                context.loc[valid_holdout_idx],
                horizon=config.horizon,
                uncertainty_gate=config.uncertainty_gate,
                break_threshold_mult=config.break_threshold_mult,
                event_threshold_mult=config.event_threshold_mult,
                direction_prob_positive=direction_prob_positive.loc[valid_holdout_idx]
                if direction_prob_positive is not None
                else None,
                direction_gate_prob=config.direction_gate_prob,
                meta_label_prob=None,
                meta_label_gate_prob=config.meta_label_gate_prob,
                min_size_multiplier=config.min_size_multiplier,
                max_size_multiplier=config.max_size_multiplier,
                size_sigmoid_k=config.size_sigmoid_k,
                size_conf_mid=config.size_conf_mid,
                reference_sigma=metrics["residual_sigma"],
            )
            router_holdout_results = {
                "holdout_predictions": router_holdout_pred,
                "holdout_metrics": router_holdout_metrics,
                "holdout_size": len(valid_holdout_idx),
            }

    latest_is_stress = bool(stress_mask.iloc[-1])
    final_prediction = float(
        evaluated[stress_name]["final_prediction"] if latest_is_stress else evaluated[calm_name]["final_prediction"]
    )

    return {
        "predictions": router_pred,
        "metrics": metrics,
        "final_prediction": final_prediction,
        "feature_importance": [],
        "members": [calm_name, stress_name],
        "router_rules": {
            "calm_model": calm_name,
            "stress_model": stress_name,
            "calm_condition": "break_flag==0 and macro_event_window==0",
        },
        "holdout_results": router_holdout_results,
    }


def build_stacked_blend_candidate(
    evaluated: dict[str, Any],
    y: pd.Series,
    context: pd.DataFrame,
    config: RunConfig,
    direction_prob_positive: pd.Series | None,
) -> dict[str, Any] | None:
    base_names = [name for name, payload in evaluated.items() if "members" not in payload]
    if len(base_names) < 2:
        return None

    ranked_base_names = sorted(
        base_names,
        key=lambda name: (
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
        reverse=True,
    )
    selected_names = ranked_base_names[: max(2, min(len(ranked_base_names), config.top_k_ensemble + 1))]

    meta_columns: dict[str, pd.Series] = {}
    holdout_index: pd.Index | None = None
    prediction_column_names: list[str] = []
    for name in selected_names:
        combined_pred = evaluated[name]["predictions"].copy()
        holdout_pred = evaluated[name].get("holdout_results", {}).get("holdout_predictions")
        if holdout_pred is None:
            return None
        combined_pred.loc[holdout_pred.index] = holdout_pred
        column_name = f"{name}_pred"
        prediction_column_names.append(column_name)
        meta_columns[column_name] = combined_pred
        if holdout_index is None:
            holdout_index = holdout_pred.index

    if holdout_index is None or len(holdout_index) == 0:
        return None

    meta_X = pd.DataFrame(meta_columns, index=y.index)
    meta_X["prediction_mean"] = meta_X[prediction_column_names].mean(axis=1)
    meta_X["prediction_dispersion"] = meta_X[prediction_column_names].std(axis=1)
    meta_X["break_flag"] = context["break_flag"].astype(float)
    meta_X["macro_event_window"] = context["macro_event_window"].astype(float)
    meta_X = meta_X.dropna()

    meta_holdout_index = holdout_index.intersection(meta_X.index)
    if len(meta_holdout_index) < 50 or len(meta_X) <= len(meta_holdout_index) + 200:
        return None

    meta_y = y.loc[meta_X.index]
    meta_context = context.loc[meta_X.index]
    meta_direction_prob = direction_prob_positive.loc[meta_X.index] if direction_prob_positive is not None else None

    meta_builder = lambda: make_elastic_net(alpha=0.01, l1_ratio=0.15)
    predictions, metrics, holdout_results = walk_forward_backtest(
        meta_builder,
        meta_X,
        meta_y,
        meta_context,
        config.n_splits,
        config.horizon,
        config.uncertainty_gate,
        config.break_threshold_mult,
        config.event_threshold_mult,
        config.stability_penalty_weight,
        vol_scale=None,
        direction_prob_positive=meta_direction_prob,
        direction_gate_prob=config.direction_gate_prob,
        use_meta_label_filter=config.use_meta_label_filter,
        meta_label_gate_prob=config.meta_label_gate_prob,
        min_size_multiplier=config.min_size_multiplier,
        max_size_multiplier=config.max_size_multiplier,
        size_sigmoid_k=config.size_sigmoid_k,
        size_conf_mid=config.size_conf_mid,
        use_purged_cv=config.use_purged_cv,
        holdout_fraction=config.holdout_fraction,
        holdout_size=len(meta_holdout_index),
    )
    metrics = apply_prediction_stability_penalty(
        predictions,
        metrics,
        meta_y,
        meta_context,
        config,
        meta_direction_prob,
    )

    fitted_model = meta_builder()
    fitted_model.fit(meta_X, meta_y)
    final_prediction = float(fitted_model.predict(meta_X.iloc[[-1]])[0])

    return {
        "predictions": predictions,
        "metrics": metrics,
        "final_prediction": final_prediction,
        "feature_importance": summarize_feature_importance(fitted_model, list(meta_X.columns)),
        "members": selected_names,
        "holdout_results": holdout_results,
    }


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

    adaptive_edge = metrics["edge_threshold"] * (
        1.0
        + config.break_threshold_mult * float(break_alarm)
        + config.event_threshold_mult * float(event_window)
    )
    edge_excess = (abs(final_prediction) - adaptive_edge) / max(sigma, 1e-8)
    confidence_curve = 1.0 / (1.0 + np.exp(-config.size_sigmoid_k * (edge_excess - config.size_conf_mid)))
    dynamic_size = config.min_size_multiplier + (config.max_size_multiplier - config.min_size_multiplier) * confidence_curve
    recommended_size = float(size_multiplier * dynamic_size)

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
        "recommended_size_multiplier": recommended_size,
    }

def deployment_score(payload: dict[str, Any]) -> float:
    cv_score = float(payload["metrics"]["selection_score"])
    holdout_score = float(payload.get("holdout_results", {}).get("holdout_metrics", {}).get("selection_score", 0.0))
    holdout_penalty = 1.5 * max(0.0, -holdout_score)
    return 0.65 * cv_score + 0.35 * holdout_score - holdout_penalty


def evaluate_models(config: RunConfig, dataset: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    X = dataset.drop(columns=["dS_H", "S_now"])
    y = dataset["dS_H"]
    vol_scale: pd.Series | None = dataset["spread_vol_20"].clip(lower=1e-8) if config.normalize_label else None
    context = dataset[["macro_event_window", "break_flag", "S_now", "spread_vol_20"]].copy()
    direction_prob_positive: pd.Series | None = None
    if config.use_direction_filter:
        direction_prob_positive, _ = walk_forward_direction_filter(X, y, config.n_splits)

    prediction_table = pd.DataFrame(index=dataset.index)
    prediction_table["actual_dS_H"] = y
    if direction_prob_positive is not None:
        prediction_table["direction_prob_positive"] = direction_prob_positive

    evaluated: dict[str, Any] = {}
    for model_name, builder in model_factories(config).items():
        predictions, metrics, holdout_results = walk_forward_backtest(
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
            use_meta_label_filter=config.use_meta_label_filter,
            meta_label_gate_prob=config.meta_label_gate_prob,
            min_size_multiplier=config.min_size_multiplier,
            max_size_multiplier=config.max_size_multiplier,
            size_sigmoid_k=config.size_sigmoid_k,
            size_conf_mid=config.size_conf_mid,
            use_purged_cv=config.use_purged_cv,
            holdout_fraction=config.holdout_fraction,
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
            "holdout_results": holdout_results,
        }
        prediction_table[f"{model_name}_pred"] = predictions

    if config.use_regime_router:
        router_payload = build_regime_router_candidate(
            evaluated,
            y,
            context,
            config,
            direction_prob_positive,
        )
        if router_payload is not None:
            evaluated["regime_router"] = router_payload
            prediction_table["regime_router_pred"] = router_payload["predictions"]

    stacked_payload = build_stacked_blend_candidate(
        evaluated,
        y,
        context,
        config,
        direction_prob_positive,
    )
    if stacked_payload is not None:
        evaluated["stacked_blend"] = stacked_payload
        prediction_table["stacked_blend_pred"] = stacked_payload["predictions"]

    ranked_names = sorted(
        evaluated,
        key=lambda name: (
            deployment_score(evaluated[name]),
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name].get("holdout_results", {}).get("holdout_metrics", {}).get("selection_score", 0.0),
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
        reverse=True,
    )

    ensemble_candidate_names = [name for name in ranked_names if "members" not in evaluated[name]]
    if not ensemble_candidate_names:
        ensemble_candidate_names = ranked_names
    ensemble_members = ensemble_candidate_names[: max(1, config.top_k_ensemble)]
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
        horizon=config.horizon,
        uncertainty_gate=config.uncertainty_gate,
        break_threshold_mult=config.break_threshold_mult,
        event_threshold_mult=config.event_threshold_mult,
        direction_prob_positive=direction_prob_positive.loc[ensemble_valid_index]
        if direction_prob_positive is not None
        else None,
        direction_gate_prob=config.direction_gate_prob,
        meta_label_prob=None,
        meta_label_gate_prob=config.meta_label_gate_prob,
        min_size_multiplier=config.min_size_multiplier,
        max_size_multiplier=config.max_size_multiplier,
        size_sigmoid_k=config.size_sigmoid_k,
        size_conf_mid=config.size_conf_mid,
    )
    ensemble_metrics = apply_prediction_stability_penalty(
        ensemble_predictions.loc[ensemble_valid_index],
        ensemble_metrics,
        y,
        context,
        config,
        direction_prob_positive,
    )
    ensemble_final_prediction = float(
        np.dot(member_weights, np.asarray([evaluated[name]["final_prediction"] for name in ensemble_members], dtype=float))
    )

    ensemble_holdout_results: dict[str, Any] | None = None
    holdout_frames: list[pd.Series] = []
    for name in ensemble_members:
        holdout_pred = evaluated[name].get("holdout_results", {}).get("holdout_predictions")
        if holdout_pred is None:
            holdout_frames = []
            break
        holdout_frames.append(holdout_pred.rename(name))

    if holdout_frames:
        ensemble_holdout_frame = pd.concat(holdout_frames, axis=1)
        weighted_holdout_frame = ensemble_holdout_frame.mul(member_weights, axis=1)
        ensemble_holdout_predictions = weighted_holdout_frame.sum(axis=1)
        ensemble_holdout_valid_index = ensemble_holdout_predictions.dropna().index
        if len(ensemble_holdout_valid_index) > 0:
            ensemble_holdout_metrics = summarize_predictions(
                y.loc[ensemble_holdout_valid_index],
                ensemble_holdout_predictions.loc[ensemble_holdout_valid_index],
                context.loc[ensemble_holdout_valid_index],
                horizon=config.horizon,
                uncertainty_gate=config.uncertainty_gate,
                break_threshold_mult=config.break_threshold_mult,
                event_threshold_mult=config.event_threshold_mult,
                direction_prob_positive=direction_prob_positive.loc[ensemble_holdout_valid_index]
                if direction_prob_positive is not None
                else None,
                direction_gate_prob=config.direction_gate_prob,
                meta_label_prob=None,
                meta_label_gate_prob=config.meta_label_gate_prob,
                min_size_multiplier=config.min_size_multiplier,
                max_size_multiplier=config.max_size_multiplier,
                size_sigmoid_k=config.size_sigmoid_k,
                size_conf_mid=config.size_conf_mid,
            )
            ensemble_holdout_results = {
                "holdout_predictions": ensemble_holdout_predictions,
                "holdout_metrics": ensemble_holdout_metrics,
                "holdout_size": len(ensemble_holdout_valid_index),
            }

    evaluated["ensemble_top"] = {
        "predictions": ensemble_predictions,
        "metrics": ensemble_metrics,
        "final_prediction": ensemble_final_prediction,
        "members": ensemble_members,
        "member_weights": {
            name: float(weight) for name, weight in zip(ensemble_members, member_weights, strict=False)
        },
        "feature_importance": [],
        "holdout_results": ensemble_holdout_results,
    }
    prediction_table["ensemble_top_pred"] = ensemble_predictions

    champion_name = max(
        evaluated,
        key=lambda name: (
            deployment_score(evaluated[name]),
            evaluated[name]["metrics"]["selection_score"],
            evaluated[name].get("holdout_results", {}).get("holdout_metrics", {}).get("selection_score", 0.0),
            evaluated[name]["metrics"]["cum_pnl_proxy"],
            -evaluated[name]["metrics"]["rmse"],
        ),
    )
    prediction_table["champion_pred"] = evaluated[champion_name]["predictions"]
    return {"evaluated": evaluated, "champion_name": champion_name}, prediction_table


def tune_objective_hyperparameters(config: RunConfig, dataset: pd.DataFrame) -> tuple[RunConfig, dict[str, Any]]:
    """Tune thresholds toward selection score / Sharpe with trade-rate guardrails."""
    uncertainty_grid = [0.25, 0.35, 0.50, 0.70]
    break_grid = [0.35, 0.60, 0.90]
    event_grid = [0.10, 0.30, 0.50]
    stability_grid = [0.40, 0.60, 0.80]
    meta_gate_grid = [0.54, 0.56, 0.60]
    sign_penalty_grid = [2.0, 2.5, 3.0, 3.5]
    top_k_grid = [3, 4, 5]
    use_router_grid = [True, False]
    min_size_grid = [0.3, 0.5, 0.7]
    max_size_grid = [1.3, 1.5, 1.8]
    all_candidates = list(
        product(
            uncertainty_grid,
            break_grid,
            event_grid,
            stability_grid,
            meta_gate_grid,
            sign_penalty_grid,
            top_k_grid,
            use_router_grid,
            min_size_grid,
            max_size_grid,
        )
    )

    trial_count = max(1, min(config.tuning_trials, len(all_candidates)))
    current_candidate = (
        float(config.uncertainty_gate),
        float(config.break_threshold_mult),
        float(config.event_threshold_mult),
        float(config.stability_penalty_weight),
        float(config.meta_label_gate_prob),
        float(config.sign_error_penalty),
        int(config.top_k_ensemble),
        bool(config.use_regime_router),
        float(config.min_size_multiplier),
        float(config.max_size_multiplier),
    )
    remaining_candidates = [candidate for candidate in all_candidates if candidate != current_candidate]
    rng = random.Random(7)
    sampled = [current_candidate]
    if trial_count > 1 and remaining_candidates:
        sampled.extend(rng.sample(remaining_candidates, k=min(trial_count - 1, len(remaining_candidates))))

    best_cfg = config
    best_score = -np.inf
    best_details: dict[str, Any] = {
        "trials": [],
        "best_trial": None,
    }

    for (
        uncertainty_gate,
        break_mult,
        event_mult,
        stability_weight,
        meta_gate,
        sign_penalty,
        top_k,
        use_router,
        min_size,
        max_size,
    ) in sampled:
        candidate_cfg = clone_config(
            config,
            uncertainty_gate=float(uncertainty_gate),
            break_threshold_mult=float(break_mult),
            event_threshold_mult=float(event_mult),
            stability_penalty_weight=float(stability_weight),
            meta_label_gate_prob=float(meta_gate),
            sign_error_penalty=float(sign_penalty),
            top_k_ensemble=int(top_k),
            use_regime_router=bool(use_router),
            min_size_multiplier=float(min_size),
            max_size_multiplier=float(max_size),
            run_objective_tuning=False,
        )
        evaluation, _ = evaluate_models(candidate_cfg, dataset)
        champion = evaluation["champion_name"]
        metrics = evaluation["evaluated"][champion]["metrics"]
        trade_rate = float(metrics["trade_rate"])
        holdout_metrics = evaluation["evaluated"][champion].get("holdout_results", {}).get("holdout_metrics", {})
        holdout_score = float(holdout_metrics.get("selection_score", 0.0))
        holdout_trade_rate = float(holdout_metrics.get("trade_rate", 0.0))

        trade_floor_penalty = max(0.0, 0.25 - trade_rate)
        trade_ceiling_penalty = max(0.0, trade_rate - 0.45)
        holdout_trade_ceiling_penalty = max(0.0, holdout_trade_rate - 0.55)
        trade_rate_ratio = (holdout_trade_rate + 1e-9) / (trade_rate + 1e-9)
        ratio_mismatch = abs(float(np.log(max(trade_rate_ratio, 1e-9))))
        ratio_tolerance = float(np.log(1.25))
        trade_divergence_penalty = max(0.0, ratio_mismatch - ratio_tolerance)

        # Score emphasizes selection quality and holdout robustness while enforcing trade-rate guardrails.
        objective_score = (
            float(metrics["selection_score"])
            + 0.45 * float(metrics["sharpe_proxy"])
            + 0.35 * holdout_score
            - 1.25 * max(0.0, -holdout_score)
            - 0.80 * trade_floor_penalty
            - 2.00 * trade_ceiling_penalty
            - 2.50 * holdout_trade_ceiling_penalty
            - 0.80 * trade_divergence_penalty
        )

        trial_record = {
            "uncertainty_gate": float(uncertainty_gate),
            "break_threshold_mult": float(break_mult),
            "event_threshold_mult": float(event_mult),
            "stability_penalty_weight": float(stability_weight),
            "meta_label_gate_prob": float(meta_gate),
            "sign_error_penalty": float(sign_penalty),
            "top_k_ensemble": int(top_k),
            "use_regime_router": bool(use_router),
            "min_size_multiplier": float(min_size),
            "max_size_multiplier": float(max_size),
            "champion": champion,
            "selection_score": float(metrics["selection_score"]),
            "sharpe_proxy": float(metrics["sharpe_proxy"]),
            "trade_rate": trade_rate,
            "holdout_trade_rate": holdout_trade_rate,
            "trade_rate_ratio": float(trade_rate_ratio),
            "trade_floor_penalty": float(trade_floor_penalty),
            "trade_ceiling_penalty": float(trade_ceiling_penalty),
            "holdout_trade_ceiling_penalty": float(holdout_trade_ceiling_penalty),
            "trade_divergence_penalty": float(trade_divergence_penalty),
            "holdout_selection_score": holdout_score,
            "objective_score": float(objective_score),
        }
        best_details["trials"].append(trial_record)

        if objective_score > best_score:
            best_score = objective_score
            best_cfg = candidate_cfg
            best_details["best_trial"] = trial_record

    return best_cfg, best_details


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    dataset = build_feature_frame(config)
    effective_config = config
    tuning_report: dict[str, Any] | None = None
    if config.run_objective_tuning:
        effective_config, tuning_report = tune_objective_hyperparameters(config, dataset)
    evaluation, prediction_table = evaluate_models(effective_config, dataset)
    latest_row = dataset.iloc[-1]

    model_results: dict[str, Any] = {}
    for model_name, payload in evaluation["evaluated"].items():
        holdout_payload = payload.get("holdout_results", {})
        holdout_metrics = holdout_payload.get("holdout_metrics")
        model_results[model_name] = {
            "metrics": payload["metrics"],
            "deployment_score": deployment_score(payload),
            "live": build_quote_snapshot(config, latest_row, payload["metrics"], payload["final_prediction"]),
            "feature_importance": payload["feature_importance"],
        }
        if holdout_metrics is not None:
            model_results[model_name]["holdout_metrics"] = holdout_metrics
            model_results[model_name]["holdout_size"] = int(holdout_payload.get("holdout_size", 0))
        if "members" in payload:
            model_results[model_name]["members"] = payload["members"]
        if "member_weights" in payload:
            model_results[model_name]["member_weights"] = payload["member_weights"]

    leaderboard = [
        {
            "model": model_name,
            "deployment_score": float(deployment_score(payload)),
            "selection_score": float(payload["metrics"]["selection_score"]),
            "cum_pnl_proxy": float(payload["metrics"]["cum_pnl_proxy"]),
            "rmse": float(payload["metrics"]["rmse"]),
            "traded_hit_rate": float(payload["metrics"]["traded_hit_rate"]),
            "holdout_selection_score": float(
                payload.get("holdout_results", {}).get("holdout_metrics", {}).get("selection_score", 0.0)
            ),
        }
        for model_name, payload in sorted(
            evaluation["evaluated"].items(),
            key=lambda item: (
                deployment_score(item[1]),
                item[1]["metrics"]["selection_score"],
                item[1].get("holdout_results", {}).get("holdout_metrics", {}).get("selection_score", 0.0),
                item[1]["metrics"]["cum_pnl_proxy"],
                -item[1]["metrics"]["rmse"],
            ),
            reverse=True,
        )
    ]

    output = {
        "config": {
            **asdict(effective_config),
            "output_dir": str(effective_config.output_dir),
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
    if tuning_report is not None:
        output["tuning_report"] = tuning_report
    write_outputs(effective_config.output_dir, output, prediction_table)
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
    parser.add_argument("--top-k-ensemble", type=int, default=5, help="Number of top models to average in the ensemble.")
    parser.add_argument(
        "--uncertainty-gate",
        type=float,
        default=0.25,
        help="Minimum normalized confidence (|pred|/sigma) required to trade.",
    )
    parser.add_argument(
        "--break-threshold-mult",
        type=float,
        default=0.35,
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
        default=0.40,
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
        "--no-regime-router",
        action="store_true",
        help="Disable regime-router candidate that switches between calm/stress specialists.",
    )
    parser.add_argument(
        "--normalize-label",
        action="store_true",
        help="Enable volatility-normalized label training (off by default; harms linear models).",
    )
    parser.add_argument(
        "--sign-error-penalty",
        type=float,
        default=3.5,
        help="XGBoost sign-error loss multiplier (>1 penalises wrong-direction predictions).",
    )
    parser.add_argument(
        "--min-size-multiplier",
        type=float,
        default=1.0,
        help="Minimum dynamic size multiplier when a trade is taken.",
    )
    parser.add_argument(
        "--max-size-multiplier",
        type=float,
        default=1.0,
        help="Maximum dynamic size multiplier when confidence is high.",
    )
    parser.add_argument(
        "--size-sigmoid-k",
        type=float,
        default=3.0,
        help="Steepness of confidence-to-size sigmoid.",
    )
    parser.add_argument(
        "--size-conf-mid",
        type=float,
        default=0.60,
        help="Midpoint of edge-excess confidence for size scaling.",
    )
    parser.add_argument(
        "--use-purged-cv",
        action="store_true",
        default=True,
        help="Enable purged walk-forward CV to eliminate lookahead bias (default: enabled).",
    )
    parser.add_argument(
        "--no-purged-cv",
        action="store_true",
        help="Disable purged walk-forward CV and use standard time-series splits.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.15,
        help="Fraction of data to reserve as locked holdout test block.",
    )
    parser.add_argument(
        "--use-meta-label-filter",
        action="store_true",
        help="Enable meta-label trade filter for quality selection (experimental).",
    )
    parser.add_argument(
        "--meta-label-gate-prob",
        type=float,
        default=0.54,
        help="Minimum meta-label quality probability required for trade entry.",
    )
    parser.add_argument(
        "--run-objective-tuning",
        action="store_true",
        help="Run constrained objective-aligned tuning before final evaluation.",
    )
    parser.add_argument(
        "--tuning-trials",
        type=int,
        default=24,
        help="Number of tuning trials to run when objective tuning is enabled.",
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
        use_regime_router=not args.no_regime_router,
        min_size_multiplier=args.min_size_multiplier,
        max_size_multiplier=args.max_size_multiplier,
        size_sigmoid_k=args.size_sigmoid_k,
        size_conf_mid=args.size_conf_mid,
        use_purged_cv=not args.no_purged_cv,
        holdout_fraction=args.holdout_fraction,
        use_meta_label_filter=args.use_meta_label_filter,
        meta_label_gate_prob=args.meta_label_gate_prob,
        run_objective_tuning=args.run_objective_tuning,
        tuning_trials=args.tuning_trials,
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