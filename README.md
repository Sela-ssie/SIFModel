# SIFModel

This project implements a stronger version of the guide's workflow for the AIJ26 spread:

$$
S_t = \frac{\mathrm{SOXX}_t}{4} - \mathrm{IGV}_t
$$

The baseline model forecasts the $H$-day-ahead spread change:

$$
\Delta S_{t,H} = S_{t+H} - S_t
$$

It pulls public data, engineers spread, market, rates, volatility, and event-risk features, runs a walk-forward model tournament, evaluates several ElasticNet variants plus optional XGBoost challengers, builds an ensemble of the top models, and converts the champion forecast into a fair value, quote band, and risk-adjusted size recommendation.

## What It Does

- Loads daily SOXX and IGV prices from Stooq.
- Loads QQQ and SPY from Stooq plus `DGS10`, `DGS2`, and `VIXCLS` from FRED.
- Builds spread, momentum, volatility, correlation, beta-gap, rates, curve, VIX, and event-window features.
- Evaluates multiple candidate models with walk-forward time-series splits instead of a single fixed baseline.
- Produces a leaderboard using out-of-sample RMSE, traded hit rate, and a risk-aware PnL proxy.
- Builds an ensemble from the top-ranked models and selects a champion automatically.
- Flags macro-event windows and break-like market conditions to widen quotes and reduce size.
- Saves results to `outputs/latest_run.json` and `outputs/latest_predictions.csv`.

## Install

```bash
pip install -e .
```

Recommended stronger setup:

```bash
pip install -e .[full]
```

## Run

Baseline linear-only mode:

```bash
python run_model.py --horizon 15
```

Best current model-tournament mode:

```bash
python run_model.py --horizon 15 --use-xgboost --use-lightgbm --top-k-ensemble 4
```

Installed-package equivalent:

```bash
python -m sif_model --horizon 15 --use-xgboost --use-lightgbm --top-k-ensemble 4
```

## Outputs

The run writes:

- `outputs/latest_run.json`: metrics, live estimate, and risk signals.
- `outputs/latest_predictions.csv`: aligned walk-forward predictions.

The JSON output includes:

- `champion_model`: best out-of-sample model in the tournament.
- `leaderboard`: ranked candidate models and ensemble.
- `models.*.feature_importance`: top drivers for fitted single models.
- `models.*.live`: fair value, bid/ask, signal-to-noise, and size guidance.

If you use the root launcher, it prepends `src` to `sys.path` so you can run the model immediately.


The live section includes:

- current spread
- predicted $\Delta S$
- fair value
- bid and ask
- signal-to-noise ratio
- event-window flag
- break-alarm flag
- recommended size multiplier

## Current Best Run

Using `--use-xgboost --use-lightgbm --top-k-ensemble 4`, the latest validated run selected `regime_router` as champion by switching between calm-regime and stress-regime specialists. That setup improved selection score, cumulative PnL proxy, and traded hit rate versus the prior single-model champion while keeping RMSE in a similar range.
