# Backtest Evaluation Results — Forecasting Engine

Measured by running the pipeline — not asserted. Reproduce with:

```bash
python scripts/run_backtest.py   # writes results/backtest_sample.json
```

## Evaluation setup

| Item | Value |
|------|-------|
| Series | `business_demand.csv`: 1,095 daily rows, 2021-01-01 → 2023-12-31, seed 42 (trend + yearly/weekly seasonality + promo + noise + 1.5% anomalies) |
| Forecast horizon | 30 days (point forecast + intervals) |
| Validation | 4 expanding-window folds from 60% initial train; no random splits |
| Selection metric | sMAPE (horizon-weighted average across folds) |
| Fold validation windows | F1: 2022-10-20→2023-02-05 · F2: →2023-05-25 · F3: →2023-09-12 · F4: →2023-12-31 |

## Headline result (verified)

**ARIMA wins with 11.35% aggregated sMAPE**, beating the seasonal-naive
baseline (12.29%) by ~0.9pp and plain naive (15.96%) by ~4.6pp. The
improvement over baselines — not the isolated 11.35% — is the result.

## Full comparison (aggregated across folds)

| Rank | Model | sMAPE | MAE | RMSE | WAPE |
|------|-------|------:|----:|-----:|-----:|
| 1 | ARIMA | 11.35 | 126.5 | 152.0 | 11.08 |
| 2 | Seasonal Naive | 12.29 | 136.0 | 163.2 | 11.89 |
| 3 | XGBoost | 12.55 | 138.7 | 168.9 | 12.24 |
| 4 | ETS | 12.80 | 136.6 | 162.9 | 12.37 |
| 5 | Naive | 15.96 | 183.6 | 214.0 | 16.02 |

## Per-fold sMAPE (why aggregates, not single folds)

| Model | F1 | F2 | F3 | F4 |
|-------|---:|---:|---:|---:|
| ARIMA | 17.50 | 9.73 | 5.92 | 12.32 |
| Seasonal Naive | 17.00 | 8.67 | 6.67 | 16.84 |
| XGBoost | 17.86 | 10.78 | 9.03 | 12.55 |
| ETS | 10.76 | 12.90 | 18.56 | 8.96 |
| Naive | 17.66 | 14.84 | 9.03 | 22.32 |

No model wins every fold: seasonal-naive beats ARIMA in F2, ETS collapses
in F3 (18.56) yet wins F4. ARIMA is selected because it is the most
consistent — lowest horizon-weighted average — which is exactly what the
backtesting stage is for. A single train/test split could have crowned any
of three models depending on the chosen window.

## Winner residual diagnostics (ARIMA refit on full history, n=1095)

| Check | Value | Reading |
|-------|-------|---------|
| Mean / std | 1.35 / 92.7 | approximately unbiased |
| Durbin-Watson | 1.97 | no lag-1 autocorrelation |
| Ljung-Box (lag 10) | p ≈ 0.0 | **rejects** whiteness — predictable structure remains |
| Jarque-Bera | p ≈ 0.0, excess kurtosis 14.0 | heavy tails from injected anomalies |

Honest reading: the model captures trend/seasonality (unbiased, no lag-1
correlation) but leaves weekly structure and anomaly spikes on the table.
Prediction intervals built on a Gaussian assumption are therefore
optimistic in anomaly weeks — documented in the API response metadata.

## Limitations

- One synthetic series: rankings may not transfer to series with different seasonality, noise, or structural breaks.
- sMAPE is the selection metric by default but can mislead near-zero actuals; MAE/RMSE/WAPE are reported alongside for that reason.
- Uncertainty for XGBoost/naive baselines is a residual-based approximation, not a native predictive distribution.
- No online adaptation: if the process drifts, validation metrics will not reflect future performance.
