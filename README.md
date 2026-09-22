# quant-signal-backtest

### Systematic Value/Quality Signal Research & Backtest Pipeline

A research infrastructure project: a reusable signal library, a point-in-time
walk-forward backtest engine, and a transaction cost model built to mirror
how a quant research team actually evaluates investment ideas, rather than a
one-off "train a model, report accuracy" exercise.

## What this is

- **Signal**: a value/quality composite (earnings yield, book-to-market,
  ROE, gross margin, leverage, earnings stability), cross-sectionally
  z-scored and combined, long top quintile / short bottom quintile.
- **Backtest engine**: monthly walk-forward rebalancing that enforces
  point-in-time fundamentals, a stock's fundamentals are only visible to
  the engine once their (approximate) filing date has passed, not the
  instant the fiscal quarter ends. See "Methodology notes" below.
- **Cost model**: linear bps-per-unit-of-turnover, applied every rebalance,
  reported alongside gross performance so the cost drag is explicit rather
  than hidden.
- **Statistics**: Sharpe ratio, max drawdown, and a t-test on mean period
  return — because a nice looking equity curve isn't the same as a
  statistically distinguishable-from-noise signal.

## Status

Architecture, signal logic, backtest engine, and cost model are built and
unit-tested, including a dedicated test suite (`tests/test_backtest.py`)
that specifically checks for look-ahead bias in the point-in-time
fundamentals logic. Validated end-to-end on a synthetic dataset with a
known planted signal (positive control) — the pipeline correctly recovers
it with a statistically significant t-stat. **Real S&P 500 data has not
yet been run** (see "Running on real data" below) — that's the next step.

## Repo structure

```
config.py                  # all tunable parameters in one place
data/
  synthetic.py              # synthetic data generator (for logic validation)
  loader.py                 # real data loader (yfinance) — run locally
signals/
  base.py                   # Signal interface + winsorize/zscore helpers
  quality_value.py           # the value/quality composite signal
backtest/
  engine.py                  # walk-forward engine, point-in-time enforcement
  costs.py                   # transaction cost model
  metrics.py                 # Sharpe, drawdown, t-stat, etc.
tests/
  test_signal.py              # signal construction correctness
  test_backtest.py            # look-ahead bias + portfolio mechanics checks
scripts/
  run_backtest.py             # CLI entry point (synthetic or real mode)
```

## Methodology notes (read before trusting the numbers)

- **Point-in-time fundamentals lag**: real fundamentals data has a
  reporting lag — a company's Q1 numbers aren't public the day Q1 ends.
  `data/loader.py` approximates this with a flat 95-day lag
  (`config.fundamentals_lag_days`) since yfinance doesn't expose true SEC
  filing dates. This is a documented approximation, not true point-in-time
  data — see the docstring in `data/loader.py` for how to swap in a real
  point-in-time source (e.g. SEC EDGAR).
- **Survivorship bias**: the real-data loader pulls *today's* S&P 500
  constituents and backtests them over the full historical window. This
  means the backtest implicitly excludes companies that were delisted,
  acquired, or dropped from the index — which biases results upward, since
  survivors tend to have done better. Fixing this requires point-in-time
  index membership data, which I've noted as a clear "future work" item
  rather than pretending it isn't a limitation.
- **Transaction costs** are modeled as a simple linear bps-per-turnover
  charge. Real costs also depend on order size relative to liquidity and
  market impact, which this model doesn't capture — a reasonable v2
  extension.

## Running on synthetic data (works anywhere, no internet needed)

```bash
pip install -r requirements.txt
python scripts/run_backtest.py --mode synthetic
python -m pytest tests/ -v      # or: python tests/test_backtest.py
```

## Running on real data (do this locally)

```bash
pip install -r requirements.txt
python scripts/run_backtest.py --mode real
```

This pulls S&P 500 tickers from Wikipedia, then prices and quarterly
fundamentals via `yfinance`. The fundamentals pull is slow (one API call
per ticker) and results are cached to `data/cache/` so repeated runs while
tuning the signal don't re-hit the API every time. Delete the cache folder
to force a refresh.

## Results (synthetic positive control)

Run against synthetic data with a known planted value/quality factor
(so the "true" signal is known), the pipeline recovers it cleanly:

| Metric | Gross | Net (after costs) |
|---|---|---|
| Annualized return | ~4.7% | ~4.6% |
| Sharpe ratio | ~2.0 | ~1.96 |
| t-stat (mean return) | ~6.3 | ~6.2 |
| Avg monthly turnover | ~9.4% | — |

These numbers are **from synthetic data and are not a claim about real
market performance** — their only purpose is confirming the pipeline logic
works correctly (no look-ahead leakage, correct cost accounting, correct
portfolio construction) before running on real data. Real S&P 500 results
will differ substantially and are the actual deliverable of this project.

## Next steps

1. Run `--mode real` locally, inspect fundamentals coverage/quality (some
   tickers will have missing data — decide on an exclusion policy).
2. Compare results across sub-periods (pre/post 2020) to check regime
   stability.
3. Optional extension: add a second signal (e.g. momentum or an
   earnings-call sentiment signal) and combine it with this one, to show
   the infrastructure generalizes beyond a single signal.
