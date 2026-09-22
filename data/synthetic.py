"""
Synthetic data generator.

Claude's build/test sandbox has no live market data access, so this module
generates realistic-shaped synthetic prices + quarterly fundamentals purely
to validate pipeline LOGIC (no look-ahead bugs, correct quantile construction,
correct cost accounting, etc.) before real data is substituted in.

The synthetic fundamentals are constructed so that a *true* underlying
value/quality factor exists and is correlated with forward returns — this
lets us sanity-check that the pipeline actually recovers a signal when one
is present (a positive control), which is a different and more useful test
than "does the code run."
"""
import numpy as np
import pandas as pd


def generate_universe(n_stocks: int = 500, seed: int = 42) -> pd.DataFrame:
    """Static per-stock latent characteristics (don't change over time)."""
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:04d}" for i in range(n_stocks)]
    # Latent "true quality" and "true value" scores per stock (persistent skill/characteristic)
    true_quality = rng.normal(0, 1, n_stocks)
    true_value = rng.normal(0, 1, n_stocks)
    sector = rng.choice(
        ["Tech", "Financials", "Healthcare", "Industrials", "Consumer", "Energy"],
        size=n_stocks,
    )
    return pd.DataFrame(
        {"ticker": tickers, "true_quality": true_quality, "true_value": true_value, "sector": sector}
    )


def generate_prices(universe: pd.DataFrame, start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    """
    Daily prices. Forward monthly returns are nudged by the latent value+quality
    score plus noise, so the signal has *some* true predictive power to recover,
    plus a lot of noise so the backtest doesn't look unrealistically clean.
    """
    rng = np.random.default_rng(seed + 1)
    dates = pd.bdate_range(start_date, end_date)
    n_days = len(dates)
    n_stocks = len(universe)

    # Monthly "true alpha" contribution from the latent factor (small, realistic magnitude)
    monthly_alpha = 0.004 * (universe["true_quality"].values + universe["true_value"].values) / 2

    daily_alpha = np.repeat(monthly_alpha / 21, n_days).reshape(n_stocks, n_days) if False else None
    # Build day-by-day to keep memory reasonable and allow per-day noise
    prices = np.zeros((n_days, n_stocks))
    prices[0, :] = 100.0
    daily_vol = rng.uniform(0.012, 0.025, n_stocks)  # per-stock idiosyncratic vol
    market_factor = rng.normal(0.0003, 0.01, n_days)  # common market return each day

    for t in range(1, n_days):
        idio = rng.normal(0, 1, n_stocks) * daily_vol
        daily_ret = market_factor[t] + (monthly_alpha / 21.0) + idio
        prices[t, :] = prices[t - 1, :] * (1 + daily_ret)

    price_df = pd.DataFrame(prices, index=dates, columns=universe["ticker"])
    return price_df


def generate_fundamentals(universe: pd.DataFrame, start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    """
    Quarterly fundamentals per ticker, in LONG format:
    columns: ticker, fiscal_period_end, report_date, earnings_yield, book_to_market,
             roe, gross_margin, debt_to_equity, earnings_stability

    report_date = fiscal_period_end + a random filing delay (simulates real-world
    reporting lag; the backtest engine must respect report_date, not period_end).
    """
    rng = np.random.default_rng(seed + 2)
    quarter_ends = pd.date_range(start_date, end_date, freq="QE")
    rows = []
    for _, stock in universe.iterrows():
        tq, tv = stock["true_quality"], stock["true_value"]
        for q_end in quarter_ends:
            filing_delay = rng.integers(30, 90)  # days after quarter end the filing hits
            report_date = q_end + pd.Timedelta(days=int(filing_delay))
            rows.append(
                {
                    "ticker": stock["ticker"],
                    "fiscal_period_end": q_end,
                    "report_date": report_date,
                    "earnings_yield": 0.05 + 0.02 * tv + rng.normal(0, 0.01),
                    "book_to_market": 0.5 + 0.15 * tv + rng.normal(0, 0.05),
                    "roe": 0.10 + 0.04 * tq + rng.normal(0, 0.02),
                    "gross_margin": 0.35 + 0.05 * tq + rng.normal(0, 0.03),
                    "debt_to_equity": max(0.05, 0.6 - 0.15 * tq + rng.normal(0, 0.1)),
                    "earnings_stability": 0.7 + 0.1 * tq + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic_dataset(n_stocks: int = 500, start_date: str = "2015-01-01", end_date: str = "2025-01-01"):
    universe = generate_universe(n_stocks)
    prices = generate_prices(universe, start_date, end_date)
    fundamentals = generate_fundamentals(universe, start_date, end_date)
    return universe, prices, fundamentals