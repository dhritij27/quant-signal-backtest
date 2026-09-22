"""
Simple linear transaction cost model: cost = turnover * bps_per_turnover.

Turnover here is defined as the sum of absolute weight changes between two
consecutive rebalances (one-way). This is a simplification of real trading
costs (which depend on liquidity, order size, market impact, spread) but is
the standard starting point in academic/practitioner backtests and is
explicitly what the JD means by "trading cost forecasts" at a basic level —
document this as a simplification and a natural extension point.
"""
import pandas as pd


def compute_turnover(weights_t: pd.Series, weights_t_minus_1: pd.Series) -> float:
    """One-way turnover between two rebalance dates' portfolio weights."""
    all_tickers = weights_t.index.union(weights_t_minus_1.index)
    wt = weights_t.reindex(all_tickers, fill_value=0.0)
    wt1 = weights_t_minus_1.reindex(all_tickers, fill_value=0.0)
    return 0.5 * (wt - wt1).abs().sum()


def apply_costs(gross_return: float, turnover: float, cost_bps_per_turnover: float) -> float:
    cost = turnover * (cost_bps_per_turnover / 10000.0)
    return gross_return - cost