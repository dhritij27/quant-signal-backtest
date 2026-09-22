"""
Central configuration for the value/quality signal research pipeline.
Keep all "tunable knobs" here so backtests are reproducible and diffable.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    # Universe
    universe_name: str = "SP500"
    start_date: str = "2024-06-01"
    end_date: str = "2026-09-01"

    # Rebalancing
    rebalance_freq: str = "M"  # 'M' = monthly

    # Point-in-time fundamentals lag (CRITICAL for avoiding look-ahead bias).
    # Quarterly filings are not public the instant a fiscal quarter ends —
    # companies have up to 45 days (10-Q) or 90 days (10-K) to file with the SEC.
    # We use a conservative flat lag so no signal ever uses data that wasn't
    # actually public yet on the rebalance date.
    fundamentals_lag_days: int = 95

    # Portfolio construction
    n_quantiles: int = 5
    long_quantile: int = 5   # top quintile (highest composite score)
    short_quantile: int = 1  # bottom quintile
    long_only: bool = False  # if True, ignore short leg (long top quintile vs. universe)

    # Signal weights (within the composite). Must sum to 1.0 across the two legs;
    # within a leg, each sub-factor is equally weighted after z-scoring.
    value_weight: float = 0.5
    quality_weight: float = 0.5
    value_factors: List[str] = field(default_factory=lambda: ["earnings_yield", "book_to_market"])
    quality_factors: List[str] = field(default_factory=lambda: ["roe", "gross_margin", "debt_to_equity_inv", "earnings_stability"])

    # Transaction cost model (simple linear bps-per-turnover model; see backtest/costs.py)
    cost_bps_per_turnover: float = 10.0  # 10 bps per unit of one-way turnover

    # Winsorization / outlier handling for factor z-scores
    winsorize_pct: float = 0.01  # clip to 1st/99th percentile before z-scoring


DEFAULT_CONFIG = BacktestConfig()