"""
These tests exist specifically to catch look-ahead bias and portfolio
construction bugs — the failure modes that make a backtest's results
meaningless even when the code "runs fine." If you only run one test file
before an interview, run this one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG
from signals.quality_value import ValueQualityComposite
from backtest.engine import BacktestEngine
from backtest.costs import compute_turnover, apply_costs


def make_fundamentals_with_future_leak():
    """One row reported AFTER the as_of_date we'll query, plus one reported before.
    The engine must return only the earlier one."""
    return pd.DataFrame(
        [
            {"ticker": "A", "fiscal_period_end": pd.Timestamp("2020-01-01"),
             "report_date": pd.Timestamp("2020-02-01"), "earnings_yield": 0.05,
             "book_to_market": 0.5, "roe": 0.1, "gross_margin": 0.3,
             "debt_to_equity": 0.5, "earnings_stability": 0.6},
            {"ticker": "A", "fiscal_period_end": pd.Timestamp("2020-04-01"),
             "report_date": pd.Timestamp("2020-05-15"), "earnings_yield": 0.99,
             "book_to_market": 0.99, "roe": 0.99, "gross_margin": 0.99,
             "debt_to_equity": 0.01, "earnings_stability": 0.99},
        ]
    )


def test_pit_snapshot_excludes_future_data():
    """THE critical test: querying as-of 2020-03-01 must NOT see the
    2020-05-15-reported row, even though its fiscal_period_end (2020-04-01)
    doesn't matter — only report_date governs visibility."""
    fundamentals = make_fundamentals_with_future_leak()
    prices = pd.DataFrame({"A": [100, 101]}, index=pd.bdate_range("2020-01-01", periods=2))
    signal = ValueQualityComposite(DEFAULT_CONFIG)
    engine = BacktestEngine(DEFAULT_CONFIG, signal, prices, fundamentals)

    snapshot = engine._get_pit_snapshot(pd.Timestamp("2020-03-01"))
    assert snapshot.loc["A", "earnings_yield"] == 0.05, \
        "LOOK-AHEAD BUG: engine leaked a fundamentals row reported in the future!"


def test_pit_snapshot_updates_after_report_date():
    fundamentals = make_fundamentals_with_future_leak()
    prices = pd.DataFrame({"A": [100, 101]}, index=pd.bdate_range("2020-01-01", periods=2))
    signal = ValueQualityComposite(DEFAULT_CONFIG)
    engine = BacktestEngine(DEFAULT_CONFIG, signal, prices, fundamentals)

    snapshot = engine._get_pit_snapshot(pd.Timestamp("2020-06-01"))
    assert snapshot.loc["A", "earnings_yield"] == 0.99, \
        "Engine failed to pick up newly-available data after its report date."


def test_turnover_zero_when_weights_unchanged():
    w = pd.Series({"A": 0.5, "B": -0.5})
    assert compute_turnover(w, w) == 0.0


def test_turnover_full_on_complete_reshuffle():
    w1 = pd.Series({"A": 0.5, "B": -0.5})
    w2 = pd.Series({"C": 0.5, "D": -0.5})
    # Completely disjoint portfolios: one-way turnover should be 1.0 (100%)
    assert compute_turnover(w2, w1) == 1.0


def test_apply_costs_reduces_return():
    net = apply_costs(gross_return=0.05, turnover=0.5, cost_bps_per_turnover=10.0)
    assert net < 0.05
    assert abs(net - (0.05 - 0.5 * 0.001)) < 1e-9


def test_long_short_weights_sum_to_zero():
    """Dollar-neutral construction: long and short legs should net to ~0 exposure."""
    from config import BacktestConfig
    cfg = BacktestConfig()
    signal = ValueQualityComposite(cfg)
    scores = pd.Series(np.linspace(-2, 2, 50), index=[f"S{i}" for i in range(50)])
    prices = pd.DataFrame({t: [100] for t in scores.index}, index=[pd.Timestamp("2020-01-01")])
    fundamentals = pd.DataFrame(columns=["ticker", "report_date"])
    engine = BacktestEngine(cfg, signal, prices, fundamentals)
    weights = engine._construct_weights(scores)
    assert abs(weights.sum()) < 1e-9, f"Expected ~0 net exposure, got {weights.sum()}"


if __name__ == "__main__":
    test_pit_snapshot_excludes_future_data()
    test_pit_snapshot_updates_after_report_date()
    test_turnover_zero_when_weights_unchanged()
    test_turnover_full_on_complete_reshuffle()
    test_apply_costs_reduces_return()
    test_long_short_weights_sum_to_zero()
    print("All backtest mechanics tests passed.")