import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG
from signals.quality_value import ValueQualityComposite
from signals.base import winsorize, zscore


def make_dummy_snapshot():
    return pd.DataFrame(
        {
            "earnings_yield": [0.02, 0.10, 0.06, 0.04],
            "book_to_market": [0.3, 0.8, 0.5, 0.4],
            "roe": [0.05, 0.20, 0.12, 0.08],
            "gross_margin": [0.2, 0.5, 0.35, 0.3],
            "debt_to_equity": [1.2, 0.3, 0.6, 0.9],
            "earnings_stability": [0.5, 0.9, 0.7, 0.6],
        },
        index=["A", "B", "C", "D"],
    )


def test_zscore_mean_zero():
    s = pd.Series([1, 2, 3, 4, 5])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9


def test_winsorize_clips_extremes():
    s = pd.Series(list(range(100)))
    w = winsorize(s, 0.05)
    assert w.max() <= s.quantile(0.95)
    assert w.min() >= s.quantile(0.05)


def test_composite_score_ranks_best_stock_highest():
    """Stock B has the best value AND quality metrics in every column -> should rank #1."""
    snapshot = make_dummy_snapshot()
    signal = ValueQualityComposite(DEFAULT_CONFIG)
    scores = signal.score(snapshot)
    assert scores.idxmax() == "B", f"Expected B to score highest, got {scores.idxmax()}"


def test_composite_score_handles_missing_columns():
    snapshot = make_dummy_snapshot().drop(columns=["debt_to_equity"])
    signal = ValueQualityComposite(DEFAULT_CONFIG)
    scores = signal.score(snapshot)
    assert len(scores) == 4  # should degrade gracefully, not crash


if __name__ == "__main__":
    test_zscore_mean_zero()
    test_winsorize_clips_extremes()
    test_composite_score_ranks_best_stock_highest()
    test_composite_score_handles_missing_columns()
    print("All signal tests passed.")