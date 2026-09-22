"""
Base Signal interface. Every signal takes point-in-time fundamentals
(already lagged/filtered upstream) and returns a per-stock composite score
for a single rebalance date. New signals (momentum, earnings drift, etc.)
should subclass this so they plug into the same backtest engine unchanged.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Signal(ABC):
    name: str = "base_signal"

    @abstractmethod
    def score(self, fundamentals_snapshot: pd.DataFrame) -> pd.Series:
        """
        fundamentals_snapshot: one row per ticker, containing only data that
        was actually public as-of the rebalance date (the engine enforces this).

        Returns: pd.Series indexed by ticker, higher = more attractive.
        """
        raise NotImplementedError


def winsorize(s: pd.Series, pct: float) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)