"""
Value/Quality composite signal.

Value leg: earnings_yield, book_to_market (higher = cheaper = more attractive)
Quality leg: roe, gross_margin, earnings_stability (higher = better),
             debt_to_equity (LOWER = better, so we invert it before z-scoring)

Each factor is winsorized and cross-sectionally z-scored independently
(so no single factor's scale dominates), then averaged within its leg,
then the two legs are combined with config-driven weights.
"""
import pandas as pd
from signals.base import Signal, winsorize, zscore


class ValueQualityComposite(Signal):
    name = "value_quality_composite"

    def __init__(self, config):
        self.cfg = config

    def score(self, fundamentals_snapshot: pd.DataFrame) -> pd.Series:
        df = fundamentals_snapshot.copy()

        # Quality leg needs debt_to_equity inverted (lower leverage = higher quality)
        if "debt_to_equity" in df.columns:
            df["debt_to_equity_inv"] = -df["debt_to_equity"]

        value_z = pd.DataFrame(index=df.index)
        for factor in self.cfg.value_factors:
            if factor not in df.columns:
                continue
            col = winsorize(df[factor].dropna(), self.cfg.winsorize_pct)
            value_z[factor] = zscore(col)

        quality_z = pd.DataFrame(index=df.index)
        for factor in self.cfg.quality_factors:
            if factor not in df.columns:
                continue
            col = winsorize(df[factor].dropna(), self.cfg.winsorize_pct)
            quality_z[factor] = zscore(col)

        value_score = value_z.mean(axis=1, skipna=True)
        quality_score = quality_z.mean(axis=1, skipna=True)

        composite = (
            self.cfg.value_weight * value_score.fillna(0)
            + self.cfg.quality_weight * quality_score.fillna(0)
        )
        # Drop names with no usable data at all (both legs entirely NaN)
        has_data = value_z.notna().any(axis=1) | quality_z.notna().any(axis=1)
        composite = composite[has_data]
        return composite.rename("composite_score")