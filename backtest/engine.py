"""
Walk-forward backtest engine.

Core design principle: at every rebalance date T, the engine must only use
fundamentals whose `report_date` <= T. This is the single most important
correctness property of the whole project — get it wrong and every result
is a look-ahead-biased fantasy. We enforce it in one place
(_get_pit_snapshot) so it can't accidentally leak elsewhere.
"""
import pandas as pd
import numpy as np

from signals.base import Signal
from backtest.costs import compute_turnover, apply_costs
from backtest import metrics as bt_metrics


class BacktestEngine:
    def __init__(self, config, signal: Signal, prices: pd.DataFrame, fundamentals: pd.DataFrame):
        self.cfg = config
        self.signal = signal
        self.prices = prices
        self.fundamentals = fundamentals

    def _rebalance_dates(self) -> pd.DatetimeIndex:
        # pandas >=2.2 deprecated the 'M' alias in favor of 'ME' (month-end);
        # normalize here so config.py can keep the shorter, more readable 'M'
        # regardless of which pandas version is installed locally.
        freq = self.cfg.rebalance_freq
        freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
        freq = freq_map.get(freq, freq)
        return pd.date_range(self.cfg.start_date, self.cfg.end_date, freq=freq)

    def _get_pit_snapshot(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Point-in-time fundamentals snapshot: for each ticker, the MOST RECENT
        fundamentals row whose report_date <= as_of_date. Anything reported
        after as_of_date is invisible to the engine at this point, full stop.
        """
        available = self.fundamentals[self.fundamentals["report_date"] <= as_of_date]
        if available.empty:
            return pd.DataFrame()
        latest = available.sort_values("report_date").groupby("ticker").tail(1)
        return latest.set_index("ticker")

    def _construct_weights(self, scores: pd.Series) -> pd.Series:
        """Equal-weight long top quantile, equal-weight short bottom quantile
        (dollar-neutral: long weights sum to +0.5, short to -0.5), unless
        long_only, in which case long weights sum to 1.0."""
        ranked = scores.rank(pct=True)
        n_q = self.cfg.n_quantiles
        long_cut = (self.cfg.long_quantile - 1) / n_q
        short_cut = self.cfg.short_quantile / n_q

        long_names = ranked[ranked > long_cut].index
        weights = pd.Series(0.0, index=scores.index)

        if len(long_names) > 0:
            long_w = (1.0 if self.cfg.long_only else 0.5) / len(long_names)
            weights.loc[long_names] = long_w

        if not self.cfg.long_only:
            short_names = ranked[ranked <= short_cut].index
            if len(short_names) > 0:
                short_w = -0.5 / len(short_names)
                weights.loc[short_names] = short_w

        return weights

    def _period_return(self, weights: pd.Series, date_t: pd.Timestamp, date_t1: pd.Timestamp) -> float:
        """Portfolio return from date_t to date_t1 given weights set at date_t."""
        valid_tickers = [t for t in weights.index if t in self.prices.columns]
        if not valid_tickers:
            return 0.0
        p_t = self.prices.loc[:date_t, valid_tickers].iloc[-1]
        p_t1 = self.prices.loc[:date_t1, valid_tickers].iloc[-1]
        stock_returns = (p_t1 / p_t - 1).fillna(0.0)
        return (weights.loc[valid_tickers] * stock_returns).sum()

    def run(self) -> dict:
        rebal_dates = self._rebalance_dates()
        gross_returns, net_returns, turnovers = [], [], []
        prev_weights = pd.Series(dtype=float)
        weights_history = {}

        for i in range(len(rebal_dates) - 1):
            date_t, date_t1 = rebal_dates[i], rebal_dates[i + 1]

            snapshot = self._get_pit_snapshot(date_t)
            if snapshot.empty:
                gross_returns.append(0.0)
                net_returns.append(0.0)
                turnovers.append(0.0)
                continue

            scores = self.signal.score(snapshot)
            weights = self._construct_weights(scores)
            weights_history[date_t] = weights

            turnover = compute_turnover(weights, prev_weights)
            gross_ret = self._period_return(weights, date_t, date_t1)
            net_ret = apply_costs(gross_ret, turnover, self.cfg.cost_bps_per_turnover)

            gross_returns.append(gross_ret)
            net_returns.append(net_ret)
            turnovers.append(turnover)
            prev_weights = weights

        idx = rebal_dates[: len(gross_returns)]
        gross_series = pd.Series(gross_returns, index=idx, name="gross_return")
        net_series = pd.Series(net_returns, index=idx, name="net_return")
        turnover_series = pd.Series(turnovers, index=idx, name="turnover")

        periods_per_year = 12 if self.cfg.rebalance_freq == "M" else 252
        return {
            "gross_returns": gross_series,
            "net_returns": net_series,
            "turnovers": turnover_series,
            "weights_history": weights_history,
            "gross_summary": bt_metrics.summarize_performance(gross_series, turnover_series, periods_per_year),
            "net_summary": bt_metrics.summarize_performance(net_series, turnover_series, periods_per_year),
        }