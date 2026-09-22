"""
Performance and statistical significance metrics for a return series.
"""
import numpy as np
import pandas as pd
from scipy import stats


def annualize_return(period_returns: pd.Series, periods_per_year: int = 12) -> float:
    compounded = (1 + period_returns).prod()
    n_periods = len(period_returns)
    if n_periods == 0:
        return np.nan
    return compounded ** (periods_per_year / n_periods) - 1


def annualize_vol(period_returns: pd.Series, periods_per_year: int = 12) -> float:
    return period_returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(period_returns: pd.Series, periods_per_year: int = 12, rf: float = 0.0) -> float:
    excess = period_returns - rf / periods_per_year
    if excess.std(ddof=1) == 0:
        return np.nan
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(period_returns: pd.Series) -> float:
    cum = (1 + period_returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    return drawdown.min()


def t_stat_mean_return(period_returns: pd.Series) -> tuple:
    """t-stat and p-value testing whether mean period return is significantly != 0.
    This is the basic statistical significance check the JD's 'probability and
    statistics' qualification is pointing at — a Sharpe ratio alone doesn't tell
    you whether the signal is distinguishable from noise."""
    clean = period_returns.dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    t_stat, p_value = stats.ttest_1samp(clean, 0.0)
    return t_stat, p_value


def summarize_performance(period_returns: pd.Series, turnovers: pd.Series, periods_per_year: int = 12) -> dict:
    t_stat, p_value = t_stat_mean_return(period_returns)
    return {
        "annualized_return": annualize_return(period_returns, periods_per_year),
        "annualized_vol": annualize_vol(period_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(period_returns, periods_per_year),
        "max_drawdown": max_drawdown(period_returns),
        "avg_turnover": turnovers.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "n_periods": len(period_returns.dropna()),
    }