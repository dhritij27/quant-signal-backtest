"""
Main entry point.

  python scripts/run_backtest.py --mode synthetic     # runs anywhere, validates logic
  python scripts/run_backtest.py --mode real           # pulls S&P 500 data locally (needs internet)

Real-data run caches prices/fundamentals to data/cache/ so you're not
re-hitting yfinance on every iteration while you tune the signal.
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG
from data.synthetic import generate_synthetic_dataset
from signals.quality_value import ValueQualityComposite
from backtest.engine import BacktestEngine


def run_synthetic(cfg):
    print(f"Generating synthetic dataset ({cfg.universe_name}-sized universe)...")
    universe, prices, fundamentals = generate_synthetic_dataset(
        n_stocks=500, start_date=cfg.start_date, end_date=cfg.end_date
    )
    return prices, fundamentals


def run_real(cfg):
    from data.loader import get_sp500_tickers, load_prices, load_fundamentals

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    prices_path = os.path.join(cache_dir, "prices.parquet")
    fund_path = os.path.join(cache_dir, "fundamentals.parquet")

    if os.path.exists(prices_path) and os.path.exists(fund_path):
        print("Loading cached data...")
        prices = pd.read_parquet(prices_path)
        fundamentals = pd.read_parquet(fund_path)
        return prices, fundamentals

    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Got {len(tickers)} tickers. Pulling prices (this takes a few minutes)...")
    prices = load_prices(tickers, cfg.start_date, cfg.end_date)
    prices.to_parquet(prices_path)

    print("Pulling fundamentals (slow — one call per ticker)...")
    fundamentals = load_fundamentals(tickers, cfg.fundamentals_lag_days)
    fundamentals.to_parquet(fund_path)

    return prices, fundamentals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG

    if args.mode == "synthetic":
        prices, fundamentals = run_synthetic(cfg)
    else:
        prices, fundamentals = run_real(cfg)

    signal = ValueQualityComposite(cfg)
    engine = BacktestEngine(cfg, signal, prices, fundamentals)
    results = engine.run()

    print("\n=== GROSS (before costs) ===")
    for k, v in results["gross_summary"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== NET (after transaction costs) ===")
    for k, v in results["net_summary"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    results["net_returns"].to_csv(os.path.join(out_dir, "net_returns.csv"))
    results["gross_returns"].to_csv(os.path.join(out_dir, "gross_returns.csv"))
    results["turnovers"].to_csv(os.path.join(out_dir, "turnovers.csv"))
    print(f"\nSaved return series to {out_dir}/")


if __name__ == "__main__":
    main()