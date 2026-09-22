"""
REAL data loader — run this locally (not in a sandboxed environment without
internet access). Pulls prices and quarterly fundamentals via yfinance.

IMPORTANT point-in-time caveat (read this before trusting results):
yfinance's quarterly fundamentals do NOT reliably expose the actual SEC filing
date, only the fiscal period end. We approximate the report_date as
`fiscal_period_end + config.fundamentals_lag_days`. This is a reasonable
conservative approximation (real filings land 30-90 days out) but is NOT
true point-in-time data. If you want to remove this approximation, swap this
loader for a point-in-time fundamentals source (e.g. SEC EDGAR full-text
search API, or a paid point-in-time vendor) — the rest of the pipeline
(signals/, backtest/) doesn't care where the DataFrame came from as long as
it has the same schema as data/synthetic.py's generate_fundamentals().

Install: pip install yfinance pandas numpy
"""
import time
import io
import requests
import pandas as pd
import numpy as np


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list:
    """Scrape current S&P 500 tickers from Wikipedia. Survivorship-biased
    (uses today's constituents for the whole backtest window) — note this
    limitation in your README; fixing it requires point-in-time index
    membership data, which is a good 'future work' item.

    Wikipedia (like many sites) returns HTTP 403 for requests with no
    browser-like User-Agent header, which is what pandas/urllib send by
    default — hence the explicit requests.get() with headers below instead
    of calling pd.read_html(url) directly.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    response = requests.get(SP500_WIKI_URL, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def load_prices(tickers: list, start_date: str, end_date: str, batch_size: int = 50) -> pd.DataFrame:
    import yfinance as yf

    all_frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = yf.download(batch, start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        all_frames.append(data)
        time.sleep(1)  # be polite to the API
    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices


def load_fundamentals(tickers: list, fundamentals_lag_days: int, stability_window: int = 4) -> pd.DataFrame:
    """
    Pulls quarterly fundamentals per ticker and derives the factor columns
    used by signals/quality_value.py. Slow (one API call per ticker) —
    cache the result to disk (see scripts/run_backtest.py) rather than
    re-pulling on every run.

    earnings_stability is derived, not pulled directly: for each quarter it's
    the NEGATIVE standard deviation of ROE over the trailing `stability_window`
    quarters (including the current one) — so a less volatile earnings history
    scores higher (more "stable" = better quality). Crucially this only looks
    backward from each quarter's own period_end, never forward, so it can't
    introduce look-ahead bias on top of what the point-in-time report_date
    lag already handles. Needs at least 2 quarters of history to compute;
    earlier quarters (or tickers with sparse data) get NaN, which the signal
    layer already handles gracefully (see signals/quality_value.py).
    """
    import yfinance as yf

    rows = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            q_financials = tk.quarterly_financials
            q_balance = tk.quarterly_balance_sheet
            info = tk.info

            if q_financials.empty or q_balance.empty:
                continue

            ticker_rows = []
            for period_end in q_financials.columns:
                if period_end not in q_balance.columns:
                    continue
                try:
                    net_income = q_financials.loc["Net Income", period_end]
                    revenue = q_financials.loc["Total Revenue", period_end]
                    gross_profit = q_financials.loc.get("Gross Profit", np.nan) if hasattr(q_financials.loc, "get") else q_financials.loc["Gross Profit", period_end]
                    total_equity = q_balance.loc["Stockholders Equity", period_end]
                    total_debt = q_balance.loc["Total Debt", period_end] if "Total Debt" in q_balance.index else np.nan
                    shares_out = info.get("sharesOutstanding", np.nan)
                    price_now = info.get("currentPrice", np.nan)

                    market_cap = shares_out * price_now if shares_out and price_now else np.nan
                    book_value = total_equity

                    earnings_yield = net_income / market_cap if market_cap else np.nan
                    book_to_market = book_value / market_cap if market_cap else np.nan
                    roe = net_income / total_equity if total_equity else np.nan
                    gross_margin = gross_profit / revenue if revenue else np.nan
                    debt_to_equity = total_debt / total_equity if total_equity else np.nan

                    report_date = pd.Timestamp(period_end) + pd.Timedelta(days=fundamentals_lag_days)

                    ticker_rows.append(
                        {
                            "ticker": ticker,
                            "fiscal_period_end": pd.Timestamp(period_end),
                            "report_date": report_date,
                            "earnings_yield": earnings_yield,
                            "book_to_market": book_to_market,
                            "roe": roe,
                            "gross_margin": gross_margin,
                            "debt_to_equity": debt_to_equity,
                        }
                    )
                except (KeyError, TypeError):
                    continue

            if not ticker_rows:
                continue

            # Compute trailing ROE stability per quarter, ascending by period_end
            # so the rolling window only ever looks backward from each quarter.
            tdf = pd.DataFrame(ticker_rows).sort_values("fiscal_period_end").reset_index(drop=True)
            rolling_std = tdf["roe"].rolling(window=stability_window, min_periods=2).std(ddof=0)
            tdf["earnings_stability"] = -rolling_std

            rows.extend(tdf.to_dict("records"))

        except Exception as e:
            print(f"  [warn] skipping {ticker}: {e}")
            continue

    return pd.DataFrame(rows)