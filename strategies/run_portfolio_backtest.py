#!/usr/bin/env python3
"""
Run portfolio backtest using local market_data.
- Prints high-level risk metrics (std, beta vs NIFTY) per symbol
- Runs portfolio allocator with IV-based sizing, neutrality, and theta cap
"""

import os
from datetime import datetime

from .data_loader import OptionsDataLoader
from .risk_metrics import summarize_symbols
from .portfolio_framework import (
    PortfolioBacktester, AllocationConfig, TradingConfig
)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'market_data')
    loader = OptionsDataLoader(data_dir=data_dir)

    symbols = loader.get_available_symbols()
    print(f"Found {len(symbols)} symbols with options data")
    if 'nifty' not in symbols:
        print("NIFTY data is required for beta and index trading. Exiting.")
        return

    # Print high-level risk metrics for first few symbols
    to_summarize = [s for s in symbols if s != 'nifty'][:5]
    summary = summarize_symbols(loader, to_summarize)
    print("\n=== High-Level Risk Metrics ===")
    for sym, s in summary.items():
        print(f"{sym}: std={s.daily_return_std}, beta_vs_nifty={s.beta_vs_nifty}, range=({s.start_date} to {s.end_date})")

    # Configure and run backtest
    alloc = AllocationConfig(
        iv_low_pct=0.25, iv_med_pct=0.35, iv_high_pct=0.50,
        index_pct=0.75, stock_pct=0.25,
        theta_cap_pct_per_day=0.002,
        neutrality_threshold=0.05,
    )
    trading = TradingConfig(
        symbol_index='nifty',
        symbols_stocks=to_summarize,
        start_date=None,  # Use full available data range
        end_date=None,    # Use full available data range
        initial_capital=10_000_000.0,
        use_strangle=True,
        use_iron_condor=True,
        print_positions_every=20,
        debug_timings=False,
        fast_greeks=True,
        iv_strategy='per_expiry',
    )

    bt = PortfolioBacktester(loader, alloc, trading)
    results = bt.backtest()

    if not results:
        print("Backtest produced no results.")
        return

    print("\n=== Portfolio Backtest Results ===")
    print(f"Final Value: {results['final_value']:.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")


if __name__ == '__main__':
    main()
