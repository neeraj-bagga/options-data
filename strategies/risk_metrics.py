#!/usr/bin/env python3
"""
Risk metrics utilities:
- Daily return std dev for stock/index
- Beta of stock vs NIFTY
- IV regime detection using historical options IV distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .data_loader import OptionsDataLoader
from .options_utils_lib import process_options_data_lib


@dataclass
class SymbolRiskSummary:
    symbol: str
    start_date: Optional[str]
    end_date: Optional[str]
    daily_return_std: Optional[float]
    beta_vs_nifty: Optional[float]


def _compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    price_col = 'Close' if 'Close' in df.columns else 'Adj Close' if 'Adj Close' in df.columns else None
    if price_col is None:
        return pd.Series(dtype=float)
    s = df[price_col].astype(float)
    return s.pct_change().dropna()


def compute_return_std_and_beta(
    loader: OptionsDataLoader,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    nifty_symbol: str = 'nifty'
) -> SymbolRiskSummary:
    """
    Compute daily return std dev and beta of `symbol` vs `nifty` using local stock_data.
    Uses overlapping dates between the two series.
    """
    # Use available stock dates lists to avoid non-trading days
    dates_symbol = loader.get_available_stock_dates(symbol)
    dates_nifty = loader.get_available_stock_dates(nifty_symbol)
    if not dates_symbol or not dates_nifty:
        return SymbolRiskSummary(symbol, None, None, None, None)

    start = start_date or max(dates_symbol[0], dates_nifty[0])
    end = end_date or min(dates_symbol[-1], dates_nifty[-1])

    # Restrict to overlapping set
    range_filter = lambda d: start <= d <= end
    dates_symbol_in = [d for d in dates_symbol if range_filter(d)]
    dates_nifty_in = [d for d in dates_nifty if range_filter(d)]
    date_set = sorted(set(dates_symbol_in).intersection(dates_nifty_in))
    if not date_set:
        return SymbolRiskSummary(symbol, start, end, None, None)

    # Build series
    def _load_series(sym: str, date_list: List[str]) -> pd.Series:
        prices = []
        dts = []
        for date_str in date_list:
            sdf = loader.load_stock_data(sym, date_str)
            if sdf is not None and not sdf.empty:
                price_col = 'Close' if 'Close' in sdf.columns else None
                if price_col:
                    prices.append(float(sdf[price_col].iloc[0]))
                    dts.append(pd.to_datetime(date_str))
        s = pd.Series(prices, index=pd.to_datetime(dts))
        return s

    s_symbol = _load_series(symbol, date_set)
    s_nifty = _load_series(nifty_symbol, date_set)

    if s_symbol.empty or s_nifty.empty:
        return SymbolRiskSummary(symbol, start, end, None, None)

    # Align
    df = pd.concat({'sym': s_symbol, 'nifty': s_nifty}, axis=1).dropna()
    if df.empty:
        return SymbolRiskSummary(symbol, start, end, None, None)

    r_sym = df['sym'].pct_change().dropna()
    r_nifty = df['nifty'].pct_change().dropna()
    aligned = pd.concat([r_sym, r_nifty], axis=1).dropna()
    if aligned.empty:
        return SymbolRiskSummary(symbol, start, end, None, None)

    daily_std = float(aligned['sym'].std())
    var_nifty = float(aligned['nifty'].var())
    cov = float(np.cov(aligned['sym'], aligned['nifty'])[0, 1])
    beta = cov / var_nifty if var_nifty > 0 else None

    return SymbolRiskSummary(symbol, start, end, daily_std, beta)


@dataclass
class IVRegime:
    percentile: float
    bucket: str  # 'low' | 'medium' | 'high'


def classify_iv_regime(
    loader: OptionsDataLoader,
    symbol: str,
    current_date: str,
    lookback_days: int = 180,
    low_thresh: float = 33.33,
    high_thresh: float = 66.67,
) -> IVRegime:
    """
    Compute IV percentile using avg implied_volatility across all options on each day
    (within lookback window), then classify into buckets.
    """
    # Build a rolling window of dates
    all_dates = loader.get_available_dates(symbol)
    if not all_dates:
        return IVRegime(percentile=50.0, bucket='medium')
    # Ensure current_date is within
    if current_date not in all_dates:
        # fallback to nearest prior date
        prior = [d for d in all_dates if d <= current_date]
        if not prior:
            return IVRegime(percentile=50.0, bucket='medium')
        current_date = prior[-1]

    # Collect last `lookback_days` dates including current_date
    idx = all_dates.index(current_date)
    start_idx = max(0, idx - lookback_days + 1)
    window_dates = all_dates[start_idx: idx + 1]

    # Compute average IV per date
    iv_avgs: List[Tuple[str, float]] = []
    for d in window_dates:
        opt_df = loader.load_options_data(symbol, d)
        px = loader.get_stock_price(symbol, d)
        if opt_df is None or px is None:
            continue
        try:
            processed = process_options_data_lib(opt_df, float(px), d)
            if processed is not None and not processed.empty and 'implied_volatility' in processed.columns:
                iv_avgs.append((d, float(processed['implied_volatility'].dropna().mean())))
        except Exception:
            continue

    if not iv_avgs:
        return IVRegime(percentile=50.0, bucket='medium')

    df_iv = pd.DataFrame(iv_avgs, columns=['date', 'avg_iv']).set_index('date')
    cur_iv = float(df_iv.loc[current_date, 'avg_iv']) if current_date in df_iv.index else float(df_iv['avg_iv'].iloc[-1])

    # Percentile of current IV within historical window
    pct = float((df_iv['avg_iv'] <= cur_iv).sum() / len(df_iv) * 100.0)
    if pct < low_thresh:
        bucket = 'low'
    elif pct < high_thresh:
        bucket = 'medium'
    else:
        bucket = 'high'

    return IVRegime(percentile=pct, bucket=bucket)


def summarize_symbols(loader: OptionsDataLoader, symbols: List[str]) -> Dict[str, SymbolRiskSummary]:
    summary: Dict[str, SymbolRiskSummary] = {}
    for sym in symbols:
        summary[sym] = compute_return_std_and_beta(loader, sym)
    return summary
