#!/usr/bin/env python3
"""
Reusable options trading strategies that generate desired positions based on
Greeks and tunable parameters. These are building blocks for portfolio allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from .data_loader import OptionsDataLoader
from .options_utils_lib import process_options_data_lib


@dataclass
class ExitRules:
    take_profit_pct: float = 0.25    # 25% profit of received premium
    stop_loss_pct: float = 0.5       # 50% loss of received premium
    max_days_to_expiry: int = 21     # exit if <= 21 days to expiry


@dataclass
class ShortStrangleParams:
    target_delta: float = 0.16
    days_to_expiry_target: int = 30


@dataclass
class IronCondorParams:
    short_delta: float = 0.16
    wing_width_pct: float = 0.05     # distance of long wing as % of underlying
    days_to_expiry_target: int = 30


class ShortStrangle:
    def __init__(self, params: ShortStrangleParams, exit_rules: ExitRules):
        self.params = params
        self.exit_rules = exit_rules

    def select_positions(self, options_df: pd.DataFrame, stock_price: float, date: str) -> List[Dict[str, Any]]:
        processed = process_options_data_lib(options_df, stock_price, date)
        if processed is None or processed.empty:
            return []

        # Select a single expiry near target DTE
        processed = processed.sort_values('DAYS_TO_EXPIRY')
        processed['dte_days'] = (processed['DAYS_TO_EXPIRY'] * 365.25).round()
        expiry = None
        for _, grp in processed.groupby('EXPIRY_DT'):
            dte = grp['dte_days'].iloc[0]
            if abs(dte - self.params.days_to_expiry_target) <= 7:
                expiry = grp['EXPIRY_DT'].iloc[0]
                break
        if expiry is None:
            return []

        expiry_df = processed[processed['EXPIRY_DT'] == expiry].copy()
        # Find CE/PE closest to target absolute delta
        ce = expiry_df[expiry_df['OPTION_TYP'] == 'CE'].copy()
        pe = expiry_df[expiry_df['OPTION_TYP'] == 'PE'].copy()
        if ce.empty or pe.empty or 'delta' not in ce.columns:
            return []

        ce['delta_abs'] = ce['delta'].abs()
        pe['delta_abs'] = pe['delta'].abs()
        ce_sel = ce.iloc[(ce['delta_abs'] - self.params.target_delta).abs().argsort()[:1]]
        pe_sel = pe.iloc[(pe['delta_abs'] - self.params.target_delta).abs().argsort()[:1]]

        positions: List[Dict[str, Any]] = []
        for _, row in pd.concat([ce_sel, pe_sel]).iterrows():
            positions.append({
                'type': 'option',
                'action': 'sell',
                'option_type': row['OPTION_TYP'],
                'strike': float(row['STRIKE_PR']),
                'expiry': row['EXPIRY_DT'],
                'price': float(row['SETTLE_PR']),
                'delta': float(row['delta']),
                'theta': float(row.get('theta', 0.0)),
                'quantity': 1
            })
        return positions


class IronCondor:
    def __init__(self, params: IronCondorParams, exit_rules: ExitRules):
        self.params = params
        self.exit_rules = exit_rules

    def select_positions(self, options_df: pd.DataFrame, stock_price: float, date: str) -> List[Dict[str, Any]]:
        processed = process_options_data_lib(options_df, stock_price, date)
        if processed is None or processed.empty:
            return []

        processed = processed.sort_values('DAYS_TO_EXPIRY')
        processed['dte_days'] = (processed['DAYS_TO_EXPIRY'] * 365.25).round()
        expiry = None
        for _, grp in processed.groupby('EXPIRY_DT'):
            dte = grp['dte_days'].iloc[0]
            if abs(dte - self.params.days_to_expiry_target) <= 7:
                expiry = grp['EXPIRY_DT'].iloc[0]
                break
        if expiry is None:
            return []

        expiry_df = processed[processed['EXPIRY_DT'] == expiry].copy()
        ce = expiry_df[expiry_df['OPTION_TYP'] == 'CE'].copy()
        pe = expiry_df[expiry_df['OPTION_TYP'] == 'PE'].copy()
        if ce.empty or pe.empty or 'delta' not in ce.columns:
            return []

        # Select short strikes at ~target_delta
        ce['delta_abs'] = ce['delta'].abs()
        pe['delta_abs'] = pe['delta'].abs()
        short_call = ce.iloc[(ce['delta_abs'] - self.params.short_delta).abs().argsort()[:1]].iloc[0]
        short_put = pe.iloc[(pe['delta_abs'] - self.params.short_delta).abs().argsort()[:1]].iloc[0]

        # Long wings at fixed width (percent of underlying)
        wing = self.params.wing_width_pct * stock_price
        long_call_strike = float(short_call['STRIKE_PR'] + wing)
        long_put_strike = float(max(0.0, short_put['STRIKE_PR'] - wing))

        # Pick nearest available strikes for wings
        def nearest_strike(df: pd.DataFrame, strike: float, opt_type: str) -> Optional[pd.Series]:
            sub = df[df['OPTION_TYP'] == opt_type].copy()
            if sub.empty:
                return None
            sub['dist'] = (sub['STRIKE_PR'] - strike).abs()
            return sub.sort_values('dist').iloc[0]

        long_call = nearest_strike(expiry_df, long_call_strike, 'CE')
        long_put = nearest_strike(expiry_df, long_put_strike, 'PE')
        if long_call is None or long_put is None:
            return []

        positions = []
        # Short legs
        positions.append({
            'type': 'option', 'action': 'sell', 'option_type': 'CE',
            'strike': float(short_call['STRIKE_PR']), 'expiry': short_call['EXPIRY_DT'],
            'price': float(short_call['SETTLE_PR']), 'delta': float(short_call['delta']),
            'theta': float(short_call.get('theta', 0.0)), 'quantity': 1
        })
        positions.append({
            'type': 'option', 'action': 'sell', 'option_type': 'PE',
            'strike': float(short_put['STRIKE_PR']), 'expiry': short_put['EXPIRY_DT'],
            'price': float(short_put['SETTLE_PR']), 'delta': float(short_put['delta']),
            'theta': float(short_put.get('theta', 0.0)), 'quantity': 1
        })
        # Long wings
        positions.append({
            'type': 'option', 'action': 'buy', 'option_type': 'CE',
            'strike': float(long_call['STRIKE_PR']), 'expiry': long_call['EXPIRY_DT'],
            'price': float(long_call['SETTLE_PR']), 'delta': float(long_call.get('delta', 0.0)),
            'theta': float(long_call.get('theta', 0.0)), 'quantity': 1
        })
        positions.append({
            'type': 'option', 'action': 'buy', 'option_type': 'PE',
            'strike': float(long_put['STRIKE_PR']), 'expiry': long_put['EXPIRY_DT'],
            'price': float(long_put['SETTLE_PR']), 'delta': float(long_put.get('delta', 0.0)),
            'theta': float(long_put.get('theta', 0.0)), 'quantity': 1
        })
        return positions
