#!/usr/bin/env python3
"""
Portfolio backtesting framework that composes multiple trading strategies and applies
allocation logic:
- Allocate buying power based on IV regime (low/medium/high)
- Split 75% index vs 25% stock
- Enforce beta*delta neutrality and theta cap
- Apply basic PnL exits via reusable ExitRules

This is intentionally simplified but designed to be extended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from time import perf_counter

import numpy as np
import pandas as pd

from .data_loader import OptionsDataLoader
from .risk_metrics import classify_iv_regime, compute_return_std_and_beta
from .trading_strategies import (
    ShortStrangle, ShortStrangleParams, IronCondor, IronCondorParams, ExitRules
)
from .options_utils_lib import process_options_data_lib
from .stocks import get_lot_size


@dataclass
class AllocationConfig:
    iv_low_pct: float = 0.25
    iv_med_pct: float = 0.35
    iv_high_pct: float = 0.50
    index_pct: float = 0.75
    stock_pct: float = 0.25
    theta_cap_pct_per_day: float = 0.002  # 0.2% per-day on total buying power
    neutrality_threshold: float = 0.06    # allowable deviation in beta*delta neutrality


@dataclass
class TradingConfig:
    symbol_index: str = 'nifty'
    symbols_stocks: Optional[List[str]] = None  # if None, auto-pick from loader
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 10000000.0

    # Default strategies per regime (can be extended)
    use_strangle: bool = True
    use_iron_condor: bool = True



    strangle_params: ShortStrangleParams = field(default_factory=ShortStrangleParams)
    iron_params: IronCondorParams = field(default_factory=IronCondorParams)
    exit_rules: ExitRules = field(default_factory=ExitRules)

    # Diagnostics
    print_positions_every: Optional[int] = None  # e.g., 10 → print every 10 steps
    debug_timings: bool = True                  # print timing logs

    # Performance knobs for options processing
    fast_greeks: bool = True                     # use vectorized greeks + grouped IV
    iv_strategy: str = 'per_expiry'              # 'per_expiry' or 'per_row'


@dataclass
class Position:
    symbol: str
    option_type: str
    strike: float
    expiry: str
    quantity: int # positive for long, negative for short
    entry_price: float
    entry_date: str
    blocked_margin: float  # Margin blocked for this position


class PortfolioBacktester:
    def __init__(self, loader: OptionsDataLoader, alloc: AllocationConfig, trading: TradingConfig):
        self.loader = loader
        self.alloc = alloc
        self.trading = trading
        self.positions: List[Position] = []
        self.daily_records: List[Dict[str, Any]] = []
        self.capital = trading.initial_capital
        self.blocked_margin = 0.0  # Total margin currently blocked

        # Strategy instances
        self.strangle = ShortStrangle(trading.strangle_params, trading.exit_rules)
        self.condor = IronCondor(trading.iron_params, trading.exit_rules)

        # Beta map filled in backtest() once symbols are finalized
        self.beta_map: Dict[str, float] = {}

        # Cache processed options by (symbol, date)
        self._processed_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def _calculate_strategy_margin(self, candidates: List[Dict[str, Any]], lot_size: int) -> float:
        """Calculate margin requirement using the appropriate strategy's margin method."""
        if len(candidates) == 0:
            return 0.0
        
        # Convert candidates to positions format for margin calculation
        strategy_positions = []
        for c in candidates:
            qty_sign = -1 if c['action'] == 'sell' else 1
            strategy_positions.append({
                'strike': float(c['strike']),
                'quantity': qty_sign * lot_size,
                'price': float(c['price']),
                'option_type': c['option_type'],
                'action': c['action']
            })
        
        # Determine which strategy to use based on candidates and use its margin calculation
        if len(candidates) == 2:
            # Short Strangle
            return self.strangle.calculate_margin_requirement(strategy_positions)
        elif len(candidates) == 4:
            # Iron Condor
            return self.condor.calculate_margin_requirement(strategy_positions)
        else:
            # Fallback to naked margin for unknown strategies
            return self._calculate_naked_margin_fallback(strategy_positions[0])
    
    def _calculate_naked_margin_fallback(self, position: Dict[str, Any]) -> float:
        """Fallback margin calculation for unknown strategies"""
        strike = position['strike']
        quantity = position['quantity'] 
        price = position['price']
        
        underlying_value = abs(strike * quantity)
        margin_base = 0.20 * underlying_value
        net_premium = quantity * price  # negative for short (credit)
        commission = 0.001 * abs(quantity * price)
        
        return max(0, margin_base - net_premium + commission)

    def _print_current_positions(self, date: str) -> None:
        if self.trading.print_positions_every is None:
            return
        # Compute current portfolio value (capital + unrealized PnL on open positions)
        unrealized_pnl = 0.0
        if self.positions:
            for p in self.positions:
                cur = self._current_option_price(p.symbol, date, p.option_type, p.strike, p.expiry)
                if cur is None:
                    continue
                if p.quantity >= 0:  # Long position
                    unrealized_pnl += p.quantity * (cur - p.entry_price)
                else:  # Short position
                    unrealized_pnl += abs(p.quantity) * (p.entry_price - cur)
        total_value = self.capital + unrealized_pnl
        if not self.positions:
            print(f"[{date}] Open positions: 0 | Portfolio Value = {total_value:,.2f}")
            return
        parts: List[str] = []
        for p in self.positions:
            lot_size = get_lot_size(p.symbol)
            lots = abs(p.quantity) // lot_size
            parts.append(
                f"{p.symbol} {p.option_type} {int(p.strike)} {p.expiry} lots={lots} qty={p.quantity} entry={p.entry_price:.2f}"
            )
        print(f"[{date}] Open positions ({len(self.positions)}): " + "; ".join(parts) + f" | Portfolio Value = {total_value:,.2f} | Blocked Margin = {self.blocked_margin:,.2f}")

    def _date_range(self, start: str, end: str) -> List[str]:
        """Generate list of trading days between start and end dates."""
        # Get all available trading days from the index symbol
        all_trading_days = self.loader.get_available_dates(self.trading.symbol_index)
        if not all_trading_days:
            return []
        
        # Filter to the requested range
        trading_days = [d for d in all_trading_days if start <= d <= end]
        return trading_days

    def _available_range(self, symbol: str) -> Optional[Tuple[str, str]]:
        dates = self.loader.get_available_dates(symbol)
        if not dates:
            return None
        return (dates[0], dates[-1])

    def _compute_allocation_by_iv(self, symbol: str, date: str) -> float:
        regime = classify_iv_regime(self.loader, symbol, date)
        if regime.bucket == 'low':
            return self.alloc.iv_low_pct
        if regime.bucket == 'medium':
            return self.alloc.iv_med_pct
        return self.alloc.iv_high_pct

    def _current_option_price(self, symbol: str, date: str, option_type: str, strike: float, expiry: str) -> Optional[float]:
        df = self.loader.load_options_data(symbol, date)
        if df is None or df.empty:
            return None
        mask = (
            (df['INSTRUMENT'].isin(['OPTIDX', 'OPTSTK'])) &
            (df['OPTION_TYP'] == option_type) &
            (df['STRIKE_PR'] == strike) &
            (df['EXPIRY_DT'] == expiry)
        )
        if mask.any():
            row = df[mask].iloc[0]
            # Prefer settlement price when available, fallback to close
            price_col = 'SETTLE_PR' if 'SETTLE_PR' in df.columns else 'CLOSE'
            val = row.get(price_col, np.nan)
            try:
                return float(val)
            except Exception:
                return None
        return None

    def _get_processed(self, symbol: str, date: str) -> pd.DataFrame:
        key = (symbol, date)
        if key in self._processed_cache:
            return self._processed_cache[key]
        # try on-disk cache first
        cached = self.loader.load_processed_options_cache(symbol, date)
        if cached is not None and not cached.empty:
            self._processed_cache[key] = cached
            return cached
        # compute and persist
        opt = self.loader.load_options_data(symbol, date)
        px = self.loader.get_stock_price(symbol, date)
        if opt is None or px is None:
            dfp = pd.DataFrame()
        else:
            try:
                dfp = process_options_data_lib(
                    opt,
                    float(px),
                    date,
                    verbose=False,  # Disable verbose output for speed
                    fast_mode=self.trading.fast_greeks,
                    iv_strategy=self.trading.iv_strategy,
                )
            except Exception:
                dfp = pd.DataFrame()
        if dfp is not None and not dfp.empty:
            self.loader.save_processed_options_cache(symbol, date, dfp)
        self._processed_cache[key] = dfp
        return dfp

    def _portfolio_greeks(self, date: str) -> Tuple[float, float]:
        """Return (beta_delta, daily_theta_sum) as crude aggregates from current positions."""
        beta_delta = 0.0
        theta_sum_daily = 0.0
        for pos in self.positions:
            dfp = self._get_processed(pos.symbol, date)
            if dfp is None or dfp.empty:
                continue
            mask = (
                (dfp['OPTION_TYP'] == pos.option_type) &
                (dfp['STRIKE_PR'] == pos.strike) &
                (dfp['EXPIRY_DT'] == pos.expiry)
            )
            if not mask.any():
                continue
            row = dfp[mask].iloc[0]
            delta = float(row.get('delta', 0.0))
            theta = float(row.get('theta', 0.0)) / 365.0  # per-day approximation
            beta = self.beta_map.get(pos.symbol, 1.0)
            beta_delta += pos.quantity * beta * delta
            theta_sum_daily += pos.quantity * theta
        return beta_delta, theta_sum_daily

    def _enforce_neutrality_and_theta_cap(self, date: str, buying_power: float):
        beta_delta, theta_sum_daily = self._portfolio_greeks(date)
        if abs(beta_delta) > self.alloc.neutrality_threshold and self.positions:
            reduce_ratio = min(0.5, abs(beta_delta))
            for i in range(len(self.positions)):
                self.positions[i].quantity = int(self.positions[i].quantity * (1 - reduce_ratio))
            self.positions = [p for p in self.positions if p.quantity != 0]
        theta_cap_value = self.alloc.theta_cap_pct_per_day * buying_power
        if abs(theta_sum_daily) > 0:
            scale = min(1.0, theta_cap_value / abs(theta_sum_daily))
            if scale < 1.0:
                for i in range(len(self.positions)):
                    new_qty = int(np.sign(self.positions[i].quantity) * max(1, int(abs(self.positions[i].quantity) * scale)))
                    self.positions[i].quantity = new_qty

    def _apply_exits(self, date: str):
        rules = self.trading.exit_rules
        still_open: List[Position] = []
        for pos in self.positions:
            cur_price = self._current_option_price(pos.symbol, date, pos.option_type, pos.strike, pos.expiry)
            if cur_price is None:
                still_open.append(pos)
                continue
            pnl = (pos.entry_price - cur_price) if pos.quantity < 0 else (cur_price - pos.entry_price)
            ref = abs(pos.entry_price)
            reached_tp = ref > 0 and (pnl / ref) >= rules.take_profit_pct
            reached_sl = ref > 0 and (pnl / ref) <= -rules.stop_loss_pct
            try:
                d = datetime.strptime(date, '%Y-%m-%d')
                e = datetime.strptime(pos.expiry, '%Y-%m-%d')
                dte_days = (e - d).days
            except Exception:
                dte_days = 9999
            time_exit = dte_days <= rules.max_days_to_expiry
            if reached_tp or reached_sl or time_exit:
                # Close position: pay/receive current price and realize PnL
                capital_change = pos.quantity * cur_price  # reverse of opening trade
                self.capital += capital_change
                # Release blocked margin
                self.blocked_margin -= pos.blocked_margin
                continue
            else:
                still_open.append(pos)
        self.positions = still_open

    def _open_positions_for_symbol(self, symbol: str, date: str, capital_alloc: float):
        options_df = self.loader.load_options_data(symbol, date)
        px = self.loader.get_stock_price(symbol, date)
        if options_df is None or px is None:
            return
        candidates: List[Dict[str, Any]] = []
        if self.trading.use_strangle:
            candidates.extend(self.strangle.select_positions(options_df, float(px), date))
        if self.trading.use_iron_condor:
            candidates.extend(self.condor.select_positions(options_df, float(px), date))
        if not candidates:
            return
        per_trade_cap = capital_alloc / max(1, len(candidates))
        lot_size = get_lot_size(symbol)
        
        # Debug: Print capital allocation details
        # if len(candidates) > 0:
        #     print(f"DEBUG {symbol}: capital_alloc={capital_alloc:,.0f}, candidates={len(candidates)}, per_trade_cap={per_trade_cap:,.0f}")
        
        # Calculate margin for the entire strategy (all candidates together)
        if not candidates:
            return
            
        # Convert candidates to the format expected by margin calculation
        strategy_positions = []
        for i, c in enumerate(candidates):
            price = float(c['price'])
            strike = float(c['strike'])
            qty_sign = -1 if c['action'] == 'sell' else 1
            test_quantity = qty_sign * lot_size
            
            strategy_positions.append({
                'strike': strike,
                'quantity': test_quantity,
                'price': price,
                'option_type': c['option_type'],
                'action': c['action']
            })
        
        # Calculate margin requirement for the entire strategy using strategy-specific method
        strategy_margin = self._calculate_strategy_margin(candidates, lot_size)
        
        # Check if we have enough available margin
        available_margin = capital_alloc - self.blocked_margin
        if strategy_margin > available_margin:
            # print(f"DEBUG {symbol}: Skipping strategy - need {strategy_margin:.0f}, available {available_margin:.0f}")
            return  # Skip entire strategy - not enough margin
        
        # Calculate how many lots we can afford
        max_lots_by_margin = int(available_margin / (strategy_margin / lot_size))
        max_lots_by_capital = int(per_trade_cap / max(1.0, sum(abs(c['price']) for c in candidates) * lot_size))
        
        # Use the more restrictive limit  
        affordable_lots = max(1, min(max_lots_by_margin, max_lots_by_capital))
        
        # Scale the strategy margin by actual lots
        actual_strategy_margin = strategy_margin * (affordable_lots / 1.0)
        
        # print(f"DEBUG {symbol}: Strategy margin={strategy_margin:.0f}, lots={affordable_lots}, total_margin={actual_strategy_margin:.0f}")
        
        # Open all positions in the strategy
        total_capital_change = 0.0
        for c in candidates:
            price = float(c['price'])
            strike = float(c['strike'])
            qty_sign = -1 if c['action'] == 'sell' else 1
            quantity = qty_sign * affordable_lots * lot_size
            
            # Update capital 
            capital_change = -quantity * price  # negative quantity means short (receive premium)
            total_capital_change += capital_change
            
            self.positions.append(
                Position(
                    symbol=symbol,
                    option_type=c['option_type'],
                    strike=strike,
                    expiry=str(c['expiry']),
                    quantity=quantity,
                    entry_price=price,
                    entry_date=date,
                    blocked_margin=actual_strategy_margin / len(candidates),  # Split margin across positions
                )
            )
        
        # Update portfolio capital and blocked margin
        self.capital += total_capital_change
        self.blocked_margin += actual_strategy_margin

    def backtest(self) -> Dict[str, Any]:
        idx_range = self._available_range(self.trading.symbol_index)
        if idx_range is None:
            raise ValueError('No data for index symbol.')
        if self.trading.symbols_stocks is None:
            syms = self.loader.get_available_symbols()
            symbols_stocks = [s for s in syms if s != self.trading.symbol_index]
            symbols_stocks = symbols_stocks[:5]
        else:
            symbols_stocks = self.trading.symbols_stocks
        self.beta_map = {self.trading.symbol_index: 1.0}
        for s in symbols_stocks:
            try:
                summ = compute_return_std_and_beta(self.loader, s)
                beta = 1.0 if summ.beta_vs_nifty is None else float(summ.beta_vs_nifty)
            except Exception:
                beta = 1.0
            self.beta_map[s] = beta
        start = self.trading.start_date or idx_range[0]
        end = self.trading.end_date or idx_range[1]
        dates = self._date_range(start, end)

        for i, d in enumerate(dates):
            iv_alloc_pct = self._compute_allocation_by_iv(self.trading.symbol_index, d)
            buying_power = self.capital * iv_alloc_pct
            index_cap = buying_power * self.alloc.index_pct
            stock_cap = buying_power * self.alloc.stock_pct

            self._apply_exits(d)

            # Open new positions every day
            self._open_positions_for_symbol(self.trading.symbol_index, d, index_cap)
            if symbols_stocks:
                stock_symbol = symbols_stocks[i % len(symbols_stocks)]
                self._open_positions_for_symbol(stock_symbol, d, stock_cap)

            self._enforce_neutrality_and_theta_cap(d, buying_power)

            # Periodically print current positions for visibility
            if (
                self.trading.print_positions_every is not None
                and self.trading.print_positions_every > 0
                and i > 0
                and i % self.trading.print_positions_every == 0
            ):
                self._print_current_positions(d)

            # Calculate unrealized PnL of open positions
            unrealized_pnl = 0.0
            for pos in self.positions:
                cur = self._current_option_price(pos.symbol, d, pos.option_type, pos.strike, pos.expiry)
                if cur is None:
                    continue
                # Unrealized PnL = current value - cost basis
                if pos.quantity >= 0:  # Long position
                    unrealized_pnl += pos.quantity * (cur - pos.entry_price)
                else:  # Short position  
                    unrealized_pnl += abs(pos.quantity) * (pos.entry_price - cur)

            total_value = self.capital + unrealized_pnl
            self.daily_records.append({'date': d, 'total_value': total_value})

        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        if df.empty:
            return {}
        final_value = float(df['total_value'].iloc[-1])
        total_return = (final_value - self.trading.initial_capital) / self.trading.initial_capital
        daily_ret = df['total_value'].pct_change().dropna()
        ann_vol = float(daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else 0.0
        ann_ret = float(((final_value / self.trading.initial_capital) ** (365 / max(1, (df.index[-1] - df.index[0]).days))) - 1)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': ann_ret,
            'volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'equity_curve': df
        }
