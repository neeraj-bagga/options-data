#!/usr/bin/env python3
"""
Options Strategy Backtesting Framework
Provides a framework for implementing and backtesting options strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from options_utils_lib import OptionsCalculatorLib as OptionsCalculator, process_options_data_lib as process_options_data
from data_loader import OptionsDataLoader

class OptionsStrategy(ABC):
    """
    Abstract base class for options strategies
    """
    
    def __init__(self, name: str, data_loader: OptionsDataLoader, 
                 risk_free_rate: float = 0.05):
        """
        Initialize the strategy
        
        Args:
            name: Strategy name
            data_loader: Data loader instance
            risk_free_rate: Risk-free rate for calculations
        """
        self.name = name
        self.data_loader = data_loader
        self.calculator = OptionsCalculator(risk_free_rate)
        self.risk_free_rate = risk_free_rate
        self.positions = []
        self.trades = []
        self.portfolio_value = []
        
    @abstractmethod
    def generate_signals(self, symbol: str, date: str, options_data: pd.DataFrame, 
                        stock_price: float) -> Dict[str, Any]:
        """
        Generate trading signals for the strategy
        
        Args:
            symbol: Symbol name
            date: Current date
            options_data: Options data for the date
            stock_price: Current stock price
            
        Returns:
            Dictionary with trading signals
        """
        pass
    
    @abstractmethod
    def execute_trades(self, signals: Dict[str, Any], symbol: str, date: str, 
                      options_data: pd.DataFrame, stock_price: float) -> List[Dict]:
        """
        Execute trades based on signals
        
        Args:
            signals: Trading signals from generate_signals
            symbol: Symbol name
            date: Current date
            options_data: Options data for the date
            stock_price: Current stock price
            
        Returns:
            List of executed trades
        """
        pass
    
    def calculate_portfolio_value(self, symbol: str, date: str, 
                                positions: List[Dict]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            symbol: Symbol name
            date: Current date
            positions: Current positions
            
        Returns:
            Portfolio value
        """
        if not positions:
            return 0.0
        
        total_value = 0.0
        
        for position in positions:
            if position['type'] == 'option':
                # Get current option price
                options_data = self.data_loader.load_options_data(symbol, date)
                if options_data is not None:
                    # Find the specific option
                    option_mask = (
                        (options_data['STRIKE_PR'] == position['strike']) &
                        (options_data['OPTION_TYP'] == position['option_type']) &
                        (options_data['EXPIRY_DT'] == position['expiry'])
                    )
                    
                    if option_mask.any():
                        current_price = options_data[option_mask]['SETTLE_PR'].iloc[0]  # Use SETTLE_PR instead of CLOSE
                        
                        # For short positions, profit = entry_price - current_price
                        # For long positions, profit = current_price - entry_price
                        if position['quantity'] > 0:  # Long position
                            position_value = position['quantity'] * (current_price - position['avg_price'])
                        else:  # Short position
                            position_value = abs(position['quantity']) * (position['avg_price'] - current_price)
                        
                        total_value += position_value
                        
            elif position['type'] == 'stock':
                # Stock position value
                stock_price = self.data_loader.get_stock_price(symbol, date)
                if stock_price:
                    position_value = position['quantity'] * stock_price
                    total_value += position_value
        
        return total_value
    
    def backtest(self, symbol: str, start_date: str, end_date: str, 
                initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run backtest for the strategy
        
        Args:
            symbol: Symbol to backtest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        print(f"🚀 Starting backtest for {self.name} on {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💰 Initial capital: ${initial_capital:,.2f}")
        
        # Initialize tracking variables
        self.positions = []
        self.trades = []
        self.portfolio_value = []
        
        current_capital = initial_capital
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Track daily portfolio values
        daily_values = []
        
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            try:
                # Load data for current date
                options_data = self.data_loader.load_options_data(symbol, date_str)
                stock_price = self.data_loader.get_stock_price(symbol, date_str)
                
                if options_data is not None and stock_price is not None:
                    # Generate signals
                    signals = self.generate_signals(symbol, date_str, options_data, stock_price)
                    
                    # Execute trades
                    new_trades = self.execute_trades(signals, symbol, date_str, options_data, stock_price)
                    
                    # Update positions and trades
                    self.trades.extend(new_trades)
                    self._update_positions(new_trades)
                    
                    # Calculate portfolio value
                    portfolio_value = self.calculate_portfolio_value(symbol, date_str, self.positions)
                    total_value = current_capital + portfolio_value
                    
                    daily_values.append({
                        'date': date_str,
                        'portfolio_value': portfolio_value,
                        'total_value': total_value,
                        'capital': current_capital,
                        'positions_count': len(self.positions)
                    })
                    
                    # Print progress every 30 days
                    if len(daily_values) % 30 == 0:
                        print(f"📊 {date_str}: Portfolio Value = ${total_value:,.2f}, Positions = {len(self.positions)}")
                
            except Exception as e:
                print(f"⚠️  Error processing {date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(daily_values, initial_capital)
        
        print(f"✅ Backtest completed!")
        print(f"📈 Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"💰 Total Return: {results['total_return']:.2%}")
        print(f"📊 Total Trades: {len(self.trades)}")
        
        return results
    
    def _update_positions(self, new_trades: List[Dict]):
        """Update positions based on new trades"""
        for trade in new_trades:
            if trade['action'] == 'buy':
                # Add new long position or increase existing
                self._add_position(trade)
            elif trade['action'] == 'sell':
                # For options, selling creates a short position
                if trade['type'] == 'option':
                    self._add_short_position(trade)
                else:
                    # For stocks, selling reduces long position
                    self._reduce_position(trade)
    
    def _add_position(self, trade: Dict):
        """Add or increase a position"""
        # Check if position already exists
        existing_position = None
        for pos in self.positions:
            if (pos['type'] == trade['type'] and 
                pos.get('strike') == trade.get('strike') and
                pos.get('option_type') == trade.get('option_type') and
                pos.get('expiry') == trade.get('expiry')):
                existing_position = pos
                break
        
        if existing_position:
            # Update existing position
            existing_position['quantity'] += trade['quantity']
            existing_position['avg_price'] = (
                (existing_position['avg_price'] * (existing_position['quantity'] - trade['quantity']) +
                 trade['price'] * trade['quantity']) / existing_position['quantity']
            )
        else:
            # Add new position
            self.positions.append({
                'type': trade['type'],
                'quantity': trade['quantity'],
                'avg_price': trade['price'],
                'strike': trade.get('strike'),
                'option_type': trade.get('option_type'),
                'expiry': trade.get('expiry')
            })
    
    def _reduce_position(self, trade: Dict):
        """Reduce or close a position"""
        for i, pos in enumerate(self.positions):
            if (pos['type'] == trade['type'] and 
                pos.get('strike') == trade.get('strike') and
                pos.get('option_type') == trade.get('option_type') and
                pos.get('expiry') == trade.get('expiry')):
                
                if trade['quantity'] >= pos['quantity']:
                    # Close position completely
                    self.positions.pop(i)
                else:
                    # Reduce position
                    pos['quantity'] -= trade['quantity']
                break
    
    def _add_short_position(self, trade: Dict):
        """Add a short position"""
        # Check if short position already exists
        existing_short_position = None
        for pos in self.positions:
            if (pos['type'] == 'option' and pos['option_type'] == trade['option_type'] and
                pos.get('strike') == trade.get('strike') and
                pos.get('expiry') == trade.get('expiry')):
                existing_short_position = pos
                break
        
        if existing_short_position:
            # Increase quantity of existing short position (negative quantity)
            existing_short_position['quantity'] -= trade['quantity']  # Subtract to make more negative
        else:
            # Add new short position with negative quantity
            self.positions.append({
                'type': 'option',
                'quantity': -trade['quantity'],  # Negative for short position
                'avg_price': trade['price'], # For short positions, avg_price is the entry price
                'option_type': trade['option_type'],
                'strike': trade.get('strike'),
                'expiry': trade.get('expiry')
            })
    
    def _calculate_performance_metrics(self, daily_values: List[Dict], 
                                     initial_capital: float) -> Dict[str, Any]:
        """Calculate performance metrics from daily values"""
        if not daily_values:
            return {}
        
        df = pd.DataFrame(daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate returns
        df['daily_return'] = df['total_value'].pct_change()
        
        # Performance metrics
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Annualized return (assuming 252 trading days)
        days = (df.index[-1] - df.index[0]).days
        annualized_return = ((final_value / initial_capital) ** (365 / days)) - 1 if days > 0 else 0
        
        # Volatility
        volatility = df['daily_return'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + df['daily_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'daily_values': df,
            'trades': self.trades
        }

class IronCondorStrategy(OptionsStrategy):
    """
    Iron Condor Strategy Implementation
    Sells OTM call and put spreads
    """
    
    def __init__(self, data_loader: OptionsDataLoader, 
                 delta_threshold: float = 0.15,
                 days_to_expiry: int = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize Iron Condor strategy
        
        Args:
            data_loader: Data loader instance
            delta_threshold: Target delta for short strikes
            days_to_expiry: Target days to expiry
            risk_free_rate: Risk-free rate
        """
        super().__init__("Iron Condor", data_loader, risk_free_rate)
        self.delta_threshold = delta_threshold
        self.days_to_expiry = days_to_expiry
    
    def generate_signals(self, symbol: str, date: str, options_data: pd.DataFrame, 
                        stock_price: float) -> Dict[str, Any]:
        """Generate Iron Condor signals"""
        signals = {
            'action': 'none',
            'reason': 'No suitable opportunities',
            'positions': []
        }
        
        # Filter for options only - use OPTIDX for options
        options_df = options_data[options_data['INSTRUMENT'] == 'OPTIDX'].copy()
        
        if options_df.empty:
            return signals
        
        # Process options data to get Greeks
        processed_df = process_options_data(options_df, stock_price, date, self.risk_free_rate)
        
        if processed_df.empty:
            return signals
        
        # Find suitable expiries
        expiries = processed_df['EXPIRY_DT'].unique()
        suitable_expiry = None
        
        for expiry in expiries:
            expiry_options = processed_df[processed_df['EXPIRY_DT'] == expiry]
            days_to_exp = expiry_options['DAYS_TO_EXPIRY'].iloc[0]
            
            if abs(days_to_exp * 365.25 - self.days_to_expiry) <= 7:  # Within 7 days
                suitable_expiry = expiry
                break
        
        if suitable_expiry is None:
            return signals
        
        # Get options for the suitable expiry
        expiry_options = processed_df[processed_df['EXPIRY_DT'] == suitable_expiry]
        
        # Find short strikes based on delta
        ce_options = expiry_options[expiry_options['OPTION_TYP'] == 'CE'].copy()
        pe_options = expiry_options[expiry_options['OPTION_TYP'] == 'PE'].copy()
        
        # Find CE with delta close to threshold
        ce_options['delta_abs'] = abs(ce_options['delta'])
        short_call = ce_options[ce_options['delta_abs'] >= self.delta_threshold].iloc[0] if not ce_options.empty else None
        
        # Find PE with delta close to threshold
        pe_options['delta_abs'] = abs(pe_options['delta'])
        short_put = pe_options[pe_options['delta_abs'] >= self.delta_threshold].iloc[0] if not pe_options.empty else None
        
        if short_call is not None and short_put is not None:
            signals['action'] = 'sell_iron_condor'
            signals['reason'] = f'Found suitable strikes: CE {short_call["STRIKE_PR"]}, PE {short_put["STRIKE_PR"]}'
            signals['positions'] = [
                {
                    'type': 'option',
                    'action': 'sell',
                    'option_type': 'CE',
                    'strike': short_call['STRIKE_PR'],
                    'expiry': suitable_expiry,
                    'delta': short_call['delta'],
                    'price': short_call['CLOSE']
                },
                {
                    'type': 'option',
                    'action': 'sell',
                    'option_type': 'PE',
                    'strike': short_put['STRIKE_PR'],
                    'expiry': suitable_expiry,
                    'delta': short_put['delta'],
                    'price': short_put['CLOSE']
                }
            ]
        
        return signals
    
    def execute_trades(self, signals: Dict[str, Any], symbol: str, date: str, 
                      options_data: pd.DataFrame, stock_price: float) -> List[Dict]:
        """Execute Iron Condor trades"""
        trades = []
        
        if signals['action'] == 'sell_iron_condor':
            for position in signals['positions']:
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'type': 'option',
                    'option_type': position['option_type'],
                    'strike': position['strike'],
                    'expiry': position['expiry'],
                    'quantity': 1,  # 1 lot
                    'price': position['price'],
                    'reason': signals['reason']
                }
                trades.append(trade)
        
        return trades

class StraddleStrategy(OptionsStrategy):
    """
    Long Straddle Strategy Implementation
    Buys ATM call and put options
    """
    
    def __init__(self, data_loader: OptionsDataLoader, 
                 iv_percentile_threshold: float = 20.0,
                 days_to_expiry: int = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize Straddle strategy
        
        Args:
            data_loader: Data loader instance
            iv_percentile_threshold: IV percentile threshold for entry
            days_to_expiry: Target days to expiry
            risk_free_rate: Risk-free rate
        """
        super().__init__("Long Straddle", data_loader, risk_free_rate)
        self.iv_percentile_threshold = iv_percentile_threshold
        self.days_to_expiry = days_to_expiry
    
    def generate_signals(self, symbol: str, date: str, options_data: pd.DataFrame, 
                        stock_price: float) -> Dict[str, Any]:
        """Generate Straddle signals"""
        signals = {
            'action': 'none',
            'reason': 'No suitable opportunities',
            'positions': []
        }
        
        # Get ATM options
        atm_options = self.data_loader.get_atm_options(symbol, date, tolerance=0.01)
        
        if not atm_options or atm_options['CE'].empty or atm_options['PE'].empty:
            return signals
        
        # Check if we have suitable expiry
        ce_options = atm_options['CE']
        pe_options = atm_options['PE']
        
        # Find options with suitable expiry
        suitable_expiry = None
        for expiry in ce_options['EXPIRY_DT'].unique():
            expiry_ce = ce_options[ce_options['EXPIRY_DT'] == expiry]
            expiry_pe = pe_options[pe_options['EXPIRY_DT'] == expiry]
            
            if not expiry_ce.empty and not expiry_pe.empty:
                days_to_exp = expiry_ce['DAYS_TO_EXPIRY'].iloc[0]
                if abs(days_to_exp * 365.25 - self.days_to_expiry) <= 7:
                    suitable_expiry = expiry
                    break
        
        if suitable_expiry is None:
            return signals
        
        # Get ATM options for the suitable expiry
        atm_ce = ce_options[ce_options['EXPIRY_DT'] == suitable_expiry].iloc[0]
        atm_pe = pe_options[pe_options['EXPIRY_DT'] == suitable_expiry].iloc[0]
        
        # Check IV percentile (simplified - in practice you'd need historical IV data)
        # For now, we'll use a simple check based on current IV
        avg_iv = (atm_ce.get('IV', 0.3) + atm_pe.get('IV', 0.3)) / 2
        
        # Simple IV check (in practice, you'd calculate actual percentile)
        if avg_iv < 0.25:  # Low IV environment
            signals['action'] = 'buy_straddle'
            signals['reason'] = f'Low IV environment (avg IV: {avg_iv:.2%}), suitable for straddle'
            signals['positions'] = [
                {
                    'type': 'option',
                    'action': 'buy',
                    'option_type': 'CE',
                    'strike': atm_ce['STRIKE_PR'],
                    'expiry': suitable_expiry,
                    'price': atm_ce['CLOSE']
                },
                {
                    'type': 'option',
                    'action': 'buy',
                    'option_type': 'PE',
                    'strike': atm_pe['STRIKE_PR'],
                    'expiry': suitable_expiry,
                    'price': atm_pe['CLOSE']
                }
            ]
        
        return signals
    
    def execute_trades(self, signals: Dict[str, Any], symbol: str, date: str, 
                      options_data: pd.DataFrame, stock_price: float) -> List[Dict]:
        """Execute Straddle trades"""
        trades = []
        
        if signals['action'] == 'buy_straddle':
            for position in signals['positions']:
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': 'buy',
                    'type': 'option',
                    'option_type': position['option_type'],
                    'strike': position['strike'],
                    'expiry': position['expiry'],
                    'quantity': 1,  # 1 lot
                    'price': position['price'],
                    'reason': signals['reason']
                }
                trades.append(trade)
        
        return trades

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = OptionsDataLoader()
    
    # Test with available data
    symbols = loader.get_available_symbols()
    
    if symbols:
        test_symbol = symbols[0]  # Use first available symbol
        dates = loader.get_available_dates(test_symbol)
        
        if len(dates) >= 10:
            # Test Iron Condor strategy
            print("=== Testing Iron Condor Strategy ===")
            iron_condor = IronCondorStrategy(loader)
            
            # Run backtest for last 10 days
            start_date = dates[-10]
            end_date = dates[-1]
            
            results = iron_condor.backtest(test_symbol, start_date, end_date, 100000)
            
            print(f"\n📊 Iron Condor Results:")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            
            # Test Straddle strategy
            print("\n=== Testing Straddle Strategy ===")
            straddle = StraddleStrategy(loader)
            
            results = straddle.backtest(test_symbol, start_date, end_date, 100000)
            
            print(f"\n📊 Straddle Results:")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    print("\n🎉 Strategy framework test completed!") 