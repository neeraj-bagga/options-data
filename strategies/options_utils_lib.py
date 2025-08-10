#!/usr/bin/env python3
"""
Options Utility Functions using py_vollib library
Calculates Greeks, Implied Volatility, IV Percentile, and other option metrics
Uses py_vollib for accurate Black-Scholes calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import py_vollib.black_scholes.implied_volatility as bs_iv
    import py_vollib.black_scholes.greeks.analytical as bs_greeks
    from py_vollib.black_scholes import black_scholes as bs
    PY_VOLLIB_AVAILABLE = True
except ImportError:
    PY_VOLLIB_AVAILABLE = False
    print("Warning: py_vollib not available. Install with: pip install py_vollib")

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("Warning: QuantLib not available.")

class OptionsCalculatorLib:
    """
    Comprehensive options calculator using py_vollib library
    """
    
    def __init__(self, risk_free_rate: float = 0.055):
        """
        Initialize the options calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.risk_free_rate = risk_free_rate
        
        if not PY_VOLLIB_AVAILABLE:
            raise ImportError("py_vollib is required. Install with: pip install py_vollib")
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option price using Black-Scholes model
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(0, S - K)
        
        return bs('c', S, K, T, r, sigma)
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option price using Black-Scholes model
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(0, K - S)
        
        return bs('p', S, K, T, r, sigma)
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option delta using py_vollib
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Delta value
        """
        if T <= 0:
            if option_type == 'CE':
                return 1.0 if S > K else 0.0
            else:  # PE
                return -1.0 if S < K else 0.0
        
        flag = 'c' if option_type == 'CE' else 'p'
        return bs_greeks.delta(flag, S, K, T, r, sigma)
    
    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma (same for calls and puts)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Gamma value
        """
        if T <= 0:
            return 0.0
        
        return bs_greeks.gamma('c', S, K, T, r, sigma)  # Same for calls and puts
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option theta
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Theta value (per year)
        """
        if T <= 0:
            return 0.0
        
        flag = 'c' if option_type == 'CE' else 'p'
        return bs_greeks.theta(flag, S, K, T, r, sigma)
    
    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega (same for calls and puts)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Vega value
        """
        if T <= 0:
            return 0.0
        
        return bs_greeks.vega('c', S, K, T, r, sigma)  # Same for calls and puts
    
    def calculate_rho(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option rho
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Rho value
        """
        if T <= 0:
            return 0.0
        
        flag = 'c' if option_type == 'CE' else 'p'
        return bs_greeks.rho(flag, S, K, T, r, sigma)
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: str, tolerance: float = 1e-5, 
                                   max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using py_vollib
        
        Args:
            option_price: Observed option price
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            option_type: 'CE' for call, 'PE' for put
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Implied volatility
            
        Raises:
            ValueError: If IV calculation fails or parameters are invalid
        """
        # Type validation
        try:
            option_price = float(option_price)
            S = float(S)
            K = float(K)
            T = float(T)
            r = float(r)
            option_type = str(option_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion failed: {e}. "
                           f"Params: price={option_price} ({type(option_price)}), "
                           f"S={S} ({type(S)}), K={K} ({type(K)}), "
                           f"T={T} ({type(T)}), r={r} ({type(r)})")
        
        if T <= 0:
            raise ValueError(f"Cannot calculate IV for expired option (T={T})")
        
        if option_price <= 0:
            raise ValueError(f"Invalid option price: {option_price}")
        
        if S <= 0 or K <= 0:
            raise ValueError(f"Invalid stock price ({S}) or strike price ({K})")
        
        # Check if option price is below intrinsic value (data quality issue)
        if option_type == 'CE':
            intrinsic_value = max(0, S - K)
        else:  # PE
            intrinsic_value = max(0, K - S)
        
        if option_price < intrinsic_value:
            raise ValueError(f"Option price ({option_price}) is below intrinsic value ({intrinsic_value}). "
                           f"This indicates invalid data. S={S}, K={K}, type={option_type}")
        
        try:
            flag = 'c' if option_type == 'CE' else 'p'
            iv = bs_iv.implied_volatility(option_price, S, K, T, r, flag)
            
            if np.isnan(iv) or np.isinf(iv) or iv < 0:
                raise ValueError(f"Invalid IV result: {iv}")
            
            return iv
        except Exception as e:
            raise ValueError(f"Failed to calculate implied volatility: {e}. "
                           f"Params: price={option_price}, S={S}, K={K}, T={T}, r={r}, type={option_type}")
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str) -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Dictionary with all Greeks
        """
        return {
            'delta': self.calculate_delta(S, K, T, r, sigma, option_type),
            'gamma': self.calculate_gamma(S, K, T, r, sigma),
            'theta': self.calculate_theta(S, K, T, r, sigma, option_type),
            'vega': self.calculate_vega(S, K, T, r, sigma),
            'rho': self.calculate_rho(S, K, T, r, sigma, option_type)
        }
    
    def calculate_moneyness(self, S: float, K: float) -> float:
        """Calculate moneyness (S/K ratio)"""
        if K <= 0:
            raise ValueError(f"Invalid strike price: {K}")
        return S / K
    
    def calculate_time_value(self, option_price: float, S: float, K: float, option_type: str) -> float:
        """Calculate time value of option"""
        if option_type == 'CE':
            intrinsic_value = max(0, S - K)
        else:  # PE
            intrinsic_value = max(0, K - S)
        return option_price - intrinsic_value
    
    def calculate_breakeven(self, option_price: float, K: float, option_type: str) -> float:
        """Calculate breakeven point for option"""
        if option_type == 'CE':
            return K + option_price
        else:  # PE
            return K - option_price
    
    def calculate_probability_itm(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate probability of option ending in the money"""
        if T <= 0:
            if option_type == 'CE':
                return 1.0 if S > K else 0.0
            else:  # PE
                return 1.0 if S < K else 0.0
        
        # Use normal distribution approximation
        from scipy.stats import norm
        
        d2 = (np.log(S / K) + (self.risk_free_rate - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'CE':
            return norm.cdf(d2)
        else:  # PE
            return norm.cdf(-d2)
    
    def calculate_option_metrics(self, S: float, K: float, T: float, option_price: float, 
                               option_type: str, sigma: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive option metrics
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            option_price: Observed option price
            option_type: 'CE' for call, 'PE' for put
            sigma: Volatility (if None, will calculate IV)
            
        Returns:
            Dictionary with all option metrics
            
        Raises:
            ValueError: If IV calculation fails or parameters are invalid
        """
        # Calculate IV if not provided
        if sigma is None:
            sigma = self.calculate_implied_volatility(option_price, S, K, T, self.risk_free_rate, option_type)
        
        # Calculate all Greeks
        greeks = self.calculate_all_greeks(S, K, T, self.risk_free_rate, sigma, option_type)
        
        # Calculate other metrics
        metrics = {
            'implied_volatility': sigma,
            'moneyness': self.calculate_moneyness(S, K),
            'time_value': self.calculate_time_value(option_price, S, K, option_type),
            'breakeven': self.calculate_breakeven(option_price, K, option_type),
            'probability_itm': self.calculate_probability_itm(S, K, T, sigma, option_type),
            'intrinsic_value': max(0, S - K) if option_type == 'CE' else max(0, K - S)
        }
        
        # Combine all metrics
        all_metrics = {**greeks, **metrics}
        return all_metrics


def calculate_expiry_days(expiry_date: str, current_date: str = None) -> float:
    """
    Calculate days to expiry
    
    Args:
        expiry_date: Expiry date in various formats ('YYYY-MM-DD', 'DD-MMM-YYYY', etc.)
        current_date: Current date in 'YYYY-MM-DD' format (defaults to today)
        
    Returns:
        Days to expiry as a float
        
    Raises:
        ValueError: If date parsing fails
    """
    from datetime import datetime
    
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Try multiple date formats
    date_formats = [
        '%Y-%m-%d',      # 2024-08-05
        '%d-%b-%Y',      # 01-Apr-2020
        '%d-%B-%Y',      # 01-April-2020
        '%Y/%m/%d',      # 2024/08/05
        '%d/%m/%Y',      # 05/08/2024
        '%m/%d/%Y',      # 08/05/2024
        '%d-%m-%Y',      # 05-08-2024
        '%Y-%m-%d %H:%M:%S',  # 2024-08-05 00:00:00
    ]
    
    expiry = None
    for fmt in date_formats:
        try:
            expiry = datetime.strptime(expiry_date, fmt)
            break
        except ValueError:
            continue
    
    if expiry is None:
        raise ValueError(f"Failed to parse expiry date '{expiry_date}' with any supported format. "
                        f"Supported formats: {date_formats}")
    
    try:
        current = datetime.strptime(current_date, '%Y-%m-%d')
        days_diff = (expiry - current).days
        return max(0, days_diff / 365.0)  # Convert to years
    except ValueError as e:
        raise ValueError(f"Failed to parse current date '{current_date}'. Error: {e}")


def process_options_data_lib(options_df: pd.DataFrame, stock_price: float, 
                           current_date: str = None, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Process options data using library-based calculations
    
    Args:
        options_df: DataFrame with options data
        stock_price: Current stock price
        current_date: Current date for expiry calculation
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with added metrics
        
    Raises:
        ValueError: If data processing fails
    """
    calculator = OptionsCalculatorLib(risk_free_rate)
    
    # Create a copy to avoid modifying original
    df = options_df.copy()
    
    # Filter for options only (exclude futures)
    df = df[df['INSTRUMENT'] == 'OPTIDX'].copy()
    
    # Calculate days to expiry
    df['DAYS_TO_EXPIRY'] = df['EXPIRY_DT'].apply(
        lambda x: calculate_expiry_days(x, current_date)
    )
    
    # Filter out invalid options before processing
    initial_count = len(df)
    
    # Remove options with invalid prices or strikes
    df = df[df['SETTLE_PR'] > 0].copy()
    df = df[df['STRIKE_PR'] > 0].copy()
    
    # Remove options with zero or negative time to expiry
    df = df[df['DAYS_TO_EXPIRY'] > 0].copy()
    
    
    filtered_count = len(df)
    removed_count = initial_count - filtered_count
    
    if removed_count > 0:
        print(f"⚠️  Filtered out {removed_count} invalid options (price below intrinsic value, zero prices, etc.)")
        print(f"   Processing {filtered_count} valid options out of {initial_count} total")
    
    if filtered_count == 0:
        raise ValueError("No valid options found after filtering. Check data quality and stock price.")
    
    # Initialize new columns
    metrics_columns = ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility', 
                      'moneyness', 'time_value', 'breakeven', 'probability_itm', 'intrinsic_value']
    
    for col in metrics_columns:
        df[col] = np.nan
    
    # Calculate metrics for each option
    failed_rows = []
    for idx, row in df.iterrows():
        try:
            S = float(stock_price)
            K = float(row['STRIKE_PR'])
            T = float(row['DAYS_TO_EXPIRY'])
            option_price = float(row['SETTLE_PR'])
            option_type = str(row['OPTION_TYP'])
            
            # Validate that all numeric parameters are valid
            if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in [S, K, T, option_price]):
                raise ValueError(f"Invalid numeric parameters: S={S}, K={K}, T={T}, price={option_price}")
            
            metrics = calculator.calculate_option_metrics(
                S, K, T, option_price, option_type
            )
            
            for metric_name, value in metrics.items():
                df.at[idx, metric_name] = value
                
        except (ValueError, TypeError) as e:
            failed_rows.append({
                'index': idx,
                'symbol': row.get('SYMBOL', 'Unknown'),
                'strike': row.get('STRIKE_PR', 'Unknown'),
                'option_type': row.get('OPTION_TYP', 'Unknown'),
                'error': str(e)
            })
    
    if failed_rows:
        error_msg = f"Failed to process {len(failed_rows)} options rows:\n"
        for row in failed_rows[:5]:  # Show first 5 errors
            error_msg += f"  Row {row['index']}: {row['symbol']} {row['strike']} {row['option_type']} - {row['error']}\n"
        if len(failed_rows) > 5:
            error_msg += f"  ... and {len(failed_rows) - 5} more errors\n"
        print(error_msg)
    
    return df


def calculate_iv_percentile_for_symbol_lib(options_data: pd.DataFrame, symbol: str, 
                                         lookback_days: int = 252) -> Dict[str, float]:
    """
    Calculate IV percentile for a symbol using library-based calculations
    
    Args:
        options_data: DataFrame with options data
        symbol: Symbol to analyze
        lookback_days: Lookback period for IV percentile
        
    Returns:
        Dictionary with IV percentiles for different categories
        
    Raises:
        ValueError: If no data found for symbol
    """
    # Filter for the symbol
    symbol_data = options_data[options_data['SYMBOL'] == symbol].copy()
    
    if symbol_data.empty:
        raise ValueError(f"No options data found for symbol: {symbol}")
    
    # Group by expiry and calculate average IV
    expiry_ivs = symbol_data.groupby('EXPIRY_DT')['implied_volatility'].mean()
    
    # Group by moneyness categories
    symbol_data['moneyness_category'] = pd.cut(
        symbol_data['moneyness'], 
        bins=[0, 0.9, 0.95, 1.0, 1.05, 1.1, float('inf')],
        labels=['Deep OTM', 'OTM', 'Near ATM', 'ATM', 'ITM', 'Deep ITM']
    )
    
    moneyness_ivs = symbol_data.groupby('moneyness_category')['implied_volatility'].mean()
    
    # Calculate percentiles (simplified - in real implementation you'd need historical data)
    current_avg_iv = symbol_data['implied_volatility'].mean()
    
    iv_percentiles = {
        'overall_iv_percentile': 50.0,  # Placeholder
        'current_avg_iv': current_avg_iv,
        'expiry_ivs': expiry_ivs.to_dict(),
        'moneyness_ivs': moneyness_ivs.to_dict()
    }
    
    return iv_percentiles


# Example usage and testing
if __name__ == "__main__":
    # Test the options calculator
    calculator = OptionsCalculatorLib(risk_free_rate=0.05)
    
    # Example parameters
    S = 100.0  # Stock price
    K = 100.0  # Strike price
    T = 0.25   # 3 months to expiry
    r = 0.05   # 5% risk-free rate
    sigma = 0.3  # 30% volatility
    
    print("=== Options Calculator (Library-based) Test ===")
    print(f"py_vollib available: {PY_VOLLIB_AVAILABLE}")
    print(f"QuantLib available: {QUANTLIB_AVAILABLE}")
    print(f"Stock Price: {S}")
    print(f"Strike Price: {K}")
    print(f"Time to Expiry: {T} years")
    print(f"Risk-free Rate: {r}")
    print(f"Volatility: {sigma}")
    print()
    
    # Calculate call option metrics
    call_price = calculator.black_scholes_call(S, K, T, r, sigma)
    call_greeks = calculator.calculate_all_greeks(S, K, T, r, sigma, 'CE')
    
    print("Call Option:")
    print(f"Price: {call_price:.4f}")
    print("Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    print()
    
    # Calculate put option metrics
    put_price = calculator.black_scholes_put(S, K, T, r, sigma)
    put_greeks = calculator.calculate_all_greeks(S, K, T, r, sigma, 'PE')
    
    print("Put Option:")
    print(f"Price: {put_price:.4f}")
    print("Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    print()
    
    # Test IV calculation
    calculated_iv = calculator.calculate_implied_volatility(call_price, S, K, T, r, 'CE')
    print(f"Implied Volatility (from call price): {calculated_iv:.6f}")
    print()
    
    # Test comprehensive metrics
    call_metrics = calculator.calculate_option_metrics(S, K, T, call_price, 'CE', sigma)
    print("Call Option Comprehensive Metrics:")
    for metric, value in call_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.6f}") 