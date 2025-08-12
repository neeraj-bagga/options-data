#!/usr/bin/env python3
"""
Options Utility Functions using py_vollib library
Calculates Greeks, Implied Volatility, IV Percentile, and other option metrics
Uses py_vollib for accurate Black-Scholes calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

# Always require py_vollib and py_vollib_vectorized
import py_vollib.black_scholes.implied_volatility as bs_iv
from py_vollib.black_scholes import black_scholes as bs
import py_vollib_vectorized  # noqa: F401 - patches py_vollib to accept vectorized inputs
from py_vollib_vectorized import get_all_greeks

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
        self.risk_free_rate = risk_free_rate
        # py_vollib is required and imported above
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0, S - K)
        return bs('c', S, K, T, r, sigma)
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0, K - S)
        return bs('p', S, K, T, r, sigma)
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: str, tolerance: float = 1e-5, 
                                   max_iterations: int = 100) -> float:
        # Type validation
        try:
            option_price = float(option_price)
            S = float(S)
            K = float(K)
            T = float(T)
            r = float(r)
            option_type = str(option_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion failed: {e}.")
        if T <= 0:
            raise ValueError(f"Cannot calculate IV for expired option (T={T})")
        if option_price <= 0 or S <= 0 or K <= 0:
            raise ValueError("Invalid parameters for IV calculation")
        if option_type == 'CE':
            intrinsic_value = max(0, S - K)
        else:
            intrinsic_value = max(0, K - S)
        tol = 0.02
        min_allowed = intrinsic_value * (1 - tol)
        if option_price < min_allowed:
            raise ValueError(
                f"Option price ({option_price}) is below intrinsic value ({intrinsic_value}). "
                f"This indicates invalid data. S={S}, K={K}, type={option_type}"
            )
        option_price = max(option_price, intrinsic_value * 1.000001)
        try:
            flag = 'c' if option_type == 'CE' else 'p'
            iv = bs_iv.implied_volatility(option_price, S, K, T, r, flag)
            if np.isnan(iv) or np.isinf(iv) or iv < 0:
                raise ValueError(f"Invalid IV result: {iv}")
            return iv
        except Exception as e:
            raise ValueError(f"Failed to calculate implied volatility: {e}.")
    
    def _calculate_all_greeks_vectorized_single(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        flags = np.array(['c' if option_type == 'CE' else 'p'])
        S_arr = np.array([S], dtype='float64')
        K_arr = np.array([K], dtype='float64')
        T_arr = np.array([T], dtype='float64')
        r_arr = np.array([r], dtype='float64')
        sigma_arr = np.array([sigma], dtype='float64')
        out = get_all_greeks(flags, S_arr, K_arr, T_arr, r_arr, sigma_arr, model='black_scholes', return_as='dict')
        return {k: float(v[0]) for k, v in out.items()}
    
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
        
        # Calculate all Greeks using a single vectorized call
        greeks = self._calculate_all_greeks_vectorized_single(S, K, T, self.risk_free_rate, sigma, option_type)
        
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
    Calculate days to expiry as a fraction of years. Supports multiple date formats.
    """
    from datetime import datetime as _dt
    if current_date is None:
        current_date = _dt.now().strftime('%Y-%m-%d')
    date_formats = [
        '%Y-%m-%d',      # 2024-08-05
        '%d-%b-%Y',      # 01-Apr-2020
        '%d-%B-%Y',      # 01-April-2020
        '%Y/%m/%d',      # 2024/08/05
        '%d/%m/%Y',      # 05/08/2024
        '%m/%d/%Y',      # 08/05/2024
        '%d-%m-%Y',      # 05-08-2024
        '%Y-%m-%d %H:%M:%S',
    ]
    expiry = None
    for fmt in date_formats:
        try:
            expiry = _dt.strptime(str(expiry_date), fmt)
            break
        except ValueError:
            continue
    if expiry is None:
        # Fallback: try pandas to_datetime
        try:
            expiry = pd.to_datetime(expiry_date).to_pydatetime()
        except Exception:
            return 0.0
    try:
        current = _dt.strptime(current_date, '%Y-%m-%d')
        days_diff = (expiry - current).days
        return max(0, days_diff / 365.0)
    except ValueError:
        return 0.0


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfc
    return 0.5 * erfc(-x / np.sqrt(2.0))


def process_options_data_lib(
    options_df: pd.DataFrame,
    stock_price: float,
    current_date: str = None,
    risk_free_rate: float = 0.05,
    verbose: bool = False,
    fast_mode: bool = True,
    iv_strategy: str = 'per_expiry',
    engine: str = 'auto'
) -> pd.DataFrame:
    t_start = perf_counter()
    calculator = OptionsCalculatorLib(risk_free_rate)
    df = options_df.copy()
    if 'INSTRUMENT' in df.columns:
        df = df[df['INSTRUMENT'].isin(['OPTIDX', 'OPTSTK'])].copy()
    df['DAYS_TO_EXPIRY'] = df['EXPIRY_DT'].apply(lambda x: calculate_expiry_days(x, current_date))
    initial_count = len(df)
    if 'SETTLE_PR' in df.columns:
        df = df[df['SETTLE_PR'] > 0].copy()
    if 'STRIKE_PR' in df.columns:
        df = df[df['STRIKE_PR'] > 0].copy()
    df = df[df['DAYS_TO_EXPIRY'] > 0].copy()
    filtered_count = len(df)
    removed_count = initial_count - filtered_count
    if verbose:
        print(f"process_options_data: start rows={initial_count}, filtered={filtered_count}, removed={removed_count}")
    if filtered_count == 0:
        return pd.DataFrame()

    # Prepare output columns
    metrics_columns = [
        'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility',
        'moneyness', 'time_value', 'breakeven', 'probability_itm', 'intrinsic_value'
    ]
    for col in metrics_columns:
        df[col] = np.nan

    # Vectorized IV for all rows via py_vollib_vectorized
    t_iv_start = perf_counter()
    flags = np.where(df['OPTION_TYP'].values == 'CE', 'c', 'p')
    S_arr = np.full(len(df), float(stock_price), dtype='float64')
    K_arr = df['STRIKE_PR'].values.astype('float64')
    T_arr = df['DAYS_TO_EXPIRY'].values.astype('float64')
    r_arr = np.full(len(df), float(risk_free_rate), dtype='float64')
    prices = df['SETTLE_PR'].values.astype('float64')
    iv_arr = bs_iv.implied_volatility(prices, S_arr, K_arr, T_arr, r_arr, flags)
    # Ensure 1-D numpy array
    iv_arr = np.asarray(iv_arr, dtype='float64')
    iv_arr = np.ravel(iv_arr)
    # Sanitize
    mask_bad = (~np.isfinite(iv_arr)) | (iv_arr <= 0)
    if np.any(mask_bad):
        good = iv_arr[~mask_bad]
        fallback = float(np.median(good)) if good.size > 0 else 0.25
        iv_arr = np.where(mask_bad, fallback, iv_arr)
    sigma_series = pd.Series(iv_arr, index=df.index)
    if verbose:
        dt_iv = perf_counter() - t_iv_start
        print(f"IV(vectorized) solved {len(df)} rows in {dt_iv:.3f}s")

    # Vectorized greeks using py_vollib_vectorized
    t_greeks_start = perf_counter()
    S_arr = np.full_like(sigma_series.values.astype('float64'), float(stock_price), dtype='float64')
    K_arr = df['STRIKE_PR'].values.astype('float64')
    T_arr = df['DAYS_TO_EXPIRY'].values.astype('float64')
    sigma_arr = sigma_series.values.astype('float64')
    is_call_arr = (df['OPTION_TYP'].values == 'CE')

    flags = np.where(is_call_arr, 'c', 'p')
    r_arr = np.full(len(df), float(risk_free_rate), dtype='float64')
    greeks_dict = get_all_greeks(flags, S_arr, K_arr, T_arr, r_arr, sigma_arr,
                                 model='black_scholes', return_as='dict')
    df['delta'] = greeks_dict['delta']
    df['gamma'] = greeks_dict['gamma']
    df['theta'] = greeks_dict['theta']
    df['vega'] = greeks_dict['vega']
    df['rho'] = greeks_dict['rho']
    df['implied_volatility'] = sigma_arr

    # Other metrics (vectorized)
    df['moneyness'] = S_arr / K_arr
    intrinsic_call = np.maximum(0.0, S_arr - K_arr)
    intrinsic_put = np.maximum(0.0, K_arr - S_arr)
    is_call_num = is_call_arr.astype('float64')
    intrinsic = is_call_num * intrinsic_call + (1.0 - is_call_num) * intrinsic_put
    prices = df['SETTLE_PR'].values.astype('float64')
    df['time_value'] = prices - intrinsic
    df['breakeven'] = np.where(is_call_arr, K_arr + prices, K_arr - prices)
    # probability ITM ~ N(d2) for calls, N(-d2) for puts
    small = 1e-12
    T_arr2 = np.clip(T_arr, small, None)
    sigma_arr2 = np.clip(sigma_arr, small, None)
    d1 = (np.log(S_arr / K_arr) + (risk_free_rate + 0.5 * sigma_arr2 * sigma_arr2) * T_arr2) / (sigma_arr2 * np.sqrt(T_arr2))
    d2 = d1 - sigma_arr2 * np.sqrt(T_arr2)
    prob_itm = np.where(is_call_arr, _norm_cdf(d2), _norm_cdf(-d2))
    df['probability_itm'] = prob_itm
    df['intrinsic_value'] = intrinsic

    if verbose:
        dt_greeks = perf_counter() - t_greeks_start
        dt_total = perf_counter() - t_start
        print(f"Greeks(vectorized) computed in {dt_greeks:.3f}s; total process time {dt_total:.3f}s for {len(df)} rows")

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
    print("py_vollib available: True")
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