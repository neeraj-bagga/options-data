#!/usr/bin/env python3
"""
Options Utility Functions for Backtesting
Calculates Greeks, Implied Volatility, IV Percentile, and other option metrics
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class OptionsCalculator:
    """
    Comprehensive options calculator for Greeks, IV, and other metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the options calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function"""
        return norm.cdf(x)
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function"""
        return norm.pdf(x)
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            return (0, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
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
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        call_price = S * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        return call_price
    
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
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        put_price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
        return put_price
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
        Calculate option delta
        
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
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == 'CE':
            return self._normal_cdf(d1)
        else:  # PE
            return self._normal_cdf(d1) - 1
    
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
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        gamma = self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
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
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        theta_term1 = -(S * sigma * self._normal_pdf(d1)) / (2 * np.sqrt(T))
        
        if option_type == 'CE':
            theta_term2 = -r * K * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # PE
            theta_term2 = -r * K * np.exp(-r * T) * self._normal_cdf(-d2)
        
        return theta_term1 + theta_term2
    
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
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        vega = S * np.sqrt(T) * self._normal_pdf(d1)
        return vega
    
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
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == 'CE':
            return K * T * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # PE
            return -K * T * np.exp(-r * T) * self._normal_cdf(-d2)
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: str, tolerance: float = 1e-5, 
                                   max_iterations: int = 100) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
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
            Implied volatility or None if not found
        """
        if T <= 0:
            return None
        
        # Initial guess for volatility
        sigma = 0.5
        
        for i in range(max_iterations):
            if option_type == 'CE':
                price = self.black_scholes_call(S, K, T, r, sigma)
                vega = self.calculate_vega(S, K, T, r, sigma)
            else:  # PE
                price = self.black_scholes_put(S, K, T, r, sigma)
                vega = self.calculate_vega(S, K, T, r, sigma)
            
            diff = option_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            if abs(vega) < 1e-10:
                break
            
            sigma = sigma + diff / vega
            
            # Ensure volatility stays positive
            sigma = max(0.001, sigma)
        
        return None
    
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
    
    def calculate_iv_percentile(self, current_iv: float, historical_ivs: List[float], 
                              lookback_days: int = 252) -> float:
        """
        Calculate IV percentile based on historical data
        
        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical implied volatilities
            lookback_days: Number of days to look back (default: 252 trading days)
            
        Returns:
            IV percentile (0-100)
        """
        if not historical_ivs or len(historical_ivs) < 2:
            return 50.0  # Default to median if insufficient data
        
        # Use the most recent lookback_days
        recent_ivs = historical_ivs[-lookback_days:] if len(historical_ivs) > lookback_days else historical_ivs
        
        # Calculate percentile
        percentile = (sum(1 for iv in recent_ivs if iv <= current_iv) / len(recent_ivs)) * 100
        return percentile
    
    def calculate_moneyness(self, S: float, K: float) -> float:
        """
        Calculate moneyness (S/K ratio)
        
        Args:
            S: Current stock price
            K: Strike price
            
        Returns:
            Moneyness ratio
        """
        return S / K
    
    def calculate_time_value(self, option_price: float, S: float, K: float, option_type: str) -> float:
        """
        Calculate time value of option
        
        Args:
            option_price: Current option price
            S: Current stock price
            K: Strike price
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Time value
        """
        if option_type == 'CE':
            intrinsic_value = max(0, S - K)
        else:  # PE
            intrinsic_value = max(0, K - S)
        
        return option_price - intrinsic_value
    
    def calculate_breakeven(self, option_price: float, K: float, option_type: str) -> float:
        """
        Calculate breakeven point for option
        
        Args:
            option_price: Option price paid
            K: Strike price
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Breakeven stock price
        """
        if option_type == 'CE':
            return K + option_price
        else:  # PE
            return K - option_price
    
    def calculate_probability_itm(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """
        Calculate probability of option ending in the money
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            sigma: Volatility
            option_type: 'CE' for call, 'PE' for put
            
        Returns:
            Probability (0-1)
        """
        if T <= 0:
            if option_type == 'CE':
                return 1.0 if S > K else 0.0
            else:  # PE
                return 1.0 if S < K else 0.0
        
        d2 = self._d1_d2(S, K, T, self.risk_free_rate, sigma)[1]
        
        if option_type == 'CE':
            return self._normal_cdf(d2)
        else:  # PE
            return self._normal_cdf(-d2)
    
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
        """
        # Calculate IV if not provided
        if sigma is None:
            sigma = self.calculate_implied_volatility(option_price, S, K, T, self.risk_free_rate, option_type)
            if sigma is None:
                raise ValueError("Implied volatility not found")
        
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
    Calculate days to expiration
    
    Args:
        expiry_date: Expiry date in format 'DD-MMM-YYYY' or 'YYYY-MM-DD'
        current_date: Current date (if None, uses today)
        
    Returns:
        Days to expiration
    """
    from datetime import datetime
    
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Try different date formats
        for fmt in ['%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                expiry = datetime.strptime(expiry_date, fmt)
                current = datetime.strptime(current_date, '%Y-%m-%d')
                days = (expiry - current).days
                return max(0, days / 365.25)  # Convert to years
            except ValueError:
                continue
    except:
        pass
    
    return 0.0

def process_options_data(options_df: pd.DataFrame, stock_price: float, 
                        current_date: str = None, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Process options data and add calculated metrics
    
    Args:
        options_df: DataFrame with options data
        stock_price: Current stock price
        current_date: Current date for expiry calculation
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with added metrics
    """
    calculator = OptionsCalculator(risk_free_rate)
    
    # Create a copy to avoid modifying original
    df = options_df.copy()
    
    # Filter for options only (exclude futures)
    df = df[df['INSTRUMENT'] == 'OPTIDX'].copy()
    
    # Calculate days to expiry
    df['DAYS_TO_EXPIRY'] = df['EXPIRY_DT'].apply(
        lambda x: calculate_expiry_days(x, current_date)
    )
    
    # Initialize new columns
    metrics_columns = ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility', 
                      'moneyness', 'time_value', 'breakeven', 'probability_itm', 'intrinsic_value']
    
    for col in metrics_columns:
        df[col] = np.nan
    
    # Calculate metrics for each option
    for idx, row in df.iterrows():
        try:
            S = stock_price
            K = float(row['STRIKE_PR'])
            T = row['DAYS_TO_EXPIRY']
            option_price = float(row['CLOSE'])
            option_type = row['OPTION_TYP']
            
            if T > 0 and option_price > 0:
                metrics = calculator.calculate_option_metrics(
                    S, K, T, option_price, option_type
                )
                
                for metric_name, value in metrics.items():
                    df.at[idx, metric_name] = value
                    
        except (ValueError, TypeError) as e:
            continue
    
    return df

def calculate_iv_percentile_for_symbol(options_data: pd.DataFrame, symbol: str, 
                                     lookback_days: int = 252) -> Dict[str, float]:
    """
    Calculate IV percentile for a symbol across different expiries and strikes
    
    Args:
        options_data: DataFrame with options data
        symbol: Symbol to analyze
        lookback_days: Lookback period for IV percentile
        
    Returns:
        Dictionary with IV percentiles for different categories
    """
    # Filter for the symbol
    symbol_data = options_data[options_data['SYMBOL'] == symbol].copy()
    
    if symbol_data.empty:
        return {}
    
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
    
    # For demonstration, we'll use a simple percentile calculation
    # In practice, you'd need historical IV data
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
    calculator = OptionsCalculator(risk_free_rate=0.05)
    
    # Example parameters
    S = 100.0  # Stock price
    K = 100.0  # Strike price
    T = 0.25   # 3 months to expiry
    r = 0.05   # 5% risk-free rate
    sigma = 0.3  # 30% volatility
    
    print("=== Options Calculator Test ===")
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