#!/usr/bin/env python3
"""
Example Usage of Options Strategies Framework with Library-based Calculations
Demonstrates the use of py_vollib for accurate options pricing and Greeks calculations
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import OptionsDataLoader
from options_utils_lib import OptionsCalculatorLib, process_options_data_lib, calculate_iv_percentile_for_symbol_lib
from strategy_framework import IronCondorStrategy, StraddleStrategy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from stocks import StockName, STOCK_NAME_MAP

def main():
    print("🚀 Options Strategies Framework - Library-based Calculations")
    print("=" * 60)
    
    # Test parameters
    test_symbol = StockName.NIFTY.value
    test_date = "2023-10-10"
    
    print(f"📊 Testing with symbol: {test_symbol}")
    print(f"📅 Testing with date: {test_date}")
    print()
    
    # Example 1: Data Loading
    print("=" * 50)
    print("1. Data Loading")
    print("=" * 50)
    
    loader = OptionsDataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "market_data"))
    
    # Load options data
    print(f"📂 Loading options data for {test_symbol} on {test_date}...")
    options_data = loader.load_options_data(test_symbol, test_date)
    
    if options_data is not None:
        print(f"✅ Loaded {len(options_data)} options records")
        print(f"📋 Columns: {list(options_data.columns)}")
        
        # Show sample data
        print(f"\n📊 Sample options data:")
        print(options_data.head(3))
        
        # Get stock price
        stock_price = loader.get_stock_price(test_symbol, test_date)
        print(f"\n💰 Stock price on {test_date}: {stock_price}")
        
        # Process options data using library-based calculations
        print(f"\n🧮 Processing options data with py_vollib for Greeks and IV...")
        processed_data = process_options_data_lib(options_data, stock_price, test_date)
        
        if processed_data is not None and not processed_data.empty:
            print(f"✅ Processed {len(processed_data)} options with Greeks and IV")
            
            # Show options with calculated metrics
            metrics_columns = ['STRIKE_PR', 'OPTION_TYP', 'SETTLE_PR', 'delta', 'gamma', 
                             'theta', 'vega', 'implied_volatility', 'moneyness']
            
            available_metrics = [col for col in metrics_columns if col in processed_data.columns]
            
            print(f"\n📊 Sample processed data with metrics:")
            sample_data = processed_data[available_metrics].head(5)
            print(sample_data)
            
            # Show ATM options
            atm_options = loader.get_atm_options(test_symbol, test_date)
            if atm_options:
                print(f"\n🎯 ATM Options (Stock Price: {atm_options['stock_price']}):")
                print(f"   CE options: {len(atm_options['CE'])}")
                print(f"   PE options: {len(atm_options['PE'])}")
                
                if not atm_options['CE'].empty:
                    print(f"\n📊 Sample ATM CE option:")
                    ce_sample = atm_options['CE'].head(1)
                    # Check what columns are available and use only those that exist
                    available_columns = ['STRIKE_PR', 'SETTLE_PR', 'OPEN_INT']
                    if 'VOLUME' in ce_sample.columns:
                        available_columns.append('VOLUME')
                    if 'CLOSE' in ce_sample.columns:
                        available_columns.append('CLOSE')
                    print(ce_sample[available_columns])
    
    # Example 2: Library-based Options Calculator Usage
    print("\n" + "=" * 50)
    print("2. Library-based Options Calculator Usage")
    print("=" * 50)
    
    calculator = OptionsCalculatorLib(risk_free_rate=0.05)
    
    # Example parameters
    S = 100.0  # Stock price
    K = 100.0  # Strike price
    T = 0.25   # 3 months to expiry
    sigma = 0.3  # 30% volatility
    
    print(f"\n🧮 Calculating option metrics with py_vollib:")
    print(f"   Stock Price: {S}")
    print(f"   Strike Price: {K}")
    print(f"   Time to Expiry: {T} years")
    print(f"   Volatility: {sigma}")
    
    # Calculate call option
    call_price = calculator.black_scholes_call(S, K, T, 0.05, sigma)
    call_greeks = calculator.calculate_all_greeks(S, K, T, 0.05, sigma, 'CE')
    
    print(f"\n📈 Call Option (py_vollib):")
    print(f"   Price: {call_price:.4f}")
    print(f"   Delta: {call_greeks['delta']:.4f}")
    print(f"   Gamma: {call_greeks['gamma']:.4f}")
    print(f"   Theta: {call_greeks['theta']:.4f}")
    print(f"   Vega: {call_greeks['vega']:.4f}")
    print(f"   Rho: {call_greeks['rho']:.4f}")
    
    # Calculate put option
    put_price = calculator.black_scholes_put(S, K, T, 0.05, sigma)
    put_greeks = calculator.calculate_all_greeks(S, K, T, 0.05, sigma, 'PE')
    
    print(f"\n📉 Put Option (py_vollib):")
    print(f"   Price: {put_price:.4f}")
    print(f"   Delta: {put_greeks['delta']:.4f}")
    print(f"   Gamma: {put_greeks['gamma']:.4f}")
    print(f"   Theta: {put_greeks['theta']:.4f}")
    print(f"   Vega: {put_greeks['vega']:.4f}")
    print(f"   Rho: {put_greeks['rho']:.4f}")
    
    # Test IV calculation
    calculated_iv = calculator.calculate_implied_volatility(call_price, S, K, T, 0.05, 'CE')
    print(f"\n📊 Implied Volatility (py_vollib): {calculated_iv:.6f}")
    
    # Test comprehensive metrics
    call_metrics = calculator.calculate_option_metrics(S, K, T, call_price, 'CE', sigma)
    print(f"\n📊 Call Option Comprehensive Metrics (py_vollib):")
    for metric, value in call_metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value:.6f}")
    
    # Example 3: IV Percentile Analysis
    print("\n" + "=" * 50)
    print("3. IV Percentile Analysis")
    print("=" * 50)
    
    if processed_data is not None and not processed_data.empty:
        print(f"\n📊 Calculating IV percentiles for {STOCK_NAME_MAP[StockName(test_symbol)].nfo}...")
        iv_analysis = calculate_iv_percentile_for_symbol_lib(processed_data, STOCK_NAME_MAP[StockName(test_symbol)].nfo)
        
        if iv_analysis:
            print(f"   Overall IV Percentile: {iv_analysis.get('overall_iv_percentile', 'N/A')}")
            print(f"   Current Average IV: {iv_analysis.get('current_avg_iv', 'N/A'):.4f}")
            
            print(f"\n📊 IV by Moneyness Categories:")
            moneyness_ivs = iv_analysis.get('moneyness_ivs', {})
            for category, iv in moneyness_ivs.items():
                if pd.notna(iv):
                    print(f"   {category}: {iv:.4f}")
    
    # Example 4: Strategy Backtesting with Library-based Calculations
    print("\n" + "=" * 50)
    print("4. Strategy Backtesting with Library-based Calculations")
    print("=" * 50)
    
    if processed_data is not None and not processed_data.empty:
        print(f"\n🎯 Testing Iron Condor Strategy with library-based calculations...")
        
        # Create strategy instance
        iron_condor = IronCondorStrategy(
            data_loader=loader,
            delta_threshold=0.15,
            days_to_expiry=30,
            risk_free_rate=0.05
        )
        
        # Run backtest
        try:
            # Use a date range for backtesting
            start_date = "2023-10-10"
            end_date = "2023-10-27"
            
            results = iron_condor.backtest(
                symbol=test_symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000
            )
            
            if results:
                print(f"✅ Backtest completed successfully!")
                print(f"   Total Return: {results['total_return']:.2%}")
                print(f"   Annualized Return: {results['annualized_return']:.2%}")
                print(f"   Volatility: {results['volatility']:.2%}")
                print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
                print(f"   Total Trades: {results['total_trades']}")
            else:
                print("❌ Backtest failed or no trades generated")
                
        except Exception as e:
            print(f"❌ Error during backtest: {e}")
    
   

if __name__ == "__main__":
    main() 