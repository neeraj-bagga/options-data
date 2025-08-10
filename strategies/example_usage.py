#!/usr/bin/env python3
"""
Example Usage of Options Strategies Framework
Demonstrates how to use the framework for backtesting options strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import OptionsDataLoader
from options_utils import OptionsCalculator, process_options_data
from strategy_framework import IronCondorStrategy, StraddleStrategy
import pandas as pd

def main():
    """Main example function"""
    print("🚀 Options Strategies Framework - Example Usage")
    print("=" * 50)
    
    # Initialize data loader
    print("\n1. Initializing Data Loader...")
    loader = OptionsDataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "market_data"))
    
    # Get available symbols
    symbols = loader.get_available_symbols()
    print(f"📊 Available symbols: {len(symbols)}")
    print(f"   First 10 symbols: {symbols[:10]}")
    
    if not symbols:
        print("❌ No symbols found. Please check your data directory.")
        return
    
    # Use first available symbol for demonstration
    test_symbol = symbols[0]
    print(f"\n🎯 Using symbol: {test_symbol}")
    
    # Get available dates
    dates = loader.get_available_dates(test_symbol)
    print(f"📅 Available dates: {len(dates)}")
    
    if len(dates) < 5:
        print("❌ Insufficient data for backtesting. Need at least 5 days.")
        return
    
    # Debug: Show some sample dates
    print(f"📅 Sample dates (first 5): {dates[:5]}")
    print(f"📅 Sample dates (last 5): {dates[-5:]}")
    
    # Use historical dates for testing (avoid future dates)
    # Since system date might be incorrect, use a reasonable cutoff date
    from datetime import datetime
    # Use 2024-12-31 as a reasonable cutoff for historical data
    cutoff_date = datetime(2024, 12, 31)
    print(f"📅 Using cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    
    # Convert dates to datetime objects for proper comparison
    historical_dates = []
    for d in dates:
        try:
            date_obj = datetime.strptime(d, '%Y-%m-%d')
            if date_obj <= cutoff_date:
                historical_dates.append(d)
        except ValueError:
            continue  # Skip invalid dates
    
    print(f"📅 Historical dates found: {len(historical_dates)}")
    if historical_dates:
        print(f"📅 Sample historical dates (last 5): {historical_dates[-5:]}")
    
    if len(historical_dates) < 5:
        print("❌ Insufficient historical data for backtesting. Need at least 5 days.")
        print(f"Available historical dates: {len(historical_dates)}")
        return
    
    # Use recent historical dates for testing
    test_date = historical_dates[-1]  # Most recent historical date
    start_date = historical_dates[-5]  # 5 days ago
    end_date = historical_dates[-1]    # Most recent historical date
    
    print(f"📅 Test period: {start_date} to {end_date}")
    
    # Example 1: Basic Data Loading and Processing
    print("\n" + "=" * 50)
    print("2. Basic Data Loading and Processing")
    print("=" * 50)
    
    # Load options data
    print(f"\n📥 Loading options data for {test_symbol} on {test_date}...")
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
        
        # Process options data to get Greeks and IV
        print(f"\n🧮 Processing options data to calculate Greeks and IV...")
        
        # Debug: Show instrument types
        print(f"📊 Instrument types in data: {options_data['INSTRUMENT'].unique()}")
        print(f"📊 Option types in data: {options_data['OPTION_TYP'].unique()}")
        
        processed_data = process_options_data(options_data, stock_price, test_date)
        
        if processed_data is not None and not processed_data.empty:
            print(f"✅ Processed {len(processed_data)} options with Greeks and IV")
        else:
            print(f"⚠️  No options processed - check if options data is valid")
            
            # Show options with calculated metrics
            metrics_columns = ['STRIKE_PR', 'OPTION_TYP', 'CLOSE', 'delta', 'gamma', 
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
                    print(ce_sample[['STRIKE_PR', 'CLOSE', 'OPEN_INT', 'VOLUME']])
    
    # Example 2: Options Calculator Usage
    print("\n" + "=" * 50)
    print("3. Options Calculator Usage")
    print("=" * 50)
    
    calculator = OptionsCalculator(risk_free_rate=0.05)
    
    # Example parameters
    S = 100.0  # Stock price
    K = 100.0  # Strike price
    T = 0.25   # 3 months to expiry
    sigma = 0.3  # 30% volatility
    
    print(f"\n🧮 Calculating option metrics:")
    print(f"   Stock Price: {S}")
    print(f"   Strike Price: {K}")
    print(f"   Time to Expiry: {T} years")
    print(f"   Volatility: {sigma}")
    
    # Calculate call option
    call_price = calculator.black_scholes_call(S, K, T, 0.05, sigma)
    call_greeks = calculator.calculate_all_greeks(S, K, T, 0.05, sigma, 'CE')
    
    print(f"\n📈 Call Option:")
    print(f"   Price: {call_price:.4f}")
    print(f"   Delta: {call_greeks['delta']:.4f}")
    print(f"   Gamma: {call_greeks['gamma']:.4f}")
    print(f"   Theta: {call_greeks['theta']:.4f}")
    print(f"   Vega: {call_greeks['vega']:.4f}")
    print(f"   Rho: {call_greeks['rho']:.4f}")
    
    # Calculate put option
    put_price = calculator.black_scholes_put(S, K, T, 0.05, sigma)
    put_greeks = calculator.calculate_all_greeks(S, K, T, 0.05, sigma, 'PE')
    
    print(f"\n📉 Put Option:")
    print(f"   Price: {put_price:.4f}")
    print(f"   Delta: {put_greeks['delta']:.4f}")
    print(f"   Gamma: {put_greeks['gamma']:.4f}")
    print(f"   Theta: {put_greeks['theta']:.4f}")
    print(f"   Vega: {put_greeks['vega']:.4f}")
    print(f"   Rho: {put_greeks['rho']:.4f}")
    
    # Test IV calculation
    calculated_iv = calculator.calculate_implied_volatility(call_price, S, K, T, 0.05, 'CE')
    print(f"\n🔄 Implied Volatility (from call price): {calculated_iv:.6f}")
    
    # Example 3: Strategy Backtesting
    print("\n" + "=" * 50)
    print("4. Strategy Backtesting")
    print("=" * 50)
    
    print(f"\n🚀 Running strategy backtests for {test_symbol}...")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial capital: $100,000")
    
    # Test Iron Condor Strategy
    print(f"\n📊 Testing Iron Condor Strategy...")
    iron_condor = IronCondorStrategy(loader, delta_threshold=0.15, days_to_expiry=30)
    
    try:
        iron_results = iron_condor.backtest(test_symbol, start_date, end_date, 100000)
        
        if iron_results:
            print(f"\n✅ Iron Condor Results:")
            print(f"   Final Value: ${iron_results['final_value']:,.2f}")
            print(f"   Total Return: {iron_results['total_return']:.2%}")
            print(f"   Annualized Return: {iron_results['annualized_return']:.2%}")
            print(f"   Volatility: {iron_results['volatility']:.2%}")
            print(f"   Sharpe Ratio: {iron_results['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {iron_results['max_drawdown']:.2%}")
            print(f"   Total Trades: {iron_results['total_trades']}")
        else:
            print("❌ Iron Condor backtest failed")
    
    except Exception as e:
        print(f"❌ Error in Iron Condor backtest: {e}")
    
    # Test Straddle Strategy
    print(f"\n📊 Testing Straddle Strategy...")
    straddle = StraddleStrategy(loader, iv_percentile_threshold=20.0, days_to_expiry=30)
    
    try:
        straddle_results = straddle.backtest(test_symbol, start_date, end_date, 100000)
        
        if straddle_results:
            print(f"\n✅ Straddle Results:")
            print(f"   Final Value: ${straddle_results['final_value']:,.2f}")
            print(f"   Total Return: {straddle_results['total_return']:.2%}")
            print(f"   Annualized Return: {straddle_results['annualized_return']:.2%}")
            print(f"   Volatility: {straddle_results['volatility']:.2%}")
            print(f"   Sharpe Ratio: {straddle_results['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {straddle_results['max_drawdown']:.2%}")
            print(f"   Total Trades: {straddle_results['total_trades']}")
        else:
            print("❌ Straddle backtest failed")
    
    except Exception as e:
        print(f"❌ Error in Straddle backtest: {e}")
    
    # Example 4: Custom Analysis
    print("\n" + "=" * 50)
    print("5. Custom Analysis")
    print("=" * 50)
    
    if options_data is not None and stock_price is not None:
        print(f"\n🔍 Analyzing options data for {test_symbol}...")
        
        # Filter for options only - use OPTIDX for options
        options_df = options_data[options_data['INSTRUMENT'] == 'OPTIDX'].copy()
        
        if not options_df.empty:
            # Basic statistics
            print(f"\n📊 Basic Statistics:")
            print(f"   Total options: {len(options_df)}")
            print(f"   CE options: {len(options_df[options_df['OPTION_TYP'] == 'CE'])}")
            print(f"   PE options: {len(options_df[options_df['OPTION_TYP'] == 'PE'])}")
            print(f"   Unique expiries: {options_df['EXPIRY_DT'].nunique()}")
            print(f"   Unique strikes: {options_df['STRIKE_PR'].nunique()}")
            
            # Strike price analysis
            print(f"\n🎯 Strike Price Analysis:")
            print(f"   Min strike: {options_df['STRIKE_PR'].min():.2f}")
            print(f"   Max strike: {options_df['STRIKE_PR'].max():.2f}")
            print(f"   Stock price: {stock_price:.2f}")
            
            # Volume analysis
            if 'VOLUME' in options_df.columns:
                total_volume = options_df['VOLUME'].sum()
                avg_volume = options_df['VOLUME'].mean()
                print(f"\n📈 Volume Analysis:")
                print(f"   Total volume: {total_volume:,.0f}")
                print(f"   Average volume per option: {avg_volume:.0f}")
                
                # Most liquid options
                most_liquid = options_df.nlargest(5, 'VOLUME')
                print(f"\n🔥 Most Liquid Options:")
                for _, row in most_liquid.iterrows():
                    print(f"   {row['OPTION_TYP']} {row['STRIKE_PR']:.0f}: {row['VOLUME']:,.0f} contracts")
    
    print("\n" + "=" * 50)
    print("🎉 Example Usage Completed!")
    print("=" * 50)
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Explore different symbols and time periods")
    print(f"   2. Create custom strategies by inheriting from OptionsStrategy")
    print(f"   3. Add transaction costs and liquidity filters")
    print(f"   4. Implement more sophisticated risk management")
    print(f"   5. Add visualization and reporting capabilities")

if __name__ == "__main__":
    main() 