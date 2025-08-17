#!/usr/bin/env python3
"""
Data Query Script - Query local stock and options data cache
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass

# Import stock definitions
from stocks import StockName, STOCK_NAME_MAP

@dataclass
class DataAvailability:
    """Data availability information for a symbol"""
    symbol: str
    stock_dates: List[str]
    options_dates: List[str]
    stock_start_date: Optional[str]
    stock_end_date: Optional[str]
    options_start_date: Optional[str]
    options_end_date: Optional[str]

class DataQuery:
    def __init__(self, base_dir: str = "market_data"):
        self.base_dir = base_dir
        self.stock_data_dir = os.path.join(base_dir, "stock_data")
        self.options_data_dir = os.path.join(base_dir, "options_data")
        self.history_file = os.path.join(base_dir, "download_history.json")
        
        # Load download history
        self.download_history = self._load_download_history()
    
    def _load_download_history(self) -> Dict:
        """Load download history from JSON file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        else:
            return {"stock_data": {}, "options_data": {}, "last_update": None}
    
    def get_available_symbols(self) -> List[str]:
        """Get list of all symbols that have any data (stock or options)"""
        symbols = set()
        
        # Get symbols from stock data
        if os.path.exists(self.stock_data_dir):
            for item in os.listdir(self.stock_data_dir):
                item_path = os.path.join(self.stock_data_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory has any CSV files
                    csv_files = [f for f in os.listdir(item_path) if f.endswith('.csv') and not f.endswith('_complete.csv')]
                    if csv_files:
                        symbols.add(item)
        
        # Get symbols from options data
        if os.path.exists(self.options_data_dir):
            for item in os.listdir(self.options_data_dir):
                item_path = os.path.join(self.options_data_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory has any CSV files
                    csv_files = [f for f in os.listdir(item_path) if f.endswith('.csv') and not f.endswith('_complete.csv')]
                    if csv_files:
                        symbols.add(item)
        
        return sorted(list(symbols))
    
    def get_data_availability(self, symbol: str) -> DataAvailability:
        """Get detailed data availability for a specific symbol"""
        stock_dates = []
        options_dates = []
        
        # Get stock data dates
        stock_dir = os.path.join(self.stock_data_dir, symbol)
        if os.path.exists(stock_dir):
            for filename in os.listdir(stock_dir):
                if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                    try:
                        # Extract date from filename like "nifty_2023-01-01.csv"
                        date_part = filename.replace(f"{symbol}_", "").replace(".csv", "")
                        stock_dates.append(date_part)
                    except:
                        continue
        
        # Get options data dates
        options_dir = os.path.join(self.options_data_dir, symbol)
        if os.path.exists(options_dir):
            for filename in os.listdir(options_dir):
                if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                    try:
                        # Extract date from filename like "nifty_options_2023-01-01.csv"
                        date_part = filename.replace(f"{symbol}_options_", "").replace(".csv", "")
                        options_dates.append(date_part)
                    except:
                        continue
        
        # Sort dates
        stock_dates.sort()
        options_dates.sort()
        
        return DataAvailability(
            symbol=symbol,
            stock_dates=stock_dates,
            options_dates=options_dates,
            stock_start_date=stock_dates[0] if stock_dates else None,
            stock_end_date=stock_dates[-1] if stock_dates else None,
            options_start_date=options_dates[0] if options_dates else None,
            options_end_date=options_dates[-1] if options_dates else None
        )
    
    def get_all_data_availability(self) -> Dict[str, DataAvailability]:
        """Get data availability for all symbols"""
        symbols = self.get_available_symbols()
        availability = {}
        
        for symbol in symbols:
            availability[symbol] = self.get_data_availability(symbol)
        
        return availability
    
    def get_stock_data(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """Get stock data for a specific symbol and date"""
        try:
            # Convert date string to datetime for validation
            datetime.strptime(date, '%Y-%m-%d')
            
            # Construct file path
            filename = f"{symbol}_{date}.csv"
            filepath = os.path.join(self.stock_data_dir, symbol, filename)
            
            if not os.path.exists(filepath):
                print(f"❌ Stock data file not found: {filepath}")
                return None
            
            # Read the CSV file
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✅ Loaded stock data for {symbol} on {date}: {len(df)} records")
            return df
            
        except ValueError:
            print(f"❌ Invalid date format. Use YYYY-MM-DD format.")
            return None
        except Exception as e:
            print(f"❌ Error loading stock data for {symbol} on {date}: {e}")
            return None
    
    def get_options_data(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """Get options data for a specific symbol and date"""
        try:
            # Convert date string to datetime for validation
            datetime.strptime(date, '%Y-%m-%d')
            
            # Construct file path
            filename = f"{symbol}_options_{date}.csv"
            filepath = os.path.join(self.options_data_dir, symbol, filename)
            
            if not os.path.exists(filepath):
                print(f"❌ Options data file not found: {filepath}")
                return None
            
            # Read the CSV file
            df = pd.read_csv(filepath)
            #print(f"✅ Loaded options data for {symbol} on {date}: {len(df)} records")
            return df
            
        except ValueError:
            print(f"❌ Invalid date format. Use YYYY-MM-DD format.")
            return None
        except Exception as e:
            print(f"❌ Error loading options data for {symbol} on {date}: {e}")
            return None
    
    def get_both_data(self, symbol: str, date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get both stock and options data for a specific symbol and date"""
        stock_data = self.get_stock_data(symbol, date)
        options_data = self.get_options_data(symbol, date)
        return stock_data, options_data
    
    def print_summary(self):
        """Print a summary of all available data"""
        print("📊 DATA AVAILABILITY SUMMARY")
        print("=" * 50)
        
        availability = self.get_all_data_availability()
        
        if not availability:
            print("❌ No data found in cache")
            return
        
        print(f"📈 Total symbols with data: {len(availability)}")
        print()
        
        for symbol, data in availability.items():
            print(f"🔸 {symbol.upper()}")
            
            if data.stock_dates:
                print(f"   📊 Stock: {len(data.stock_dates)} dates ({data.stock_start_date} to {data.stock_end_date})")
            else:
                print(f"   📊 Stock: No data")
            
            if data.options_dates:
                print(f"   📈 Options: {len(data.options_dates)} dates ({data.options_start_date} to {data.options_end_date})")
            else:
                print(f"   📈 Options: No data")
            
            print()

def main():
    """Example usage of the DataQuery class"""
    print("🔍 DATA QUERY SCRIPT - EXAMPLE USAGE")
    print("=" * 50)
    
    # Initialize the data query
    query = DataQuery()
    
    # Example 1: Print summary of all available data
    print("1️⃣ PRINTING DATA SUMMARY")
    print("-" * 30)
    query.print_summary()
    
    # Example 2: Get list of available symbols
    print("2️⃣ GETTING AVAILABLE SYMBOLS")
    print("-" * 30)
    symbols = query.get_available_symbols()
    print(f"Available symbols: {symbols}")
    print()
    
    # Example 3: Get detailed availability for a specific symbol
    if symbols:
        print("3️⃣ DETAILED AVAILABILITY FOR FIRST SYMBOL")
        print("-" * 30)
        first_symbol = symbols[0]
        availability = query.get_data_availability(first_symbol)
        
        print(f"Symbol: {availability.symbol}")
        print(f"Stock dates: {len(availability.stock_dates)}")
        print(f"Options dates: {len(availability.options_dates)}")
        
        if availability.stock_dates:
            print(f"Stock date range: {availability.stock_start_date} to {availability.stock_end_date}")
        if availability.options_dates:
            print(f"Options date range: {availability.options_start_date} to {availability.options_end_date}")
        print()
    
    # Example 4: Get data for a specific date and symbol
    if symbols:
        print("4️⃣ GETTING DATA FOR SPECIFIC DATE")
        print("-" * 30)
        
        # Get the first symbol and a date it has data for
        symbol = symbols[0]
        availability = query.get_data_availability(symbol)
        
        # Try to get stock data
        if availability.stock_dates:
            test_date = availability.stock_dates[0]
            print(f"Getting stock data for {symbol} on {test_date}")
            stock_data = query.get_stock_data(symbol, test_date)
            
            if stock_data is not None:
                print(f"Stock data columns: {list(stock_data.columns)}")
                print(f"Stock data shape: {stock_data.shape}")
                print("First few rows:")
                print(stock_data.head(3))
            print()
        
        # Try to get options data
        if availability.options_dates:
            test_date = availability.options_dates[0]
            print(f"Getting options data for {symbol} on {test_date}")
            options_data = query.get_options_data(symbol, test_date)
            
            if options_data is not None:
                print(f"Options data columns: {list(options_data.columns)}")
                print(f"Options data shape: {options_data.shape}")
                print("First few rows:")
                print(options_data.head(3))
            print()
    
    # Example 5: Get both stock and options data together
    if symbols:
        print("5️⃣ GETTING BOTH STOCK AND OPTIONS DATA")
        print("-" * 30)
        
        symbol = symbols[0]
        availability = query.get_data_availability(symbol)
        
        # Find a date that has both stock and options data
        common_dates = set(availability.stock_dates) & set(availability.options_dates)
        
        if common_dates:
            test_date = sorted(common_dates)[0]
            print(f"Getting both data types for {symbol} on {test_date}")
            
            stock_data, options_data = query.get_both_data(symbol, test_date)
            
            if stock_data is not None:
                print(f"✅ Stock data: {stock_data.shape}")
            if options_data is not None:
                print(f"✅ Options data: {options_data.shape}")
        else:
            print(f"No common dates found for {symbol} with both stock and options data")
    
    print("\n🎉 Example usage completed!")

if __name__ == "__main__":
    main() 