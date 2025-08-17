#!/usr/bin/env python3
"""
Data Loader for Options Backtesting
Loads and processes options and stock data from local market_data directory
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import glob
from pathlib import Path

class OptionsDataLoader:
    """
    Load and process options and stock data for backtesting
    """
    
    def __init__(self, data_dir: str = "../market_data"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to market_data directory. Defaults to "market_data" directory relative to project root.
        """
        self.data_dir = Path(data_dir)
        self.options_dir = self.data_dir / "options_data"
        self.stock_dir = self.data_dir / "stock_data"
        # On-disk cache for processed options (Greeks/IV)
        self.greeks_cache_dir = self.data_dir / "greeks_cache"
        
        # Cache for loaded data
        self._options_cache = {}
        self._stock_cache = {}
        # Track missing file warnings to avoid noisy repeats (disabled)
        # self._missing_warned: set[str] = set()

    def get_available_stock_dates(self, symbol: str) -> List[str]:
        """
        Get list of available stock dates for a symbol by scanning daily files.
        Returns sorted list of YYYY-MM-DD dates.
        """
        symbol_dir = self.stock_dir / symbol
        if not symbol_dir.exists():
            return []
        dates: List[str] = []
        for p in symbol_dir.glob("*.csv"):
            name = p.stem
            # skip complete file
            if name.endswith("_complete"):
                continue
            if name.startswith(f"{symbol}_"):
                datestr = name.replace(f"{symbol}_", "")
                dates.append(datestr)
        return sorted(dates)

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the data
        
        Returns:
            List of available symbols
        """
        if not self.options_dir.exists():
            return []
        
        symbols = [d.name for d in self.options_dir.iterdir() if d.is_dir()]
        return sorted(symbols)
    
    def get_available_dates(self, symbol: str) -> List[str]:
        """
        Get list of available dates for a symbol
        
        Args:
            symbol: Symbol name
        
        Returns:
            List of available dates in YYYY-MM-DD format
        """
        symbol_dir = self.options_dir / symbol
        if not symbol_dir.exists():
            return []
        
        # Get all CSV files
        csv_files = list(symbol_dir.glob("*.csv"))
        
        # Extract dates from filenames
        dates = []
        for file in csv_files:
            # Extract date from filename like "nifty_options_2025-08-06.csv"
            filename = file.stem
            if "_options_" in filename:
                date_part = filename.split("_options_")[-1]
                dates.append(date_part)
        
        return sorted(dates)

    # ------- Processed options (Greeks/IV) on-disk cache -------
    def _greeks_cache_path(self, symbol: str, date: str) -> Path:
        sym_dir = self.greeks_cache_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}_greeks_{date}.csv"
        return sym_dir / filename

    def load_processed_options_cache(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """
        Load cached processed options (with Greeks/IV) if present.
        Returns DataFrame or None if not found/could not be loaded.
        """
        path = self._greeks_cache_path(symbol, date)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            # basic sanity check
            if 'STRIKE_PR' in df.columns and 'OPTION_TYP' in df.columns:
                return df
            return None
        except Exception:
            return None

    def save_processed_options_cache(self, symbol: str, date: str, df: pd.DataFrame) -> None:
        """
        Save processed options (with Greeks/IV) for reuse in follow-up runs.
        """
        try:
            path = self._greeks_cache_path(symbol, date)
            df.to_csv(path, index=False)
        except Exception:
            pass
    
    def load_options_data(self, symbol: str, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load options data for a specific symbol and date
        
        Args:
            symbol: Symbol name
            date: Date in YYYY-MM-DD format
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with options data or None if not found
        """
        cache_key = f"{symbol}_{date}"
        
        if use_cache and cache_key in self._options_cache:
            return self._options_cache[cache_key]
        
        # Construct file path
        filename = f"{symbol}_options_{date}.csv"
        filepath = self.options_dir / symbol / filename
        
        if not filepath.exists():
            # key = f"opt:{filepath}"
            # if key not in self._missing_warned:
            #     print(f"⚠️  Options data file not found: {filepath}")
            #     self._missing_warned.add(key)
            return None
        
        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            
            # Basic data cleaning
            df = self._clean_options_data(df)
            
            # Cache the data
            if use_cache:
                self._options_cache[cache_key] = df
            
            #print(f"✅ Loaded options data for {symbol} on {date}: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading options data for {symbol} on {date}: {e}")
            return None
    
    def load_stock_data(self, symbol: str, date: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load stock data for a specific symbol and date
        
        Args:
            symbol: Symbol name
            date: Date in YYYY-MM-DD format
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with stock data or None if not found
        """
        cache_key = f"{symbol}_{date}"
        
        if use_cache and cache_key in self._stock_cache:
            return self._stock_cache[cache_key]
        
        # Try individual date file first
        filename = f"{symbol}_{date}.csv"
        filepath = self.stock_dir / symbol / filename
        
        if not filepath.exists():
            # Try complete file
            complete_filename = f"{symbol}_complete.csv"
            complete_filepath = self.stock_dir / symbol / complete_filename
            
            if complete_filepath.exists():
                try:
                    # Load complete file and filter for specific date
                    complete_df = pd.read_csv(complete_filepath)
                    complete_df['Date'] = pd.to_datetime(complete_df['Date'])
                    
                    # Filter for the specific date
                    target_date = pd.to_datetime(date)
                    df = complete_df[complete_df['Date'].dt.date == target_date.date()]
                    
                    if not df.empty:
                        if use_cache:
                            self._stock_cache[cache_key] = df
                        print(f"✅ Loaded stock data for {symbol} on {date} from complete file")
                        return df
                        
                except Exception as e:
                    print(f"❌ Error loading stock data from complete file: {e}")
            
            # key = f"stk:{filepath}"
            # if key not in self._missing_warned:
            #     print(f"⚠️  Stock data file not found: {filepath}")
            #     self._missing_warned.add(key)
            return None
        
        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            
            # Basic data cleaning
            df = self._clean_stock_data(df)
            
            # Cache the data
            if use_cache:
                self._stock_cache[cache_key] = df
            
            #print(f"✅ Loaded stock data for {symbol} on {date}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading stock data for {symbol} on {date}: {e}")
            return None
    
    def load_historical_data(self, symbol: str, start_date: str, end_date: str, 
                           data_type: str = "both") -> Dict[str, pd.DataFrame]:
        """
        Load historical data for a symbol over a date range
        
        Args:
            symbol: Symbol name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            data_type: Type of data to load ('options', 'stock', or 'both')
            
        Returns:
            Dictionary with loaded data
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        options_data = []
        stock_data = []
        
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Load options data
            if data_type in ['options', 'both']:
                options_df = self.load_options_data(symbol, date_str)
                if options_df is not None:
                    options_df['date'] = date_str
                    options_data.append(options_df)
            
            # Load stock data
            if data_type in ['stock', 'both']:
                stock_df = self.load_stock_data(symbol, date_str)
                if stock_df is not None:
                    stock_df['date'] = date_str
                    stock_data.append(stock_df)
            
            current_dt += timedelta(days=1)
        
        result = {}
        
        if options_data:
            result['options'] = pd.concat(options_data, ignore_index=True)
            print(f"✅ Loaded {len(result['options'])} options records for {symbol}")
        
        if stock_data:
            result['stock'] = pd.concat(stock_data, ignore_index=True)
            print(f"✅ Loaded {len(result['stock'])} stock records for {symbol}")
        
        return result
    
    def get_stock_price(self, symbol: str, date: str) -> Optional[float]:
        """
        Get stock price for a specific date
        
        Args:
            symbol: Symbol name
            date: Date in YYYY-MM-DD format
            
        Returns:
            Stock price (close price) or None if not found
        """
        stock_df = self.load_stock_data(symbol, date)
        
        if stock_df is None or stock_df.empty:
            return None
        
        # Return close price
        return float(stock_df['Close'].iloc[0])
    
    def get_atm_options(self, symbol: str, date: str, tolerance: float = 0.02) -> Dict[str, pd.DataFrame]:
        """
        Get at-the-money options for a symbol and date
        
        Args:
            symbol: Symbol name
            date: Date in YYYY-MM-DD format
            tolerance: Tolerance for ATM definition (default: 2%)
            
        Returns:
            Dictionary with 'CE' and 'PE' DataFrames for ATM options
        """
        # Get stock price
        stock_price = self.get_stock_price(symbol, date)
        if stock_price is None:
            return {}
        
        # Load options data
        options_df = self.load_options_data(symbol, date)
        if options_df is None or options_df.empty:
            return {}
        
        # Filter for options only
        options_df = options_df[options_df['INSTRUMENT'] == 'OPTIDX'].copy()
        
        # Calculate moneyness
        options_df['moneyness'] = stock_price / options_df['STRIKE_PR']
        
        # Find ATM options (moneyness close to 1.0)
        atm_mask = (options_df['moneyness'] >= (1 - tolerance)) & (options_df['moneyness'] <= (1 + tolerance))
        atm_options = options_df[atm_mask].copy()
        
        # Separate by option type
        ce_options = atm_options[atm_options['OPTION_TYP'] == 'CE'].copy()
        pe_options = atm_options[atm_options['OPTION_TYP'] == 'PE'].copy()
        
        # Sort by strike price
        ce_options = ce_options.sort_values('STRIKE_PR')
        pe_options = pe_options.sort_values('STRIKE_PR')
        
        return {
            'CE': ce_options,
            'PE': pe_options,
            'stock_price': stock_price
        }
    
    def get_option_chain(self, symbol: str, date: str, expiry_date: str = None) -> pd.DataFrame:
        """
        Get option chain for a specific expiry
        
        Args:
            symbol: Symbol name
            date: Date in YYYY-MM-DD format
            expiry_date: Expiry date (if None, gets all expiries)
            
        Returns:
            DataFrame with option chain
        """
        options_df = self.load_options_data(symbol, date)
        if options_df is None or options_df.empty:
            return pd.DataFrame()
        
        # Filter for options only
        options_df = options_df[options_df['INSTRUMENT'] == 'OPTIDX'].copy()
        
        # Filter by expiry if specified
        if expiry_date:
            options_df = options_df[options_df['EXPIRY_DT'] == expiry_date].copy()
        
        # Sort by strike price and option type
        options_df = options_df.sort_values(['STRIKE_PR', 'OPTION_TYP'])
        
        return options_df
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize options data
        
        Args:
            df: Raw options DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert numeric columns
        numeric_columns = ['STRIKE_PR', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SETTLE_PR', 
                          'CONTRACTS', 'VAL_INLAKH', 'OPEN_INT', 'CHG_IN_OI']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out rows with invalid data
        df = df.dropna(subset=['STRIKE_PR', 'CLOSE'])
        
        return df
    
    def _clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize stock data
        
        Args:
            df: Raw stock DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def clear_cache(self):
        """Clear all cached data"""
        self._options_cache.clear()
        self._stock_cache.clear()
        print("🗑️  Cleared data cache")

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = OptionsDataLoader()
    
    print("=== Options Data Loader Test ===")
    
    # Get available symbols
    symbols = loader.get_available_symbols()
    print(f"Available symbols: {symbols[:10]}...")  # Show first 10
    
    if symbols:
        # Test with first symbol
        test_symbol = symbols[0]
        print(f"\nTesting with symbol: {test_symbol}")
        
        # Get available dates
        dates = loader.get_available_dates(test_symbol)
        print(f"Available dates: {len(dates)} dates")
        
        if dates:
            # Test with first date
            test_date = dates[0]
            print(f"\nTesting with date: {test_date}")
            
            # Load options data
            options_df = loader.load_options_data(test_symbol, test_date)
            if options_df is not None:
                print(f"Options data shape: {options_df.shape}")
                print(f"Columns: {list(options_df.columns)}")
                print(f"Sample data:")
                print(options_df.head(3))
            
            # Load stock data
            stock_df = loader.load_stock_data(test_symbol, test_date)
            if stock_df is not None:
                print(f"\nStock data shape: {stock_df.shape}")
                print(f"Stock price: {loader.get_stock_price(test_symbol, test_date)}")
            
            # Get ATM options
            atm_options = loader.get_atm_options(test_symbol, test_date)
            if atm_options:
                print(f"\nATM Options:")
                print(f"Stock price: {atm_options['stock_price']}")
                print(f"CE options: {len(atm_options['CE'])}")
                print(f"PE options: {len(atm_options['PE'])}")
                
                if not atm_options['CE'].empty:
                    print(f"Sample CE option:")
                    print(atm_options['CE'].head(1))
    
    print("\n🎉 Data loader test completed!") 