import os
import requests
import zipfile
import io
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import random

# Import stock definitions
from stocks import StockName, STOCK_NAME_MAP

# Import jugaad options downloader
try:
    from jugaad_options_downloader import JugaadOptionsDownloader
    HAS_JUGAAD_DOWNLOADER = True
except ImportError:
    HAS_JUGAAD_DOWNLOADER = False
    print("❌ jugaad-data library is required for options data")
    print("💡 Install with: pip install jugaad-data")

@dataclass
class DownloadConfig:
    """Configuration for data download"""
    base_dir: str = "market_data"
    stock_data_dir: str = "stock_data"
    options_data_dir: str = "options_data"
    start_date: datetime = None
    end_date: datetime = None
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now() - timedelta(days=6*365)  # 5 years ago
        if self.end_date is None:
            self.end_date = datetime.now()

class DataDownloader:
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.setup_directories()
        self.load_download_history()
        
        # Sync download history with actual files on disk
        self.sync_download_history_with_disk()
        
        self.session = requests.Session()
        # Set headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Initialize jugaad options downloader (required)
        if HAS_JUGAAD_DOWNLOADER:
            self.jugaad_downloader = JugaadOptionsDownloader()
        else:
            raise ImportError("jugaad-data library is required for options data. Install with: pip install jugaad-data")
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config.base_dir,
            os.path.join(self.config.base_dir, self.config.stock_data_dir),
            os.path.join(self.config.base_dir, self.config.options_data_dir),
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        # Create subdirectories for each stock
        for stock_name in StockName:
            stock_info = STOCK_NAME_MAP[stock_name]
            stock_dir = os.path.join(self.config.base_dir, self.config.stock_data_dir, stock_name.value)
            options_dir = os.path.join(self.config.base_dir, self.config.options_data_dir, stock_name.value)
            os.makedirs(stock_dir, exist_ok=True)
            os.makedirs(options_dir, exist_ok=True)
    
    def sync_download_history_with_disk(self):
        """Sync download history JSON with actual files on disk"""
        print("🔄 Syncing download history with disk files...")
        
        # Initialize counters
        stock_files_found = 0
        options_files_found = 0
        
        # Sync stock data
        print("📈 Syncing stock data history...")
        for stock_name in StockName:
            stock_dir = os.path.join(self.config.base_dir, self.config.stock_data_dir, stock_name.value)
            
            if stock_name.value not in self.download_history["stock_data"]:
                self.download_history["stock_data"][stock_name.value] = []
            
            if os.path.exists(stock_dir):
                for filename in os.listdir(stock_dir):
                    if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                        try:
                            # Extract date from filename like "nifty_2023-01-01.csv"
                            date_part = filename.replace(f"{stock_name.value}_", "").replace(".csv", "")
                            if date_part not in self.download_history["stock_data"][stock_name.value]:
                                self.download_history["stock_data"][stock_name.value].append(date_part)
                                stock_files_found += 1
                        except:
                            continue
        
        # Sync options data
        print("📊 Syncing options data history...")
        for stock_name in StockName:
            options_dir = os.path.join(self.config.base_dir, self.config.options_data_dir, stock_name.value)
            
            if stock_name.value not in self.download_history["options_data"]:
                self.download_history["options_data"][stock_name.value] = []
            
            if os.path.exists(options_dir):
                for filename in os.listdir(options_dir):
                    if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                        try:
                            # Extract date from filename like "nifty_options_2023-01-01.csv"
                            date_part = filename.replace(f"{stock_name.value}_options_", "").replace(".csv", "")
                            if date_part not in self.download_history["options_data"][stock_name.value]:
                                self.download_history["options_data"][stock_name.value].append(date_part)
                                options_files_found += 1
                        except:
                            continue
        
        # Save the updated history
        self.save_download_history()
        
        print(f"✅ Sync completed: {stock_files_found} stock files and {options_files_found} options files added to history")
        
        # Show summary
        total_stock_dates = sum(len(dates) for dates in self.download_history["stock_data"].values())
        total_options_dates = sum(len(dates) for dates in self.download_history["options_data"].values())
        print(f"📊 Total dates in history: {total_stock_dates} stock dates, {total_options_dates} options dates")

    def load_download_history(self):
        """Load download history to track what's already downloaded"""
        history_file = os.path.join(self.config.base_dir, "download_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.download_history = json.load(f)
        else:
            self.download_history = {
                "stock_data": {},
                "options_data": {},
                "last_update": None
            }
    
    def save_download_history(self):
        """Save download history"""
        history_file = os.path.join(self.config.base_dir, "download_history.json")
        self.download_history["last_update"] = datetime.now().isoformat()
        with open(history_file, 'w') as f:
            json.dump(self.download_history, f, indent=2)
    
    def get_stock_data_filename(self, stock_name: StockName, date: datetime) -> str:
        """Generate filename for stock data"""
        return f"{stock_name.value}_{date.strftime('%Y-%m-%d')}.csv"
    
    def get_options_data_filename(self, stock_name: StockName, date: datetime) -> str:
        """Generate filename for options data"""
        return f"{stock_name.value}_options_{date.strftime('%Y-%m-%d')}.csv"
    
    def get_yfinance_symbol(self, stock_name: StockName) -> str:
        """Get the correct yfinance symbol for a stock"""
        stock_info = STOCK_NAME_MAP[stock_name]
        
        # Handle special cases
        if stock_name == StockName.NIFTY:
            return "^NSEI"  # NIFTY 50 index
        elif stock_name == StockName.BANK_NIFTY:
            return "^NSEBANK"  # Bank NIFTY index
        elif stock_name == StockName.FIN_NIFTY:
            return "^NSEBANK"  # Use Bank NIFTY as proxy for FIN NIFTY
        elif stock_name == StockName.SENSEX:
            return "^BSESN"  # SENSEX index
        else:
            # For individual stocks, use the equity symbol with .NS suffix
            return f"{stock_info.eq}.NS"
    
    def download_stock_data(self, stock_name: StockName, start_date: datetime, end_date: datetime) -> bool:
        """Download stock data using yfinance"""
        try:
            symbol = self.get_yfinance_symbol(stock_name)
            ticker = yf.Ticker(symbol)
            
            # Check what data we already have
            stock_dir = os.path.join(self.config.base_dir, self.config.stock_data_dir, stock_name.value)
            existing_dates = set()
            
            # Check existing daily files
            if os.path.exists(stock_dir):
                for filename in os.listdir(stock_dir):
                    if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                        # Extract date from filename like "nifty_2023-01-01.csv"
                        try:
                            date_part = filename.replace(f"{stock_name.value}_", "").replace(".csv", "")
                            existing_dates.add(date_part)
                        except:
                            continue
            
            # Determine what dates we need to download (skip weekends)
            current_date = start_date
            missing_dates = []
            holiday_dates = []
            
            while current_date <= end_date:
                if self.is_trading_day(current_date):  # Check if it's a trading day
                    date_str = current_date.strftime('%Y-%m-%d')
                    if date_str not in existing_dates:
                        missing_dates.append(current_date)
                else:
                    # Track holidays for debugging
                    holiday_dates.append(current_date)
                current_date += timedelta(days=1)
            
            if not missing_dates:
                print(f"✅ Stock data already exists for {stock_name.value} ({symbol}) in date range")
                return True
            
            # Show detailed breakdown of missing dates
            print(f"📥 Downloading stock data for {stock_name.value} ({symbol}) - {len(missing_dates)} missing dates")
            if len(missing_dates) <= 10:  # Show all missing dates if <= 10
                missing_date_strs = [d.strftime('%Y-%m-%d') for d in missing_dates]
                print(f"   Missing dates: {', '.join(missing_date_strs)}")
            else:
                # Show first and last few dates
                first_dates = [d.strftime('%Y-%m-%d') for d in missing_dates[:3]]
                last_dates = [d.strftime('%Y-%m-%d') for d in missing_dates[-3:]]
                print(f"   Missing dates: {', '.join(first_dates)} ... {', '.join(last_dates)}")
            
            # Show holiday count for debugging
            if holiday_dates:
                print(f"   Skipped {len(holiday_dates)} holidays/weekends in date range")
            
            # Download data for missing dates
            data = ticker.history(start=min(missing_dates), end=max(missing_dates) + timedelta(days=1), interval="1d")
            
            if data.empty:
                print(f"⚠️  No stock data available for {stock_name.value} ({symbol})")
                return False
            
            # Save data
            # Save daily files
            for date in data.index:
                date_str = date.strftime('%Y-%m-%d')
                daily_data = data.loc[date:date]
                filename = self.get_stock_data_filename(stock_name, date)
                filepath = os.path.join(stock_dir, filename)
                daily_data.to_csv(filepath)
                
                # Update history
                if stock_name.value not in self.download_history["stock_data"]:
                    self.download_history["stock_data"][stock_name.value] = []
                if date_str not in self.download_history["stock_data"][stock_name.value]:
                    self.download_history["stock_data"][stock_name.value].append(date_str)
            
            # Update complete dataset
            complete_file = os.path.join(stock_dir, f"{stock_name.value}_complete.csv")
            if os.path.exists(complete_file):
                # Load existing complete data and merge
                existing_data = pd.read_csv(complete_file, index_col=0, parse_dates=True)
                combined_data = pd.concat([existing_data, data]).drop_duplicates().sort_index()
            else:
                combined_data = data
            
            combined_data.to_csv(complete_file)
            
            print(f"✅ Downloaded stock data for {stock_name.value} ({symbol}): {len(data)} new records")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading stock data for {stock_name.value}: {e}")
            return False
    
    def download_options_data_with_retry(self, stock_name: StockName, date: datetime, max_retries: int = 3) -> bool:
        """Download options data using hanya jugaad-data"""
        if not HAS_JUGAAD_DOWNLOADER:
            print(f"❌ jugaad-data library is required for options data")
            return False
        
        for attempt in range(max_retries):
            try:
                result = self._download_options_data_jugaad(stock_name, date)
                if result:
                    return True
                else:
                    print(f"❌ Options data download failed for {stock_name.value}")
                    return False
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5 + random.uniform(1, 3)
                    print(f"⏳ Error for {stock_name.value} on {date.strftime('%Y-%m-%d')}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed to download options data for {stock_name.value} on {date.strftime('%Y-%m-%d')} after {max_retries} attempts")
                    return False
        return False
    
    def _download_options_data_jugaad(self, stock_name: StockName, date: datetime) -> bool:
        """Download options data using jugaad-data"""
        try:
            if self.jugaad_downloader is None:
                print(f"❌ jugaad-data downloader not available for {stock_name.value}")
                return False
            
            stock_info = STOCK_NAME_MAP[stock_name]
            symbol = stock_info.nfo
            
            # Check if options data already exists
            options_dir = os.path.join(self.config.base_dir, self.config.options_data_dir, stock_name.value)
            filename = self.get_options_data_filename(stock_name, date)
            filepath = os.path.join(options_dir, filename)
            
            if os.path.exists(filepath):
                print(f"✅ Options data already exists for {stock_name.value} on {date.strftime('%Y-%m-%d')}")
                # Update history if not already there
                date_str = date.strftime('%Y-%m-%d')
                if stock_name.value not in self.download_history["options_data"]:
                    self.download_history["options_data"][stock_name.value] = []
                if date_str not in self.download_history["options_data"][stock_name.value]:
                    self.download_history["options_data"][stock_name.value].append(date_str)
                return True
            
            # Check if bhavcopy is already cached for this date
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self.jugaad_downloader.bhavcopy_cache:
                print(f"📋 Bhavcopy already cached for {date_str}, checking if {symbol} exists...")
                bhavcopy_df = self.jugaad_downloader.bhavcopy_cache[date_str]
                
                # Check if the symbol exists in the bhavcopy data
                if 'SYMBOL' in bhavcopy_df.columns:
                    symbol_exists = symbol in bhavcopy_df['SYMBOL'].values
                    if not symbol_exists:
                        print(f"⚠️  Symbol {symbol} not found in bhavcopy for {date_str} - contract didn't exist")
                        # Mark as processed to avoid re-downloading
                        if stock_name.value not in self.download_history["options_data"]:
                            self.download_history["options_data"][stock_name.value] = []
                        if date_str not in self.download_history["options_data"][stock_name.value]:
                            self.download_history["options_data"][stock_name.value].append(date_str)
                        return True
            
            print(f"📥 Downloading options data for {stock_name.value} ({symbol})...")
            
            # Try to get options data using jugaad-data
            df = self.jugaad_downloader.download_options_data(symbol, date)
            
            if df is not None and not df.empty:
                # Save the data
                df.to_csv(filepath, index=False)
                
                # Update history
                if stock_name.value not in self.download_history["options_data"]:
                    self.download_history["options_data"][stock_name.value] = []
                if date_str not in self.download_history["options_data"][stock_name.value]:
                    self.download_history["options_data"][stock_name.value].append(date_str)
                
                print(f"✅ Downloaded options data for {stock_name.value} on {date.strftime('%Y-%m-%d')}: {len(df)} records")
                return True
            else:
                print(f"❌ No options data available for {stock_name.value}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading options data for {stock_name.value}: {e}")
            return False
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if a date is a trading day (not weekend and not a major holiday)"""
        # Skip weekends
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Comprehensive Indian market holidays from 2019 to 2025
        # Based on official NSE/BSE trading calendars
        major_holidays = [
            # 2019
            "2019-01-26",  # Republic Day
            "2019-03-04",  # Holi
            "2019-04-17",  # Mahavir Jayanti
            "2019-04-19",  # Good Friday
            "2019-05-01",  # Maharashtra Day
            "2019-06-05",  # Ramzan Id
            "2019-08-12",  # Bakri Id
            "2019-08-15",  # Independence Day
            "2019-10-02",  # Mahatma Gandhi Jayanti
            "2019-10-08",  # Dussehra
            "2019-10-28",  # Diwali-Laxmi Pujan
            "2019-11-12",  # Gurunanak Jayanti
            "2019-12-25",  # Christmas
            
            # 2020
            "2020-01-26",  # Republic Day
            "2020-02-21",  # Mahashivratri
            "2020-03-10",  # Holi
            "2020-04-02",  # Ram Navami
            "2020-04-06",  # Mahavir Jayanti
            "2020-04-10",  # Good Friday
            "2020-05-01",  # Maharashtra Day
            "2020-05-25",  # Ramzan Id
            "2020-08-01",  # Bakri Id
            "2020-08-15",  # Independence Day
            "2020-10-02",  # Mahatma Gandhi Jayanti
            "2020-10-26",  # Diwali-Laxmi Pujan
            "2020-11-16",  # Gurunanak Jayanti
            "2020-11-30",  # Gurunanak Jayanti (Observed)
            "2020-12-25",  # Christmas
            
            # 2021
            "2021-01-26",  # Republic Day
            "2021-03-11",  # Mahashivratri
            "2021-03-29",  # Holi
            "2021-04-02",  # Good Friday
            "2021-04-14",  # Dr Ambedkar Jayanti
            "2021-04-21",  # Ram Navami
            "2021-04-25",  # Mahavir Jayanti
            "2021-05-13",  # Ramzan Id
            "2021-07-21",  # Bakri Id
            "2021-08-15",  # Independence Day
            "2021-10-02",  # Mahatma Gandhi Jayanti
            "2021-10-15",  # Dussehra
            "2021-11-04",  # Diwali-Laxmi Pujan
            "2021-11-19",  # Gurunanak Jayanti
            "2021-12-25",  # Christmas
            
            # 2022
            "2022-01-26",  # Republic Day
            "2022-03-01",  # Mahashivratri
            "2022-03-18",  # Holi
            "2022-04-14",  # Dr Ambedkar Jayanti
            "2022-04-15",  # Good Friday
            "2022-05-03",  # Ramzan Id
            "2022-07-11",  # Bakri Id
            "2022-08-15",  # Independence Day
            "2022-10-05",  # Mahatma Gandhi Jayanti
            "2022-10-24",  # Diwali-Laxmi Pujan
            "2022-11-08",  # Gurunanak Jayanti
            "2022-12-25",  # Christmas
            
            # 2023
            "2023-01-26",  # Republic Day
            "2023-02-18",  # Mahashivratri
            "2023-03-07",  # Holi
            "2023-03-30",  # Ram Navami
            "2023-04-04",  # Mahavir Jayanti
            "2023-04-07",  # Good Friday
            "2023-04-14",  # Dr Ambedkar Jayanti
            "2023-04-22",  # Ramzan Id
            "2023-06-29",  # Bakri Id
            "2023-08-15",  # Independence Day
            "2023-10-02",  # Mahatma Gandhi Jayanti
            "2023-10-24",  # Diwali-Laxmi Pujan
            "2023-11-14",  # Gurunanak Jayanti
            "2023-11-27",  # Gurunanak Jayanti (Observed)
            "2023-12-25",  # Christmas
            
            # 2024
            "2024-01-26",  # Republic Day
            "2024-02-09",  # Mahashivratri
            "2024-03-08",  # Holi
            "2024-03-25",  # Good Friday
            "2024-04-09",  # Ram Navami
            "2024-04-11",  # Id-Ul-Fitr (Ramzan Id)
            "2024-04-17",  # Mahavir Jayanti
            "2024-05-01",  # Maharashtra Day
            "2024-06-17",  # Bakri Id
            "2024-08-15",  # Independence Day
            "2024-10-02",  # Mahatma Gandhi Jayanti
            "2024-11-01",  # Diwali-Laxmi Pujan
            "2024-11-15",  # Gurunanak Jayanti
            "2024-12-25",  # Christmas
            
            # 2025
            "2025-01-26",  # Republic Day
            "2025-02-26",  # Mahashivratri
            "2025-03-26",  # Holi
            "2025-04-18",  # Good Friday
            "2025-04-29",  # Ram Navami
            "2025-03-31",  # Id-Ul-Fitr (Ramzan Id) - Tentative
            "2025-05-01",  # Maharashtra Day
            "2025-06-07",  # Bakri Id - Tentative
            "2025-08-15",  # Independence Day
            "2025-10-02",  # Mahatma Gandhi Jayanti
            "2025-10-21",  # Diwali-Laxmi Pujan
            "2025-11-05",  # Gurunanak Jayanti
            "2025-12-25",  # Christmas
        ]
        
        date_str = date.strftime('%Y-%m-%d')
        if date_str in major_holidays:
            return False
        
        return True

    def get_missing_dates(self, stock_name: StockName, data_type: str) -> List[datetime]:
        """Get dates that need to be downloaded"""
        current_date = self.config.start_date
        missing_dates = []
        
        while current_date <= self.config.end_date:
            if self.is_trading_day(current_date):  # Check if it's a trading day
                date_str = current_date.strftime('%Y-%m-%d')
                
                if data_type == "stock":
                    history_dates = self.download_history["stock_data"].get(stock_name.value, [])
                else:
                    history_dates = self.download_history["options_data"].get(stock_name.value, [])
                
                if date_str not in history_dates:
                    missing_dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return missing_dates
    
    def get_missing_dates_from_files(self, stock_name: StockName, data_type: str) -> List[datetime]:
        """Get dates that need to be downloaded by checking actual files (for non-incremental mode)"""
        current_date = self.config.start_date
        missing_dates = []
        
        # Get the directory to check
        if data_type == "stock":
            base_dir = os.path.join(self.config.base_dir, self.config.stock_data_dir, stock_name.value)
            filename_prefix = stock_name.value
        else:
            base_dir = os.path.join(self.config.base_dir, self.config.options_data_dir, stock_name.value)
            filename_prefix = f"{stock_name.value}_options"
        
        # Get existing files
        existing_dates = set()
        if os.path.exists(base_dir):
            for filename in os.listdir(base_dir):
                if filename.endswith('.csv') and not filename.endswith('_complete.csv'):
                    try:
                        # Extract date from filename like "nifty_2023-01-01.csv" or "nifty_options_2023-01-01.csv"
                        date_part = filename.replace(f"{filename_prefix}_", "").replace(".csv", "")
                        existing_dates.add(date_part)
                    except:
                        continue
        
        while current_date <= self.config.end_date:
            if self.is_trading_day(current_date):  # Check if it's a trading day
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    missing_dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return missing_dates
    
    def download_all_data(self, incremental: bool = True):
        """Download all data for all stocks"""
        print(f"🚀 Starting data download...")
        print(f"📅 Date range: {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')}")
        print(f"🔄 Incremental mode: {incremental}")
        
        if incremental and self.download_history["last_update"]:
            print(f"📝 Last update: {self.download_history['last_update']}")
            # Note: We'll still check for missing data even if last update was today
            # This allows for recovery from interrupted downloads or manual file additions
        
        total_stocks = len(StockName)
        
        # Download stock data first (all stocks)
        print(f"\n📈 Downloading stock data for {total_stocks} stocks...")
        for i, stock_name in enumerate(tqdm(StockName, desc="Downloading stock data")):
            print(f"\n📊 Processing stock: {stock_name.value} ({i+1}/{total_stocks})")
            
            # Download stock data
            if incremental:
                missing_stock_dates = self.get_missing_dates(stock_name, "stock")
                if missing_stock_dates:
                    self.download_stock_data(stock_name, min(missing_stock_dates), max(missing_stock_dates))
            else:
                # For non-incremental mode, check actual files to avoid re-downloading
                missing_stock_dates = self.get_missing_dates_from_files(stock_name, "stock")
                if missing_stock_dates:
                    self.download_stock_data(stock_name, min(missing_stock_dates), max(missing_stock_dates))
                else:
                    print(f"✅ Stock data already exists for {stock_name.value} in date range")
        
        # Download options data by date to maximize cache reuse
        print(f"\n📊 Downloading options data (optimized with caching)...")
        
        # Get all missing dates for options data
        all_missing_dates = set()
        for stock_name in StockName:
            if incremental:
                missing_dates = self.get_missing_dates(stock_name, "options")
            else:
                missing_dates = self.get_missing_dates_from_files(stock_name, "options")
            
            all_missing_dates.update(missing_dates)
        
        if all_missing_dates:
            print(f"📅 Processing {len(all_missing_dates)} dates for options data...")
            
            # Process by date to maximize cache reuse
            processed_dates_count = 0
            for date in sorted(all_missing_dates):
                print(f"\n📅 Processing date: {date.strftime('%Y-%m-%d')}")
                
                # Download options data for all stocks on this date
                for stock_name in StockName:
                    # Check if this stock needs data for this date
                    date_str = date.strftime('%Y-%m-%d')
                    
                    if incremental:
                        history_dates = self.download_history["options_data"].get(stock_name.value, [])
                        needs_download = date_str not in history_dates
                    else:
                        # For non-incremental mode, check if file exists
                        options_dir = os.path.join(self.config.base_dir, self.config.options_data_dir, stock_name.value)
                        filename = self.get_options_data_filename(stock_name, date)
                        filepath = os.path.join(options_dir, filename)
                        needs_download = not os.path.exists(filepath)
                    
                    if needs_download:
                        self.download_options_data_with_retry(stock_name, date)
                
                # Increment processed dates counter
                processed_dates_count += 1
                
                # Save download history every 100 dates
                if processed_dates_count % 100 == 0:
                    print(f"💾 Saving download history after processing {processed_dates_count} dates...")
                    self.save_download_history()
        
        # Save download history
        self.save_download_history()
        print(f"\n✅ Data download completed!")
        
        # Show cache status
        if hasattr(self.jugaad_downloader, 'bhavcopy_cache'):
            cache_size = len(self.jugaad_downloader.bhavcopy_cache)
            print(f"📊 Bhavcopy cache: {cache_size} dates cached for reuse")
        
        # Cleanup temporary files (preserves cache)
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files from jugaad-data"""
        try:
            if self.jugaad_downloader is not None:
                self.jugaad_downloader.cleanup()
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

def main():
    """Main function to run the data downloader"""
    # Configuration
    config = DownloadConfig(
        base_dir="market_data",
        stock_data_dir="stock_data",
        options_data_dir="options_data"
    )
    
    # Create downloader
    downloader = DataDownloader(config)
    
    # Check if this is an incremental update
    incremental = True
    if not os.path.exists(os.path.join(config.base_dir, "download_history.json")):
        incremental = False
        print("🆕 First time download - downloading all historical data")
    
    # Download data
    downloader.download_all_data(incremental=incremental)

if __name__ == "__main__":
    main() 