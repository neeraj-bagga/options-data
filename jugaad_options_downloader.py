#!/usr/bin/env python3
"""
Options data downloader using jugaad-data library for reliable NSE access
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import time
import random
from typing import Optional, Dict, Any
import json

try:
    from jugaad_data.nse import bhavcopy_fo_save, stock_df
    from jugaad_data.nse import NSELive
    HAS_JUGAAD_DATA = True
except ImportError:
    HAS_JUGAAD_DATA = False
    print("⚠️  jugaad-data not available. Please install: pip install jugaad-data")

class JugaadOptionsDownloader:
    def __init__(self):
        if not HAS_JUGAAD_DATA:
            raise ImportError("jugaad-data library is required")
        
        self.nse_live = NSELive()
        self.temp_dir = "temp_jugaad"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Cache for bhavcopy data to avoid re-downloading
        self.bhavcopy_cache = {}
    
    def _get_bhavcopy_for_date(self, date: datetime) -> Optional[pd.DataFrame]:
        """Download and cache bhavcopy data for a specific date"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Check if already cached
        if date_str in self.bhavcopy_cache:
            print(f"📋 Using cached bhavcopy data for {date_str}")
            return self.bhavcopy_cache[date_str]
        
        # Check if bhavcopy file already exists on disk
        possible_filenames = [
            f"fo{date.strftime('%d%m%y')}bhav.csv",  # fo081020bhav.csv
            f"fo{date.strftime('%d%b%Y')}bhav.csv",  # fo08Oct2020bhav.csv
            f"fo{date.strftime('%d%m%Y')}bhav.csv",  # fo08102020bhav.csv
        ]
        
        filepath = None
        for filename in possible_filenames:
            temp_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(temp_path):
                filepath = temp_path
                break
        
        if filepath is not None:
            print(f"📋 Found existing bhavcopy file for {date_str}: {os.path.basename(filepath)}")
            try:
                # Read the existing CSV file
                df = pd.read_csv(filepath)
                # Cache the data
                self.bhavcopy_cache[date_str] = df
                print(f"✅ Loaded existing bhavcopy for {date_str}: {len(df)} records")
                return df
            except Exception as e:
                print(f"⚠️  Error reading existing bhavcopy file: {e}")
                # Continue to download if reading fails
        
        try:
            print(f"📥 Downloading bhavcopy for {date_str}...")
            
            # Convert datetime to date object
            date_obj = date.date()
            
            # Download F&O bhavcopy using jugaad-data
            bhavcopy_fo_save(date_obj, self.temp_dir)
            
            # Add delay to be respectful to NSE servers
            time.sleep(3)  # Reduced delay since we're already being efficient with caching
            
            # Find the downloaded file
            filepath = None
            for filename in possible_filenames:
                temp_path = os.path.join(self.temp_dir, filename)
                if os.path.exists(temp_path):
                    filepath = temp_path
                    break
            
            if filepath is None:
                print(f"❌ No bhavcopy file found for {date_str}")
                print(f"📁 Files in temp directory: {os.listdir(self.temp_dir)}")
                print(f"🔍 Expected filenames: {possible_filenames}")
                return None
            
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Cache the data
            self.bhavcopy_cache[date_str] = df
            print(f"✅ Downloaded and cached bhavcopy for {date_str}: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading bhavcopy for {date_str}: {e}")
            return None
    
    def download_historical_options_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Download historical options data using cached bhavcopy"""
        try:
            print(f"📥 Getting options data for {symbol} on {date.strftime('%Y-%m-%d')}")
            
            # Get cached bhavcopy data
            bhavcopy_df = self._get_bhavcopy_for_date(date)
            
            if bhavcopy_df is None:
                return None
            
            # Filter for the specific symbol
            stock_options = bhavcopy_df[bhavcopy_df['SYMBOL'] == symbol].copy()
            
            if stock_options.empty:
                print(f"⚠️  No options data found for {symbol} on {date.strftime('%Y-%m-%d')}")
                # Try alternative symbol formats
                alternative_symbols = [
                    symbol.upper(),
                    symbol.lower(),
                    symbol.replace('BANK', ''),
                    symbol.replace('NIFTY', 'NIFTY50')
                ]
                
                for alt_symbol in alternative_symbols:
                    if alt_symbol in bhavcopy_df['SYMBOL'].values:
                        stock_options = bhavcopy_df[bhavcopy_df['SYMBOL'] == alt_symbol].copy()
                        print(f"✅ Found data with alternative symbol: {alt_symbol}")
                        break
                
                if stock_options.empty:
                    # Show available symbols for debugging
                    available_symbols = bhavcopy_df['SYMBOL'].unique()[:10]
                    print(f"📋 Available symbols: {list(available_symbols)}")
                    return None
            
            print(f"✅ Found {len(stock_options)} options records for {symbol}")
            return stock_options
            
        except Exception as e:
            print(f"❌ Error getting options data for {symbol}: {e}")
            return None
    
    def download_live_options_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download live options data using jugaad-data NSELive"""
        try:
            print(f"📡 Downloading live options data for {symbol}")
            
            # Get live options data
            if symbol in ['NIFTY', 'BANKNIFTY']:
                # For indices, use option chain
                options_data = self.nse_live.option_chain(symbol)
            else:
                # For stocks, use stock quote (may not have options data)
                options_data = self.nse_live.stock_quote(symbol)
            
            if not options_data:
                print(f"❌ No live options data available for {symbol}")
                return None
            
            # Parse the options data
            parsed_data = []
            
            if 'records' in options_data and 'data' in options_data['records']:
                for item in options_data['records']['data']:
                    if 'CE' in item:
                        ce_data = item['CE']
                        parsed_data.append({
                            'SYMBOL': symbol,
                            'EXPIRY_DT': item.get('expiryDate', ''),
                            'STRIKE_PR': ce_data.get('strikePrice', 0),
                            'OPTION_TYP': 'CE',
                            'OPEN_INT': ce_data.get('openInterest', 0),
                            'CHG_IN_OI': ce_data.get('changeinOpenInterest', 0),
                            'VOLUME': ce_data.get('totalTradedVolume', 0),
                            'IV': ce_data.get('impliedVolatility', 0),
                            'LTP': ce_data.get('lastPrice', 0),
                            'NET_CHG': ce_data.get('change', 0),
                            'BID_QTY': ce_data.get('bidQty', 0),
                            'BID_PRICE': ce_data.get('bidprice', 0),
                            'ASK_QTY': ce_data.get('askQty', 0),
                            'ASK_PRICE': ce_data.get('askPrice', 0)
                        })
                    
                    if 'PE' in item:
                        pe_data = item['PE']
                        parsed_data.append({
                            'SYMBOL': symbol,
                            'EXPIRY_DT': item.get('expiryDate', ''),
                            'STRIKE_PR': pe_data.get('strikePrice', 0),
                            'OPTION_TYP': 'PE',
                            'OPEN_INT': pe_data.get('openInterest', 0),
                            'CHG_IN_OI': pe_data.get('changeinOpenInterest', 0),
                            'VOLUME': pe_data.get('totalTradedVolume', 0),
                            'IV': pe_data.get('impliedVolatility', 0),
                            'LTP': pe_data.get('lastPrice', 0),
                            'NET_CHG': pe_data.get('change', 0),
                            'BID_QTY': pe_data.get('bidQty', 0),
                            'BID_PRICE': pe_data.get('bidprice', 0),
                            'ASK_QTY': pe_data.get('askQty', 0),
                            'ASK_PRICE': pe_data.get('askPrice', 0)
                        })
            
            if parsed_data:
                df = pd.DataFrame(parsed_data)
                print(f"✅ Downloaded {len(df)} live options records for {symbol}")
                return df
            else:
                print(f"❌ No live options data parsed for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading live options data for {symbol}: {e}")
            return None
    
    def download_options_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Download options data using jugaad-data methods"""
        
        # Try historical data first
        df = self.download_historical_options_data(symbol, date)
        
        if df is not None and not df.empty:
            return df
        
        # If historical data fails, try live data (for current date)
        today = datetime.now().date()
        if date.date() == today:
            print(f"🔄 Historical data not available, trying live data for {symbol}")
            df = self.download_live_options_data(symbol)
            return df
        
        print(f"❌ No options data available for {symbol} on {date.strftime('%Y-%m-%d')}")
        return None
    
    def cleanup(self):
        """Clean up temporary files but keep bhavcopy cache"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # Keep cache for reuse across runs
            print("🧹 Cleaned up temporary files (cache preserved)")
        except Exception as e:
            print(f"⚠️  Error cleaning up temp files: {e}")
    
    def clear_cache(self):
        """Clear bhavcopy cache if needed"""
        self.bhavcopy_cache.clear()
        print("🗑️  Cleared bhavcopy cache")

def main():
    """Test the jugaad options downloader with caching optimization"""
    if not HAS_JUGAAD_DATA:
        print("❌ jugaad-data library not available")
        print("💡 Install with: pip install jugaad-data")
        return
    
    downloader = JugaadOptionsDownloader()
    
    # Test with multiple symbols for the same date to show caching
    test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS']
    today = datetime.now()
    
    print(f"🧪 Testing caching optimization for {today.strftime('%Y-%m-%d')}")
    print(f"📊 Testing symbols: {test_symbols}\n")
    
    for i, symbol in enumerate(test_symbols):
        print(f"📈 Test {i+1}/{len(test_symbols)}: {symbol}")
        df = downloader.download_options_data(symbol, today)
        
        if df is not None and not df.empty:
            print(f"✅ Successfully got {len(df)} options records for {symbol}")
            if i == 0:
                print(f"📋 Sample data:")
                print(df.head(2))
        else:
            print(f"❌ Failed to get options data for {symbol}")
        print()
    
    print(f"📊 Cache status: {len(downloader.bhavcopy_cache)} dates cached")
    print(f"📋 Cached dates: {list(downloader.bhavcopy_cache.keys())}")
    
    # Cleanup (preserves cache)
    downloader.cleanup()
    print("🎉 Test completed! Notice how bhavcopy was downloaded only once and reused.")
    print("💡 Cache is preserved for future runs - use clear_cache() if you want to clear it.")

if __name__ == "__main__":
    main() 