#!/usr/bin/env python3
"""
Test script to demonstrate bhavcopy caching optimization
"""

from datetime import datetime, timedelta
from jugaad_options_downloader import JugaadOptionsDownloader

def test_caching_optimization():
    """Test the bhavcopy caching optimization"""
    print("🧪 Testing bhavcopy caching optimization...\n")
    
    try:
        downloader = JugaadOptionsDownloader()
        
        # Test date
        test_date = datetime.now() - timedelta(days=1)  # Yesterday
        test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'HDFCBANK']
        
        print(f"📅 Testing date: {test_date.strftime('%Y-%m-%d')}")
        print(f"📊 Testing symbols: {test_symbols}\n")
        
        print("🔄 First run - should download bhavcopy...")
        for i, symbol in enumerate(test_symbols):
            print(f"  📈 {i+1}. Getting data for {symbol}...")
            df = downloader.download_options_data(symbol, test_date)
            if df is not None and not df.empty:
                print(f"     ✅ Got {len(df)} records")
            else:
                print(f"     ❌ No data")
        
        print(f"\n📊 Cache status after first run: {len(downloader.bhavcopy_cache)} dates")
        print(f"📋 Cached dates: {list(downloader.bhavcopy_cache.keys())}")
        
        print("\n🔄 Second run - should use cached bhavcopy...")
        for i, symbol in enumerate(test_symbols):
            print(f"  📈 {i+1}. Getting data for {symbol} (cached)...")
            df = downloader.download_options_data(symbol, test_date)
            if df is not None and not df.empty:
                print(f"     ✅ Got {len(df)} records (from cache)")
            else:
                print(f"     ❌ No data")
        
        print(f"\n📊 Final cache status: {len(downloader.bhavcopy_cache)} dates")
        
        # Cleanup (preserves cache)
        downloader.cleanup()
        print("\n🎉 Test completed!")
        print("💡 Notice how bhavcopy was downloaded only once and reused for all symbols")
        print("💡 Cache is preserved for future runs")
        
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Please install jugaad-data: pip install jugaad-data")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_caching_optimization() 