#!/usr/bin/env python3
"""
Test script to verify jugaad-data integration works correctly
"""

from datetime import datetime, timedelta
from data_downloader import DataDownloader, DownloadConfig
from stocks import StockName

def test_jugaad_data():
    """Test the jugaad-data integration with a small dataset"""
    print("🧪 Testing jugaad-data integration...\n")
    
    try:
        # Configure for just today and yesterday
        config = DownloadConfig(
            base_dir="test_jugaad_data",
            stock_data_dir="stock_data",
            options_data_dir="options_data",
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        # Create downloader
        downloader = DataDownloader(config)
        
        # Test with just NIFTY
        test_stock = StockName.NIFTY
        
        print(f"📅 Testing date range: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Testing stock: {test_stock.value}\n")
        
        # Test stock data download
        print("📈 Testing stock data download...")
        success = downloader.download_stock_data(test_stock, config.start_date, config.end_date)
        if success:
            print(f"✅ Stock data test passed for {test_stock.value}")
        else:
            print(f"❌ Stock data test failed for {test_stock.value}")
        
        # Test options data download (only on weekdays during market hours)
        print("\n📊 Testing options data download with jugaad-data...")
        today = datetime.now()
        if today.weekday() < 5:  # Only test on weekdays
            print(f"🔍 Testing options for {test_stock.value} on {today.strftime('%Y-%m-%d')}...")
            print("⚠️  Note: Options data only available during market hours (9:15 AM - 3:30 PM IST)")
            success = downloader.download_options_data_with_retry(test_stock, today)
            if success:
                print(f"✅ Options data test passed for {test_stock.value}")
            else:
                print(f"❌ Options data test failed for {test_stock.value}")
                print("💡 This is normal if market is closed or data is not yet available")
        else:
            print("📅 Skipping options test (weekend)")
        
        print("\n🎉 Jugaad-data integration test completed!")
        print("📁 Check the 'test_jugaad_data' folder for downloaded files")
        
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Please install jugaad-data: pip install jugaad-data")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_jugaad_data() 