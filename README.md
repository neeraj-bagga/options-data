# Market Data Downloader

A comprehensive tool to download stock data and options data for Indian stocks using proven libraries.

## Features

- **Stock Data**: Downloads daily stock data for all stocks in `stocks.py` using yfinance
- **Options Data**: Downloads daily options data using jugaad-data (reliable NSE access)
- **Incremental Updates**: Only downloads new data when run again
- **Organized Storage**: Saves data in structured folders for easy analysis
- **5 Years Historical Data**: Downloads data for the last 5 years by default

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Important**: The `jugaad-data` library is required for options data. It provides reliable access to NSE data.

## Usage

### First Time Run (Full Download)
```bash
python data_downloader.py
```

This will:
- Download 5 years of historical data for all stocks
- Create the folder structure automatically
- Save both stock data and options data

### Daily Updates
```bash
python data_downloader.py
```

When run again, it will:
- Check what data is already downloaded
- Only download new data (typically just today's data)
- Update the local cache efficiently

## Folder Structure

```
market_data/
├── download_history.json          # Tracks what's been downloaded
├── stock_data/                    # Stock price data
│   ├── nifty/
│   │   ├── nifty_2023-01-01.csv
│   │   ├── nifty_2023-01-02.csv
│   │   └── nifty_complete.csv     # Complete dataset
│   ├── reliance/
│   └── ...
└── options_data/                  # Options data
    ├── nifty/
    │   ├── nifty_options_2023-01-01.csv
    │   └── ...
    ├── reliance/
    └── ...
```

## Data Sources

- **Stock Data**: Yahoo Finance (yfinance)
- **Options Data**: NSE via jugaad-data library

## Supported Stocks

The tool downloads data for all stocks defined in `stocks.py`, including:
- NIFTY 50
- Bank NIFTY
- Major Indian stocks (Reliance, TCS, HDFC Bank, etc.)
- And many more...

## Configuration

You can modify the download configuration in `data_downloader.py`:

```python
config = DownloadConfig(
    base_dir="market_data",           # Base directory
    stock_data_dir="stock_data",      # Stock data subdirectory
    options_data_dir="options_data",  # Options data subdirectory
    start_date=datetime(2019, 1, 1),  # Custom start date
    end_date=datetime.now()           # Custom end date
)
```

## Error Handling

- The script handles network errors gracefully
- Missing data is logged but doesn't stop the process
- Rate limiting is implemented to avoid overwhelming servers
- **Options data requires jugaad-data library** - will fail if not installed

## Notes

- Options data is only available for trading days (Monday-Friday)
- Some stocks may not have options data available for all dates
- The script includes rate limiting to be respectful to data sources
- All data is saved in CSV format for easy analysis
- **No fallback methods** - uses hanya jugaad-data for options

## Troubleshooting

1. **Network Issues**: Check your internet connection
2. **Missing Data**: Some dates may not have data available (holidays, weekends)
3. **Rate Limiting**: The script includes delays to avoid being blocked
4. **Disk Space**: Ensure you have enough disk space for 5 years of data
5. **jugaad-data Required**: Install with `pip install jugaad-data`

## Analysis Ready

The downloaded data is organized and ready for:
- Technical analysis
- Backtesting strategies
- Options analysis
- Portfolio analysis
- And more... 