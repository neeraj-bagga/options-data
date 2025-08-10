## Options Market Data and Strategies — Single-File Documentation

### Overview

This repository provides a complete pipeline to fetch, organize, query, and analyze Indian equities and options data, plus a framework to backtest options strategies. It consists of:

- Data downloader for stocks (Yahoo Finance) and options (NSE via jugaad-data)
- Local filesystem cache with incremental updates and history tracking
- Query utilities to explore available data and load specific slices
- Options analytics (Black–Scholes, Greeks, IV, IV percentile) via manual math or libraries
- A lightweight strategy backtesting framework with sample strategies


### Directory Layout

```
project-root/
├── data_downloader.py                 # Orchestrates stock/options downloads
├── data_query.py                      # Query utilities for local cache
├── jugaad_options_downloader.py       # Thin wrapper around jugaad-data with caching
├── stocks.py                          # Enum + symbol map (EQ/NFO/exchanges)
├── strategies/                        # Backtesting framework and utilities
│   ├── data_loader.py                 # Loads/cleans options & stock data
│   ├── options_utils.py               # Manual Black–Scholes + metrics
│   ├── options_utils_lib.py           # Library-based (py_vollib/QuantLib)
│   ├── strategy_framework.py          # Base class + Iron Condor, Straddle
│   ├── example_usage.py               # End-to-end example (manual utils)
│   ├── example_usage_lib.py           # End-to-end example (library utils)
│   └── README.md                      # Strategies readme
├── market_data/                       # Downloaded data (created at runtime)
│   ├── stock_data/<symbol>/           # Daily CSVs + complete CSV
│   └── options_data/<symbol>/         # Daily CSVs of options bhavcopy slices
├── requirements.txt                   # Core dependencies
├── run_downloader.bat                 # Windows convenience script
├── test_caching.py                    # Bhavcopy caching demo
├── test_jugaad_data.py                # Minimal jugaad-data integration test
└── README.md                          # Downloader quickstart
```


### Install

```bash
pip install -r requirements.txt
# For strategy analytics accuracy (recommended):
pip install py_vollib QuantLib
```


### Quickstart

- Full historical download (first run creates folders and backfills ~5–6 years):

```bash
python data_downloader.py
```

- Daily update (incremental):

```bash
python data_downloader.py
```

- Explore local cache and load specific files programmatically:

```python
from data_query import DataQuery

q = DataQuery(base_dir="market_data")
q.print_summary()

symbols = q.get_available_symbols()
avail = q.get_data_availability(symbols[0])
stock_df = q.get_stock_data(symbols[0], avail.stock_dates[0])
opt_df = q.get_options_data(symbols[0], avail.options_dates[0])
```

- Strategy example (library-based):

```python
from strategies.data_loader import OptionsDataLoader
from strategies.strategy_framework import IronCondorStrategy

loader = OptionsDataLoader(data_dir="market_data")
strategy = IronCondorStrategy(loader, delta_threshold=0.15, days_to_expiry=30)
results = strategy.backtest("nifty", "2023-10-10", "2023-10-27", 100000)
print(results["total_return"], results["sharpe_ratio"]) 
```


### Data Model and File Naming

- Stock data (Yahoo Finance daily): `market_data/stock_data/<symbol>/<symbol>_YYYY-MM-DD.csv`
  - Also `market_data/stock_data/<symbol>/<symbol>_complete.csv`
  - Columns include: `Date, Open, High, Low, Close, Volume`

- Options data (NSE bhavcopy slice via jugaad-data): `market_data/options_data/<symbol>/<symbol>_options_YYYY-MM-DD.csv`
  - Key columns used: `INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, OPEN_INT, CHG_IN_OI, VOLUME`
  - Strategy utilities focus on options rows (`INSTRUMENT == 'OPTIDX'`)

- Symbols come from `stocks.py` with unified names (e.g., `nifty`, `reliance`, `bank_nifty`) and a mapping to EQ/NFO symbols.


### Core Modules

#### stocks.py

- `StockName` (Enum): canonical symbols like `NIFTY`, `RELIANCE`, `BANK_NIFTY`, etc.
- `STOCK_NAME_MAP`: maps each enum to `StockNameInfo(eq, nfo, eq_exchange, fo_exchange)` used by both Yahoo and NSE queries.

#### data_downloader.py

- `DownloadConfig`: base directory names and date range. Defaults to ~6 years lookback to include 5+ years.
- `DataDownloader` orchestrates end-to-end downloads:
  - Directory setup and `download_history.json` maintenance
  - Syncs history against files on disk (idempotent)
  - Stock downloader (yfinance): finds missing trading days, downloads a continuous range, writes per-day CSVs and maintains a merged `*_complete.csv`
  - Options downloader (jugaad-data via `JugaadOptionsDownloader`): per-date workflow across all symbols to maximize reuse of a cached bhavcopy
  - Holiday/weekend filter via `is_trading_day` (2019–2025 calendars included)
  - Incremental vs file-based missing-dates detection
  - Periodic save of download history and cleanup of temp files (cache preserved)
- CLI: running the module auto-detects first-run vs incremental and downloads accordingly.

#### jugaad_options_downloader.py

- `JugaadOptionsDownloader` wraps `jugaad_data.nse` with:
  - Bhavcopy caching per date in-memory and reuse across multiple symbols
  - Attempts to load any existing bhavcopy file from `temp_jugaad/` before downloading
  - Historical path (preferred) and live path (only for current trading day) with simple parsing
  - Utilities: `cleanup()` (remove temp dir, keep cache) and `clear_cache()`

#### data_query.py

- `DataQuery` reads `market_data`, inspects `download_history.json`, and lists/loads available data.
- Key APIs:
  - `get_available_symbols()`
  - `get_data_availability(symbol)` → date ranges per type
  - `get_stock_data(symbol, date)` / `get_options_data(symbol, date)`
  - `get_both_data(symbol, date)` and `print_summary()`

#### strategies/data_loader.py

- `OptionsDataLoader` loads and cleans both options and stock data for backtests.
- Highlights:
  - Symbol/date discovery helpers
  - Fallback to `*_complete.csv` if a specific stock date file is missing
  - Cleans numeric columns and timestamps
  - Convenience: `get_stock_price(symbol, date)`, `get_atm_options(symbol, date, tolerance)`, `get_option_chain(symbol, date, expiry)`

#### strategies/options_utils.py (manual math)

- `OptionsCalculator`: Black–Scholes calls/puts, Greeks (delta, gamma, theta, vega, rho), IV via Newton–Raphson, and derived metrics (moneyness, time value, breakeven, probability ITM).
- `process_options_data(df, stock_price, date, r)`: filters options, computes days-to-expiry, and adds Greeks/IV/metrics row-wise.
- `calculate_iv_percentile_for_symbol(...)`: simple, illustrative percentile calculation over a snapshot (placeholder for historical IV percentiles).

#### strategies/options_utils_lib.py (preferred)

- `OptionsCalculatorLib` using `py_vollib` (and optionally QuantLib) with stricter validation and robust IV/Greeks.
- `process_options_data_lib(...)`: library-based processing with filtering of invalid rows (`SETTLE_PR>0`, `STRIKE_PR>0`, `DAYS_TO_EXPIRY>0`) and detailed error reporting for failed rows.
- `calculate_iv_percentile_for_symbol_lib(...)`: same idea as manual version, with library-generated IVs.

#### strategies/strategy_framework.py

- `OptionsStrategy` (abstract):
  - Requires `generate_signals(...)` and `execute_trades(...)`
  - Tracks positions, trades, and computes portfolio value daily
  - `backtest(symbol, start_date, end_date, initial_capital)` iterates dates, loads data via `OptionsDataLoader`, executes strategy, and reports performance
- Built-ins:
  - `IronCondorStrategy`: short OTM call/put (delta threshold, target DTE); uses `process_options_data_lib` to select strikes
  - `StraddleStrategy`: long ATM CE+PE when IV is low; uses ATM selection from loader and optional IV heuristics
- Metrics returned: final value, total/annualized return, volatility, Sharpe, max drawdown, total trades, daily equity curve.


### Typical Workflows

- End-to-end download:
  1) First run: creates `market_data/`, syncs history based on existing files (if any), downloads missing stock data in ranges, enumerates all missing options dates, and fetches per-date bhavcopy once while slicing per symbol.
  2) Subsequent runs: incremental mode computes missing dates using `download_history.json`.

- Data exploration/consumption:
  - For quick inspection or scripting, use `DataQuery` from the project root.
  - For strategy research, use `OptionsDataLoader` under `strategies/` (defaults to `../market_data`).

- Backtesting:
  - Choose a symbol/date range where both stock and options exist
  - Use library-based processing for more reliable Greeks/IV
  - Iterate strategies and parameters; extend from `OptionsStrategy` for custom logic


### Configuration Notes

- Modify `DownloadConfig` in `data_downloader.py` to adjust base directory names, start/end dates, etc.
- `get_yfinance_symbol` maps indices and equities (e.g., `^NSEI` for NIFTY) and applies `.NS` suffix for stocks.
- Options processing filters to `INSTRUMENT == 'OPTIDX'` and expects standard NSE bhavcopy columns.
- Holiday filter in downloader covers 2019–2025; non-trading days are skipped.


### Examples

#### Customizing download range

```python
from datetime import datetime
from data_downloader import DownloadConfig, DataDownloader

config = DownloadConfig(
    base_dir="market_data",
    stock_data_dir="stock_data",
    options_data_dir="options_data",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2021, 12, 31),
)

downloader = DataDownloader(config)
downloader.download_all_data(incremental=False)
```

#### Loading ATM options and Greeks (library-based)

```python
from strategies.data_loader import OptionsDataLoader
from strategies.options_utils_lib import process_options_data_lib

loader = OptionsDataLoader(data_dir="market_data")
symbol, date = "nifty", "2023-10-10"
opt = loader.load_options_data(symbol, date)
px = loader.get_stock_price(symbol, date)
processed = process_options_data_lib(opt, px, date)
atm = loader.get_atm_options(symbol, date)
```


### Tests and Utilities

- `test_caching.py`: Demonstrates that one bhavcopy download is reused for multiple symbols on the same date.
- `test_jugaad_data.py`: Minimal smoke test for stock and options integration; options only succeed during market hours.
- `run_downloader.bat` (Windows): Creates/activates `venv`, installs requirements, runs basic tests, and starts the full download.


### Limitations & Gotchas

- Options data requires `jugaad-data` and is subject to NSE availability/rate limits.
- Live options path is only attempted for the current trading day; historical path is preferred.
- FIN NIFTY uses a proxy (`^NSEBANK`) for stock data in downloader’s mapping.
- Strategies use `SETTLE_PR` over `CLOSE` for pricing where available.
- Library-based processing requires `py_vollib`; install it for robust IV/Greeks.


### Extending the System

- Add a new symbol: extend `StockName` and `STOCK_NAME_MAP` in `stocks.py` (ensure correct `eq` and `nfo` symbols).
- Add a new strategy: subclass `OptionsStrategy`, implement `generate_signals()` and `execute_trades()`, and use `OptionsDataLoader` to access data.
- Enhance metrics: extend `OptionsCalculatorLib.calculate_option_metrics()` or augment `strategy_framework.py` summaries.


### Troubleshooting

- `ImportError: jugaad-data` → `pip install jugaad-data`
- No options data on a date → may be a holiday/weekend, or the contract didn’t exist; check logs; history sync will still mark date processed to avoid rework.
- Empty IV/Greeks → ensure `SETTLE_PR > 0`, non-zero time to expiry, and use library-based processing for stability.
- Missing stock per-day CSV → loader can fallback to `<symbol>_complete.csv`.


### Frequently Asked Questions

- Can I run only stock downloads? Yes, call `download_stock_data()` per symbol/date range or customize `download_all_data` logic.
- Where is the cache? Options bhavcopy cache is in-memory for the process; temp CSVs live in `temp_jugaad/` during a run.
- How to change base folders? Edit `DownloadConfig` (base_dir and subdirs).


### License and Use

This repository is for research/education. Use at your own risk for live trading. Respect data source terms and rate limits.
