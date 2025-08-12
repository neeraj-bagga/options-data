# Options Strategies Backtesting Framework

A comprehensive framework for backtesting options strategies using local market data. This framework provides utilities for calculating options Greeks, implied volatility, and running strategy backtests with performance metrics.

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For enhanced accuracy, install options pricing libraries:**
   ```bash
   pip install py_vollib QuantLib
   ```

3. **Run the example:**
   ```bash
   python example_usage.py          # Manual calculations
   python example_usage_lib.py      # Library-based calculations (recommended)
   ```

## 📁 File Structure

```
strategies/
├── options_utils.py           # Manual options calculations
├── options_utils_lib.py       # Library-based calculations (py_vollib)
├── data_loader.py             # Data loading and processing
├── strategy_framework.py      # Strategy base classes and backtesting
├── example_usage.py           # Manual calculations demo
├── example_usage_lib.py       # Library-based calculations demo
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Components

### 1. Options Calculator

#### Manual Calculator (`options_utils.py`)
The `OptionsCalculator` class provides comprehensive options pricing and Greeks calculations:

**Key Features:**
- **Black-Scholes pricing** for calls and puts
- **Greeks calculation**: Delta, Gamma, Theta, Vega, Rho
- **Implied Volatility** calculation using Newton-Raphson method
- **Additional metrics**: Moneyness, time value, breakeven, probability ITM

#### Library-based Calculator (`options_utils_lib.py`) - **Recommended**
The `OptionsCalculatorLib` class uses industry-standard libraries for enhanced accuracy:

**Key Features:**
- **py_vollib integration** for accurate Black-Scholes calculations
- **QuantLib support** for advanced financial instruments
- **Automatic fallback** to manual calculations if libraries unavailable
- **Better numerical stability** and faster computation
- **Industry-standard algorithms**

**Example Usage:**

```python
from options_utils_lib import OptionsCalculatorLib

calculator = OptionsCalculatorLib(risk_free_rate=0.05)

# Calculate option price and Greeks
S = 100.0  # Stock price
K = 100.0  # Strike price
T = 0.25   # Time to expiry (years)
sigma = 0.3  # Volatility

# Call option with py_vollib
call_price = calculator.black_scholes_call(S, K, T, 0.05, sigma)
call_greeks = calculator.calculate_all_greeks(S, K, T, 0.05, sigma, 'CE')

print(f"Call Price: {call_price:.4f}")
print(f"Delta: {call_greeks['delta']:.4f}")
print(f"Gamma: {call_greeks['gamma']:.4f}")
print(f"Theta: {call_greeks['theta']:.4f}")
print(f"Vega: {call_greeks['vega']:.4f}")

# Calculate implied volatility
iv = calculator.calculate_implied_volatility(call_price, S, K, T, 0.05, 'CE')
print(f"Implied Volatility: {iv:.6f}")
```

### 2. Data Loader (`data_loader.py`)

The `OptionsDataLoader` class handles loading and processing market data:

#### Key Features:
- **Cached data loading** for performance
- **Automatic data cleaning** and validation
- **ATM options identification**
- **Historical data loading** for backtesting
- **Stock price retrieval**

#### Example Usage:

```python
from data_loader import OptionsDataLoader

loader = OptionsDataLoader()

# Load options data
options_data = loader.load_options_data("NIFTY", "2025-08-05")

# Get ATM options
atm_options = loader.get_atm_options("NIFTY", "2025-08-05")
print(f"ATM CE options: {len(atm_options['CE'])}")
print(f"ATM PE options: {len(atm_options['PE'])}")

# Load historical data
historical_data = loader.load_historical_data(
    "NIFTY", "2025-08-01", "2025-08-05", "both"
)
```

### 3. Strategy Framework (`strategy_framework.py`)

The framework provides an abstract base class for implementing options strategies:

#### Key Features:
- **Abstract strategy base class** for easy strategy implementation
- **Position tracking** and trade execution simulation
- **Performance metrics calculation** (returns, Sharpe ratio, drawdown)
- **Built-in strategies**: Iron Condor, Long Straddle

#### Example Strategy Implementation:

```python
from strategy_framework import OptionsStrategy

class MyCustomStrategy(OptionsStrategy):
    def __init__(self, symbol, **kwargs):
        super().__init__(symbol, **kwargs)
        # Add strategy-specific parameters
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on strategy logic"""
        signals = []
        # Implement your signal generation logic
        return signals
    
    def execute_trades(self, signals, data):
        """Execute trades based on signals"""
        # Implement your trade execution logic
        pass
```

#### Built-in Strategies:

**Iron Condor Strategy:**
- Sells OTM call and put spreads
- Configurable delta threshold and days to expiry
- Suitable for range-bound markets

**Long Straddle Strategy:**
- Buys ATM call and put options
- Profits from large price movements
- Suitable for high volatility expectations

## 📊 Performance Metrics

The framework calculates comprehensive performance metrics:

- **Total Return**: Overall strategy performance
- **Annualized Return**: Yearly performance rate
- **Volatility**: Strategy risk measure
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Total Trades**: Number of executed trades

## ⚙️ Configuration

### Risk-Free Rate
Set the risk-free rate for options calculations:
```python
calculator = OptionsCalculatorLib(risk_free_rate=0.05)  # 5%
```

### Strategy Parameters
Configure strategy-specific parameters:
```python
iron_condor = IronCondorStrategy(
    symbol="NIFTY",
    delta_threshold=0.15,    # Delta threshold for option selection
    days_to_expiry=30,       # Minimum days to expiry
    risk_free_rate=0.05      # Risk-free rate
)
```

## 📈 Data Requirements

### Options Data Format
Expected columns in options CSV files:
- `INSTRUMENT`: 'OPTIDX' for options
- `SYMBOL`: Underlying symbol (e.g., 'NIFTY')
- `EXPIRY_DT`: Expiry date (YYYY-MM-DD)
- `STRIKE_PR`: Strike price
- `OPTION_TYP`: 'CE' for calls, 'PE' for puts
- `CLOSE`: Closing price
- `OPEN_INT`: Open interest
- `VOLUME`: Trading volume

### Stock Data Format
Expected columns in stock CSV files:
- `Date`: Trading date
- `Open`, `High`, `Low`, `Close`: OHLC prices
- `Volume`: Trading volume

## 🔍 Important Notes

1. **Library vs Manual Calculations**: The library-based approach (`options_utils_lib.py`) is recommended for production use as it provides more accurate and stable calculations.

2. **Data Quality**: Ensure your options data includes all required columns and is properly formatted.

3. **Risk Management**: This framework is for educational and research purposes. Always implement proper risk management in live trading.

4. **Performance**: Library-based calculations are significantly faster and more accurate than manual implementations.

## 🛠️ Troubleshooting

### Common Issues:

1. **Missing Dependencies:**
   ```bash
   pip install py_vollib QuantLib
   ```

2. **Data Loading Errors:**
   - Check file paths and formats
   - Ensure CSV files have required columns
   - Verify date formats (YYYY-MM-DD)

3. **Calculation Errors:**
   - Check for invalid option prices (negative or zero)
   - Verify time to expiry calculations
   - Ensure stock prices are positive

### Performance Optimization:

1. **Use Library-based Calculations**: `py_vollib` provides optimized implementations
2. **Enable Data Caching**: The data loader caches results for faster subsequent access
3. **Batch Processing**: Process multiple options simultaneously when possible

## 📚 Additional Resources

- **py_vollib Documentation**: https://py-vollib.readthedocs.io/
- **QuantLib Documentation**: https://www.quantlib.org/
- **Black-Scholes Model**: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
- **Options Greeks**: https://en.wikipedia.org/wiki/Greeks_(finance)

## 🤝 Contributing

To add new strategies or improve the framework:

1. Create a new strategy class inheriting from `OptionsStrategy`
2. Implement `generate_signals()` and `execute_trades()` methods
3. Add comprehensive documentation and examples
4. Test with historical data

## 📄 License

This framework is provided for educational and research purposes. Use at your own risk in live trading environments. 

## Portfolio Backtesting Framework (New)

Added modules:
- `risk_metrics.py`: compute per-symbol std dev and beta vs `nifty`, and classify IV regime (low/medium/high) via average IV percentile.
- `trading_strategies.py`: reusable `ShortStrangle` and `IronCondor` with tunable params and exit rules (TP/SL/DTE).
- `portfolio_framework.py`: portfolio allocator applying IV-based buying power (25/35/50% default), 75/25 index-vs-stock split, beta*delta neutrality, and theta cap (<=0.2% per day on total buying power).
- `run_portfolio_backtest.py`: CLI to compute high-level stats and run a portfolio backtest on locally downloaded data.

Quick run:

```bash
python -m strategies.run_portfolio_backtest
```

Tune allocations/strategies by editing `AllocationConfig` and `TradingConfig` inside `run_portfolio_backtest.py`. 