# Trading Analysis & Backtesting Bot

A Python-based trading analysis system for intraday market data.

The project was built as a personal research/engineering project to experiment with rule-based signal generation, market-data processing, backtesting, parameter tuning, and automated Telegram notifications.

> **Disclaimer:** This project is for research and educational purposes. It does not provide financial advice and does not directly place broker orders.

[Overview](#overview) · [Local verification](#offline-verification) · [Installation](#installation) · [Limitations](#limitations)

## Offline verification

```bash
python examples/offline_check.py
```

This runs six functional checks using synthetic OHLCV data and real project functions, with no API requests or Telegram messages. It covers data cleanup, trading hours, local indicators, Parquet persistence, a constructed signal/backtest and vectorbt portfolio calculation. Temporary outputs are removed after the check. **These are software checks, not historical market results or evidence of profitability.**

See [actual output](docs/offline-output.txt) and [validation details](docs/VALIDATION.md). Live market-data and Telegram workflows were not validated during this review.

## Overview

The bot works primarily with intraday **5-minute OHLCV data** and implements a **Breakout + Retest** strategy.

The main workflow is:

```text
Financial Modeling Prep
          │
          ▼
   Market data (5m)
          │
          ▼
 Data validation / filtering
          │
          ▼
 Technical indicators
          │
          ▼
 Breakout + Retest detector
          │
          ├──────────────► Signal diagnostics
          │
          ▼
 Entry / SL / TP / R:R
          │
          ├──────────────► CSV / persistent data
          │
          └──────────────► Telegram notification

Historical data
      │
      ▼
 Backtesting
      │
      ├── Win rate
      ├── Profit Factor
      ├── CAGR
      ├── Max Drawdown
      └── PnL
```

## Main Features

### Market data

- Financial Modeling Prep API integration
- Intraday historical data
- 5-minute timeframe
- US market regular-hours filtering
- Retry handling for API failures
- HTTP 429 / rate-limit handling
- API-key rotation
- Local persistence of downloaded data
- Parquet storage when an engine is available
- CSV.GZ fallback

### Technical indicators

The project uses or calculates:

- EMA 9
- EMA 20
- EMA 21
- EMA 50
- RSI
- Short-period RSI
- MACD
- MACD histogram
- ADX
- ATR
- VWAP
- Volume moving average
- Candle body / average candle body

Indicators can be obtained through Financial Modeling Prep's technical-indicator API. The code also contains a `pandas_ta` fallback for indicator calculation.

## Trading Strategy

The core signal detector is based on a **Breakout + Retest** model.

### LONG

A potential long setup requires a bullish breakout of a consolidation level followed by a retest.

Depending on the configured profile, the detector can additionally check:

- EMA alignment
- RSI range
- MACD histogram direction
- breakout candle body size
- relative volume
- retest interaction with the breakout level
- retest candle shape
- minimum ATR conditions

The signal contains:

- direction
- entry
- stop loss
- take profit
- risk/reward ratio
- breakout level
- textual reason for the signal

### SHORT

The short-side logic mirrors the long setup:

- bearish breakout
- retest of the broken support level
- optional EMA filter
- RSI constraints
- MACD histogram confirmation
- volume confirmation
- retest validation
- calculated SL/TP/R:R

## Signal Filtering and Diagnostics

One of the project goals is not only to generate signals, but also to understand **why candidate setups were rejected**.

The detector records rejection reasons such as:

- EMA filter failure
- RSI outside the configured range
- MACD confirmation failure
- insufficient volume
- insufficient impulse size
- retest did not touch the level
- invalid retest candle
- invalid entry/SL/TP levels

Diagnostic information can be exported to CSV and JSON files.

This makes the strategy easier to debug and tune instead of treating the signal generator as a black box.

## Risk and Trade Logic

The project calculates:

- Entry
- Stop Loss
- Take Profit
- Risk/Reward ratio
- ATR-based stop adjustments
- trailing/1.5R notifications
- TP / SL monitoring

The backtester also models:

- commissions
- slippage
- partial position closing
- entry on the following bar
- equity curve
- maximum drawdown

The project does **not** implement direct broker order execution.

## Backtesting

The script contains a dedicated backtesting pipeline.

For a ticker, it:

1. Downloads historical intraday data.
2. Validates and stores the raw dataset.
3. Filters the required market hours.
4. Adds technical indicators.
5. Runs the Breakout + Retest detector over historical bars.
6. Simulates entries and exits.
7. Applies commission and slippage assumptions.
8. Calculates performance statistics.
9. Exports backtesting data and diagnostics.

### Reported metrics

The backtest reports:

- Number of trades
- Profit Factor
- Win rate
- CAGR
- Maximum Drawdown
- PnL
- Gross profit
- Gross loss
- Equity curve

Example report format:

```text
📊 Бэктест: AAPL
├─ Сделок: ...
├─ PF: ... | Win-rate: ...%
├─ CAGR: ...% | MaxDD: ...%
└─ PnL: $...
```

The repository should contain actual backtest results separately from this README rather than presenting illustrative numbers as real performance.

## Parameter Tuning

The project includes an `--autotune` mode.

It can search/tune strategy parameters for an individual ticker and save the resulting parameters back to `tickers_config.json`.

Parameters used by the strategy include, among others:

- consolidation lookback
- EMA periods
- RSI limits
- volume multiplier
- impulse size multiplier
- ATR threshold
- TP risk/reward
- entry mode
- EMA filtering

Parameter tuning should be evaluated on out-of-sample data before treating the result as evidence of strategy quality.

## Event Filters

The project also takes scheduled corporate events into account.

It retrieves earnings-calendar information and contains blackout windows around:

- earnings announcements
- stock splits

This is intended to avoid generating signals around selected event windows.

## Telegram Integration

Telegram is used for automated notifications.

The bot can send:

- trading signals
- Entry / SL / TP / R:R information
- TP notifications
- SL notifications
- 1.5R / trailing notifications
- bot status / heartbeat messages
- API failure alerts

A periodic heartbeat is implemented to detect failures in Telegram communication.

## Data Storage

The project stores several types of data locally.

Typical structure:

```text
data/
└── <TICKER>/
    └── 5m/
        ├── YYYY-MM-DD.parquet
        ├── YYYY-MM-DD.csv.gz
        └── indicators/
            ├── YYYY-MM-DD.parquet
            └── YYYY-MM-DD.csv.gz
```

Additional backtest/diagnostic artifacts can be stored under:

```text
backtests/
└── <DATE>/
    └── <TICKER>/
```

The exact files produced depend on the execution path and available storage engine.

## Configuration

Ticker configuration is loaded from:

```text
tickers_config.json
```

The file contains market information and strategy profiles for individual tickers.

A minimal example is:

```json
{
  "AAPL": {
    "market": "US"
  }
}
```

The actual project configuration can contain additional strategy parameters.

## Environment Variables

API credentials are loaded from environment variables.

Create a `.env` file locally:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
API_KEYS=your_fmp_key_1,your_fmp_key_2
```

The application has its own lightweight `.env` loader and does not require `python-dotenv`.

### Important

Never commit real credentials to GitHub.

Add this to `.gitignore`:

```gitignore
.env
data/
backtests/
*.csv
*.csv.gz
*.parquet
active_trades.json
```

If a credential has ever been committed to a public repository, revoke/rotate it even if it has subsequently been removed from the latest version.

## Installation

The project uses Python and the following third-party packages directly:

```text
requests
pandas
numpy
aiohttp
pandas-ta
vectorbt
pytz
certifi
```

For Parquet storage, one of the following engines can be installed:

```text
pyarrow
```

or

```text
fastparquet
```

Example:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` for optional live integration. Keep the existing `tickers_config.json`; it contains the project's ticker profiles. The pinned dependency set was verified with **Python 3.12** on Linux. The bundled Windows TA-Lib wheel is a legacy artifact and is not required for this setup.

## Running

### Self-test

Check the environment and configured services. **This mode sends a Telegram message when configured**; run it only against a chat you control. A successful process exit alone does not establish signal quality:

```bash
python pro_trading_bot.py --selftest
```

### Backtest one ticker

```bash
python pro_trading_bot.py --backtest AAPL
```

Specify the number of historical days:

```bash
python pro_trading_bot.py --backtest AAPL --days 30
```

### Backtest all configured tickers

```bash
python pro_trading_bot.py --backtest_all
```

Example with a custom history window:

```bash
python pro_trading_bot.py --backtest_all --days 30
```

### Parameter tuning

```bash
python pro_trading_bot.py --autotune AAPL
```

For a custom history window:

```bash
python pro_trading_bot.py --autotune AAPL --days 30
```

### Real-time mode

Running the script without one of the command-line modes starts the real-time monitoring loop:

```bash
python pro_trading_bot.py
```

## Project Structure

A recommended repository layout is:

```text
trading-bot/
│
├── pro_trading_bot.py
├── tickers_config.json
├── README.md
├── .gitignore
├── .env                  # local only, not committed
│
├── data/                 # generated market/indicator data
├── backtests/            # generated backtest artifacts
├── signals.csv           # generated signals
└── active_trades.json    # generated active-trade state
```

## Engineering Aspects

The project was built around several practical engineering problems rather than only the trading rules:

- asynchronous HTTP requests with `asyncio` / `aiohttp`
- synchronous API access where required
- retry and failure handling
- rate-limit handling
- API-key rotation
- caching
- persistent local datasets
- duplicate removal and chronological sorting
- environment-based configuration
- structured logging
- diagnostic exports
- service heartbeat monitoring
- separation of live-data processing and backtesting paths

## Limitations

This project should not be interpreted as a production trading system.

Known limitations include:

- Strategy performance depends heavily on market regime and ticker.
- Backtest results depend on data quality and parameter assumptions.
- Slippage and commission are modelled assumptions, not guaranteed execution costs.
- Historical performance does not imply future performance.
- Parameter tuning can introduce overfitting if the same data is repeatedly used for optimization and evaluation.
- The system does not directly execute broker orders.
- Data availability and API limits can affect real-time operation.
- The project has been developed as a personal research project rather than as a regulated financial product.

## Technologies

```text
Python
├── asyncio
├── aiohttp
├── requests
├── pandas
├── NumPy
├── pandas-ta
├── vectorbt
├── JSON / CSV
├── Parquet
└── Telegram Bot API

External APIs
└── Financial Modeling Prep
```

## Project Status

**Personal research / portfolio project**

The project is actively developed and used to experiment with:

- intraday market-data processing
- technical-indicator based filtering
- Breakout + Retest signal generation
- backtesting
- parameter tuning
- monitoring and notifications

## Author

**Alexandr Dolmatov**

GitHub: https://github.com/AlexandrDolmatov

LinkedIn: https://www.linkedin.com/in/aleksandr-dolmatov-suai/

---

### Portfolio note

This project demonstrates practical experience with Python, external APIs, asynchronous programming, data processing, backtesting, algorithmic rule implementation, error handling, persistence, and automated notifications.

It is presented as an engineering/research project and does not claim profitable trading performance.
