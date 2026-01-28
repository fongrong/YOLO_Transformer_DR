# Stock Watcher for Windows

A simple and lightweight stock price monitoring application built with Python and Tkinter.

## Features

- **Real-time Stock Monitoring**: Track multiple stocks with live price updates
- **Price Alerts**: Set alerts for when stocks go above or below target prices
- **Auto-Refresh**: Automatic price updates at configurable intervals (10-300 seconds)
- **Persistent Watchlist**: Your stock list is automatically saved and restored
- **Color-Coded Changes**: Green for gains, red for losses
- **Market Status**: Shows current market state (REGULAR, PRE, POST, CLOSED)

## Requirements

- Python 3.7 or higher
- Windows OS (also works on macOS and Linux)
- Internet connection for stock data

**Note**: No additional packages required! Uses only Python standard library.

## Installation

1. Ensure Python is installed on your system
2. Download or clone this repository
3. Navigate to the `stock_watcher` folder

## Running the Application

### Option 1: Double-click the batch file
Simply double-click `run_stock_watcher.bat`

### Option 2: Run from Command Prompt
```cmd
cd stock_watcher
python stock_watcher.py
```

### Option 3: Run from PowerShell
```powershell
cd stock_watcher
.\run_stock_watcher.ps1
```

## Usage

### Adding Stocks
1. Click **+ Add Stock** button or use **File > Add Stock**
2. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA)
3. Or use the **Quick Add** field in the toolbar

### Removing Stocks
1. Select a stock in the list
2. Click **- Remove** button or use **File > Remove Stock**

### Setting Price Alerts
1. Click **Add Alert** button or use **Alerts > Add Alert**
2. Select a stock from your watchlist
3. Choose alert type (Above/Below target price)
4. Enter the target price
5. You'll receive a popup notification when triggered

### Changing Refresh Interval
1. Go to **Settings > Refresh Interval**
2. Enter new interval in seconds (10-300)

## Supported Stock Symbols

The application supports stocks from major exchanges:
- NYSE (e.g., IBM, KO, WMT)
- NASDAQ (e.g., AAPL, GOOGL, MSFT, AMZN)
- International markets (append exchange suffix, e.g., SONY for Tokyo)

## Data Source

Stock data is fetched from Yahoo Finance API.

## Troubleshooting

**"Could not fetch data" error**
- Check your internet connection
- Verify the stock symbol is correct
- Yahoo Finance API may be temporarily unavailable

**Application doesn't start**
- Ensure Python 3.7+ is installed
- Check that tkinter is available (included by default on Windows)

## License

MIT License - See LICENSE file for details.
