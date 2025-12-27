
import akshare as ak
import pandas as pd
import yfinance as yf

def test_akshare(symbol):
    print(f"Testing akshare for {symbol}...")
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="")
        if not df.empty:
            print(f"Success! {symbol} rows: {len(df)}")
            print(df.head())
            return True
    except Exception as e:
        print(f"akshare failed for {symbol}: {e}")
    return False

def test_yfinance(symbol):
    print(f"Testing yfinance for {symbol}...")
    try:
        df = yf.download(symbol, start="2000-01-01", progress=False)
        if not df.empty:
            print(f"Success! {symbol} rows: {len(df)}")
            print(df.head())
            return True
    except Exception as e:
        print(f"yfinance failed for {symbol}: {e}")
    return False

symbols = ["SPY", "TLT", "GLD", "GSG"]

print("--- Checking Data Sources ---")
use_akshare = True
for s in symbols:
    if not test_akshare(s):
        use_akshare = False
        break

if not use_akshare:
    print("\nSwitching to yfinance check...")
    for s in symbols:
        test_yfinance(s)
