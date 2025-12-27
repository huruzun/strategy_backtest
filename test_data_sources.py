
import pandas_datareader.data as web
import akshare as ak
import datetime
import pandas as pd

def test_stooq(symbol):
    print(f"\n--- Testing Stooq via pandas_datareader for {symbol} ---")
    try:
        start = datetime.datetime(2000, 1, 1)
        end = datetime.datetime.now()
        # Stooq uses ticker.US for US stocks
        # e.g. SPY.US
        ticker = f"{symbol}.US"
        df = web.DataReader(ticker, 'stooq', start, end)
        if not df.empty:
            print(f"Success! {len(df)} rows.")
            print(f"Range: {df.index.min().date()} to {df.index.max().date()}")
            print(df.head())
            return True
        else:
            print("Empty DataFrame.")
    except Exception as e:
        print(f"Error: {e}")
    return False

def test_akshare_sina(symbol):
    print(f"\n--- Testing Akshare (Sina Source) for {symbol} ---")
    try:
        # stock_us_hist is typically Sina
        # Note: akshare API names change often. 
        # stock_us_hist(symbol='105.TLT') ? Sina usually needs a prefix.
        # But akshare wraps this.
        # Let's try standard symbol first.
        
        # Trying stock_us_daily again but checking if it IS Sina
        # stock_us_daily documentation says "Source: Sina Finance"
        # If it returns 2016 for TLT, then Sina might only have 2016.
        
        # Let's try another function: stock_us_hist
        # Searching available functions...
        pass 
    except Exception as e:
        print(f"Error: {e}")

# Run Tests
# 1. Stooq
test_stooq("SPY")
test_stooq("TLT")
test_stooq("GLD")
test_stooq("GSG")

# 2. Akshare
# We already know stock_us_daily works but has short history for TLT.
# Let's see if we can find another one.
