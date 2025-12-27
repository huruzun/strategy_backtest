
import akshare as ak
import pandas as pd
import time

def check_akshare(symbol):
    print(f"Checking {symbol} via akshare...")
    try:
        # stock_us_daily gets daily data
        df = ak.stock_us_daily(symbol=symbol, adjust="")
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            print(f"  {symbol}: Found {len(df)} rows. Range: {start_date.date()} to {end_date.date()}")
            return True
        else:
            print(f"  {symbol}: Empty dataframe.")
    except Exception as e:
        print(f"  {symbol}: Error {e}")
    return False

symbols = ["SPY", "TLT", "GLD", "GSG", "DBC"]

for s in symbols:
    check_akshare(s)
    time.sleep(1)
