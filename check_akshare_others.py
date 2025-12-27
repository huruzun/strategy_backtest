
import akshare as ak
import pandas as pd

def check_one(symbol):
    print(f"Checking {symbol}...")
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="")
        if not df.empty:
            print(f"SUCCESS {symbol}: {len(df)} rows. Start: {df['date'].iloc[0]}")
        else:
            print(f"FAIL {symbol}: Empty")
    except Exception as e:
        print(f"ERROR {symbol}: {e}")

check_one("GLD")
check_one("GSG")
