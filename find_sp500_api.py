import akshare as ak
import pandas as pd

try:
    print("Trying stock_us_daily with .INX")
    df = ak.stock_us_daily(symbol=".INX")
    print(df.head())
except Exception as e:
    print(f"Failed .INX: {e}")

try:
    print("Trying stock_us_daily with SPX")
    df = ak.stock_us_daily(symbol="SPX")
    print(df.head())
except Exception as e:
    print(f"Failed SPX: {e}")
