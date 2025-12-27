import pandas_datareader.data as web
import datetime
import pandas as pd

def check_all_weather_data():
    tickers = {
        'SPY': 'S&P 500',
        'TLT': 'Long-term Bonds',
        'TIP': 'TIPS',
        'GLD': 'Gold',
        'GSG': 'Commodities'
    }
    
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    
    results = []
    for ticker, name in tickers.items():
        try:
            print(f"Fetching {ticker} ({name})...")
            df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
            if not df.empty:
                results.append({
                    'Ticker': ticker,
                    'Start': df.index.min().date(),
                    'End': df.index.max().date(),
                    'Rows': len(df)
                })
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            
    res_df = pd.DataFrame(results)
    print("\nData Availability Summary:")
    print(res_df)

if __name__ == "__main__":
    check_all_weather_data()
