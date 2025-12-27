import akshare as ak
import pandas as pd
import datetime
import time

def fetch_with_retry(func, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"Error calling {func.__name__}: {e}. Retry {i+1}/{max_retries}")
            time.sleep(2)
    return None

def fetch_china_indices():
    # 1. 沪深300
    print("Fetching CSI 300...")
    hs300 = fetch_with_retry(ak.stock_zh_index_daily, symbol="sh000300")
    if hs300 is not None:
        hs300 = hs300[['date', 'close']].rename(columns={'close': 'HS300'})
    
    # 2. 中证500
    print("Fetching CSI 500...")
    zz500 = fetch_with_retry(ak.stock_zh_index_daily, symbol="sh000905")
    if zz500 is not None:
        zz500 = zz500[['date', 'close']].rename(columns={'close': 'ZZ500'})
    
    # 3. 创业板指
    print("Fetching ChiNext...")
    cyb = fetch_with_retry(ak.stock_zh_index_daily, symbol="sz399006")
    if cyb is not None:
        cyb = cyb[['date', 'close']].rename(columns={'close': 'CYB'})
    
    # 4. 中债-国债总指数
    print("Fetching CSI Treasury Bond Index...")
    bond = fetch_with_retry(ak.stock_zh_index_daily, symbol="sh000012")
    if bond is not None:
        bond = bond[['date', 'close']].rename(columns={'close': 'Bond'})
    
    # 5. 黄金 - 使用期货主力合约作为代理 (AU0)
    print("Fetching Gold (Futures AU0)...")
    gold = fetch_with_retry(ak.futures_main_sina, symbol="AU0")
    if gold is not None:
        gold = gold[['日期', '收盘价']].rename(columns={'日期': 'date', '收盘价': 'Gold'})
        gold['date'] = pd.to_datetime(gold['date']).dt.date
    
    # 合并数据
    dfs = []
    for name, df in [('HS300', hs300), ('ZZ500', zz500), ('CYB', cyb), ('Bond', bond), ('Gold', gold)]:
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.date
            dfs.append(df.set_index('date'))
        else:
            print(f"Warning: Data for {name} is missing.")
    
    if dfs:
        merged = pd.concat(dfs, axis=1).sort_index()
        # 过滤2005年以来的数据
        start_date = datetime.date(2005, 1, 1)
        merged = merged[merged.index >= start_date]
        
        # 填充缺失值
        merged = merged.ffill()
        
        merged.to_csv('china_assets_raw.csv')
        print(f"\nData fetched and saved to china_assets_raw.csv. Shape: {merged.shape}")
        print(f"Date range: {merged.index.min()} to {merged.index.max()}")
        print("\nColumns available:", merged.columns.tolist())
    else:
        print("Failed to fetch any data.")

if __name__ == "__main__":
    fetch_china_indices()
