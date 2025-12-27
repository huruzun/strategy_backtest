
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import platform

# Set font for matplotlib
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def xirr(cashflows, dates):
    """
    Calculate the Internal Rate of Return (XIRR) for a series of cashflows.
    """
    def xnpv(rate, cashflows, dates):
        if rate <= -1.0:
            return float('inf')
        d0 = dates[0]
        return sum([cf / (1 + rate) ** ((d - d0).days / 365.0) for cf, d in zip(cashflows, dates)])

    try:
        return optimize.newton(lambda r: xnpv(r, cashflows, dates), 0.1)
    except RuntimeError:
        return xnpv(0, cashflows, dates)

import pandas_datareader.data as web
import datetime

def get_data():
    tickers = ['SPY', 'TLT', 'GLD', 'GSG']
    data_frames = []
    print(f"Downloading data for {tickers} from Stooq (pandas_datareader)...")
    
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            # Stooq symbols for US stocks often need .US suffix, but pdr handles it if not found?
            # Actually pdr stooq expects 'SPY.US' usually, or just 'SPY' might work if unique.
            # Let's try adding .US explicitly to be safe as per Stooq convention, 
            # or rely on pdr's behavior. In test we used SPY.US.
            stooq_ticker = f"{ticker}.US"
            
            # Use stooq source
            df = web.DataReader(stooq_ticker, 'stooq', start, end)
            
            if not df.empty:
                # Stooq returns Open, High, Low, Close, Volume.
                # We need Close.
                # Stooq data is reverse ordered (newest first).
                df = df.sort_index()
                
                # Keep only Close
                df = df[['Close']]
                df = df.rename(columns={'Close': ticker})
                data_frames.append(df)
                print(f"  Success: {len(df)} rows, Start: {df.index[0].date()}")
            else:
                print(f"  Warning: Empty data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            # Continue to next
            
    if not data_frames:
        return pd.DataFrame()

    # Combine into one DataFrame using inner join to ensure all assets are available
    print("Merging data...")
    df_combined = pd.concat(data_frames, axis=1, join='inner')
    df_combined = df_combined.sort_index()
    return df_combined

def backtest_multi_asset():
    df = get_data()
    if df.empty:
        print("No data fetched.")
        return

    # Drop rows where any asset is missing (to ensure we can buy all 4)
    # Actually, if one asset is not available yet, should we wait?
    # The user wants to invest in ALL 4. So we must start when all 4 are available.
    df_clean = df.dropna()
    
    if df_clean.empty:
        print("No overlapping data found for all assets.")
        return

    start_date = df_clean.index[0]
    print(f"Backtest Start Date (all assets available): {start_date.date()}")
    
    # Simulation Variables
    holdings = {ticker: 0.0 for ticker in df_clean.columns}
    total_invested = 0.0
    cashflows = [] # (amount, date)
    
    portfolio_history = []
    
    # Iterate through days
    # We need to detect month changes to invest
    last_month = None
    
    print("Running backtest simulation...")
    
    for date, prices in df_clean.iterrows():
        # Monthly Investment Logic
        current_month = date.month
        if last_month is None or current_month != last_month:
            # Invest 1000 in each asset
            investment_per_asset = 1000
            total_invested += investment_per_asset * len(prices)
            
            # Record cashflow (negative for investment)
            # We assume investment happens at the CLOSE price of the first day of the month
            cashflows.append((-investment_per_asset * len(prices), date))
            
            for ticker in prices.index:
                price = prices[ticker]
                shares_bought = investment_per_asset / price
                holdings[ticker] += shares_bought
            
            last_month = current_month
        
        # Calculate Daily Portfolio Value
        current_value = 0.0
        asset_values = {}
        for ticker in prices.index:
            val = holdings[ticker] * prices[ticker]
            current_value += val
            asset_values[ticker] = val
        
        portfolio_history.append({
            'Date': date,
            'Total Value': current_value,
            'Total Invested': total_invested,
            **asset_values
        })
    
    # Create Results DataFrame
    results_df = pd.DataFrame(portfolio_history)
    results_df.set_index('Date', inplace=True)
    
    # Final Metrics
    final_value = results_df['Total Value'].iloc[-1]
    total_return = (final_value - total_invested) / total_invested
    
    # Add current value as positive cashflow for XIRR
    xirr_cashflows = [cf[0] for cf in cashflows] + [final_value]
    xirr_dates = [cf[1] for cf in cashflows] + [results_df.index[-1]]
    
    portfolio_xirr = xirr(xirr_cashflows, xirr_dates)
    
    # Max Drawdown
    results_df['Max Value'] = results_df['Total Value'].cummax()
    results_df['Drawdown'] = (results_df['Total Value'] - results_df['Max Value']) / results_df['Max Value']
    max_drawdown = results_df['Drawdown'].min()
    
    report = []
    report.append("="*40)
    report.append(f"多资产定投策略回测报告")
    report.append("="*40)
    report.append(f"回测区间: {start_date.date()} 至 {results_df.index[-1].date()}")
    report.append(f"包含资产: {list(df_clean.columns)}")
    report.append(f"总投入本金: {total_invested:,.2f}")
    report.append(f"期末总资产: {final_value:,.2f}")
    report.append(f"总收益率: {total_return:.2%}")
    report.append(f"年化收益率 (XIRR): {portfolio_xirr:.2%}")
    report.append(f"最大回撤: {max_drawdown:.2%}")
    
    # Asset Allocation at the end
    report.append("\n期末持仓分布:")
    for ticker in df_clean.columns:
        val = results_df[ticker].iloc[-1]
        pct = val / final_value
        report.append(f"  {ticker}: {val:,.2f} ({pct:.1%})")
        
    report_str = "\n".join(report)
    print(report_str)
    
    with open("multi_asset_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Total Value vs Invested
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index, results_df['Total Value'], label='总资产市值', color='red')
    plt.plot(results_df.index, results_df['Total Invested'], label='投入本金', color='blue', linestyle='--')
    plt.title('多资产定投策略：市值增长曲线')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Drawdown
    plt.subplot(2, 1, 2)
    plt.plot(results_df.index, results_df['Drawdown'], label='最大回撤', color='green')
    plt.fill_between(results_df.index, results_df['Drawdown'], 0, color='green', alpha=0.3)
    plt.title('投资组合回撤')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_asset_backtest.png')
    print("\n图表已保存至 multi_asset_backtest.png")
    # plt.show() # Can't show in this environment, but saving works.

if __name__ == "__main__":
    backtest_multi_asset()
