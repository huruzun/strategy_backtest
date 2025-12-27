import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from scipy import optimize
import platform
import os

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
    def xnpv(rate, cashflows, dates):
        if rate <= -1.0: return float('inf')
        d0 = dates[0]
        return sum([cf / (1 + rate) ** ((d - d0).days / 365.0) for cf, d in zip(cashflows, dates)])
    try:
        return optimize.newton(lambda r: xnpv(r, cashflows, dates), 0.1)
    except:
        return 0.0

def backtest_all_weather():
    # Define target allocation
    # S&P 500: 30%, Long-term Bonds: 30%, TIPS: 30%, Gold: 5%, Commodities: 5%
    allocation = {
        'SPY': 0.30,
        'TLT': 0.30,
        'TIP': 0.30,
        'GLD': 0.05,
        'GSG': 0.05
    }
    
    tickers = list(allocation.keys())
    print(f"Fetching data for {tickers} from Stooq...")
    
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    
    data_frames = []
    for ticker in tickers:
        try:
            df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
            if not df.empty:
                df = df.sort_index()
                df = df[['Close']].rename(columns={'Close': ticker})
                data_frames.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    if not data_frames:
        print("No data fetched.")
        return

    # Merge all data
    df_merged = pd.concat(data_frames, axis=1, join='inner').sort_index()
    
    # Save raw data to CSV for verification
    raw_data_path = 'all_weather_raw_data.csv'
    df_merged.to_csv(raw_data_path)
    print(f"Raw historical data saved to {raw_data_path}")
    
    # Backtest variables
    initial_investment = 10000.0 # Initial lump sum
    monthly_investment = 1000.0  # Monthly contribution
    
    # Track holdings (shares)
    shares = {ticker: 0.0 for ticker in tickers}
    total_invested = 0.0
    cashflows = []
    
    history = []
    
    # Current portfolio value
    current_value = 0.0
    
    last_year = None
    last_month = None
    
    print("Running backtest with annual rebalancing...")
    
    for date, row in df_merged.iterrows():
        # 1. Monthly Investment (First trading day of each month)
        if date.month != last_month:
            # Re-calculate current portfolio value
            current_value = sum(shares[t] * row[t] for t in tickers)
            
            # Invest new money (initial or monthly)
            amount_to_invest = initial_investment if total_invested == 0 else monthly_investment
            total_invested += amount_to_invest
            cashflows.append((-amount_to_invest, date))
            
            # Distribute new money according to target allocation
            for ticker in tickers:
                invest_amount = amount_to_invest * allocation[ticker]
                shares[ticker] += invest_amount / row[ticker]
            
            last_month = date
            
        # 2. Annual Rebalancing (First trading day of each year, excluding the very first year start)
        if date.year != last_year and last_year is not None:
            # Calculate current total value
            current_value = sum(shares[t] * row[t] for t in tickers)
            print(f"[{date.date()}] Rebalancing portfolio... Total Value: {current_value:,.2f}")
            
            # Re-allocate total value according to target
            for ticker in tickers:
                target_value = current_value * allocation[ticker]
                shares[ticker] = target_value / row[ticker]
            
            last_year = date.year
        elif last_year is None:
            last_year = date.year
            
        # 3. Daily Tracking
        current_value = sum(shares[t] * row[t] for t in tickers)
        history.append({
            'date': date,
            'total_value': current_value,
            'total_invested': total_invested
        })
        
    # Final state
    final_date = df_merged.index[-1]
    final_value = current_value
    cashflows.append((final_value, final_date))
    
    hist_df = pd.DataFrame(history).set_index('date')
    
    # Metrics
    total_return = (final_value - total_invested) / total_invested
    
    cf_amounts = [c[0] for c in cashflows]
    cf_dates = [c[1] for c in cashflows]
    annual_return = xirr(cf_amounts, cf_dates)
    
    hist_df['max_val'] = hist_df['total_value'].cummax()
    hist_df['drawdown'] = (hist_df['total_value'] - hist_df['max_val']) / hist_df['max_val']
    max_drawdown = hist_df['drawdown'].min()
    
    # Report
    print("\n" + "="*40)
    print("全天候策略 (All Weather Strategy) 回测报告")
    print("="*40)
    print(f"起始日期: {df_merged.index[0].date()}")
    print(f"结束日期: {final_date.date()}")
    print(f"资产配置: {allocation}")
    print(f"总投入本金: {total_invested:,.2f}")
    print(f"期末总资产: {final_value:,.2f}")
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"年化收益率 (XIRR): {annual_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print("="*40)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(hist_df.index, hist_df['total_value'], label='策略总资产', color='blue')
    plt.plot(hist_df.index, hist_df['total_invested'], label='投入本金', linestyle='--', color='gray')
    plt.title('全天候策略定投回测 (All Weather Strategy Backtest)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(hist_df.index, hist_df['drawdown'], label='回撤', color='red')
    plt.fill_between(hist_df.index, hist_df['drawdown'], 0, color='red', alpha=0.3)
    plt.title('回撤曲线 (Drawdown)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('all_weather_backtest.png')
    print("Backtest chart saved to all_weather_backtest.png")

if __name__ == "__main__":
    backtest_all_weather()
