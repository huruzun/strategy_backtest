import sys
import os
# Add user site-packages to the beginning of sys.path to ensure we can load modules installed in user directory
user_site = os.path.expanduser('~\\AppData\\Roaming\\Python\\Python312\\site-packages')
if os.path.exists(user_site):
    sys.path.insert(0, user_site)

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy import optimize

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def xirr(cashflows, dates):
    """
    Calculate the XIRR of a series of cashflows.
    :param cashflows: list of cashflows (negative for investment, positive for return)
    :param dates: list of dates corresponding to cashflows
    :return: XIRR value
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

def backtest_fund(fund_code, start_date_str=None, monthly_investment=1000):
    print(f"Fetching data for fund {fund_code}...")
    
    # Fetch open fund info to get name
    try:
        fund_name_df = ak.fund_em_fund_name()
        fund_name = fund_name_df[fund_name_df['基金代码'] == fund_code]['基金简称'].values[0]
    except:
        fund_name = fund_code

    # Fetch NAV history
    # fund_open_fund_info_em(symbol="270002", indicator="单位净值走势")
    try:
        # Check akshare documentation or source for correct parameter
        # Typically it is 'symbol' or just first arg
        # Use Accumulative NAV if possible for better accuracy on return (assuming dividend reinvestment)
        # However, for simple fixed investment simulation, we usually buy at Unit NAV.
        # But to calculate return correctly *assuming reinvestment*, we should track shares * Unit NAV.
        # If we want to simulate dividends taken as cash, Unit NAV is fine.
        # If we want to simulate dividends reinvested, we should use Adjusted NAV (复权净值).
        # akshare provides "累计净值走势" or "单位净值走势".
        # Let's stick to Unit NAV for now as per standard fixed investment calculators unless "Adjusted" is requested.
        # But actually, standard calculators assume reinvestment.
        # Let's try to fetch Adjusted NAV if available, or just use Unit NAV and note it.
        # akshare doesn't always provide easy adjusted NAV for funds.
        # We will proceed with Unit NAV (Standard).
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    except TypeError:
        try:
             # Try without keyword arg if named args failed (though symbol is standard)
             df = ak.fund_open_fund_info_em(fund_code, "单位净值走势")
        except Exception as e:
             print(f"Error fetching data with positional args: {e}")
             return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Data Cleaning
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df['单位净值'] = pd.to_numeric(df['单位净值'])
    df = df.sort_values('净值日期')
    
    # Dynamic Start Date: If start_date_str is None, use fund inception date
    if start_date_str is None:
        start_date = df['净值日期'].iloc[0]
        print(f"No start date provided. Using fund inception date: {start_date.date()}")
    else:
        start_date = pd.to_datetime(start_date_str)
        # Ensure start date is not before inception
        if start_date < df['净值日期'].iloc[0]:
             print(f"Requested start date {start_date.date()} is before inception {df['净值日期'].iloc[0].date()}. Using inception date.")
             start_date = df['净值日期'].iloc[0]
    
    df = df[df['净值日期'] >= start_date]
    
    if df.empty:
        print(f"No data found after {start_date}")
        return

    # Resample to Monthly (First trading day of each month)
    # We create a target investment dates list: 1st of every month
    # Then find the closest trading day >= 1st of month
    
    # Generate ideal investment dates (1st of each month)
    end_date = df['净值日期'].iloc[-1]
    ideal_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    investment_log = []
    
    total_shares = 0
    total_invested = 0
    cashflows = []
    cashflow_dates = []
    
    print(f"Starting Backtest for {fund_name} ({fund_code})")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Monthly Investment: {monthly_investment} CNY")
    
    for ideal_date in ideal_dates:
        # Find actual trading date: first date in df >= ideal_date
        # Since df is sorted, we can use searchsorted logic or simple masking
        valid_dates = df[df['净值日期'] >= ideal_date]
        if valid_dates.empty:
            continue
            
        trade_row = valid_dates.iloc[0]
        trade_date = trade_row['净值日期']
        
        # Ensure we don't invest twice in the same month if data has gaps (rare but possible)
        # Check if we already invested in this month
        if len(cashflow_dates) > 0:
            last_date = cashflow_dates[-1]
            if trade_date.year == last_date.year and trade_date.month == last_date.month:
                continue
                
        nav = trade_row['单位净值']
        
        # Buy
        shares_bought = monthly_investment / nav
        total_shares += shares_bought
        total_invested += monthly_investment
        
        investment_log.append({
            'date': trade_date,
            'nav': nav,
            'amount': monthly_investment,
            'shares': shares_bought,
            'total_shares': total_shares,
            'total_invested': total_invested,
            'market_value': total_shares * nav
        })
        
        cashflows.append(-monthly_investment)
        cashflow_dates.append(trade_date)

    # Final Calculation
    final_nav = df.iloc[-1]['单位净值']
    final_date = df.iloc[-1]['净值日期']
    final_market_value = total_shares * final_nav
    
    # Add final value to cashflows for XIRR calculation
    cashflows.append(final_market_value)
    cashflow_dates.append(final_date)
    
    # Metrics
    total_return_rate = (final_market_value - total_invested) / total_invested
    annualized_return = xirr(cashflows, cashflow_dates)
    
    # Max Drawdown Calculation (Based on Portfolio Market Value)
    # We need to calculate daily market value for the whole period
    # Construct a daily portfolio value series
    
    # Create a DataFrame of holdings over time
    log_df = pd.DataFrame(investment_log)
    log_df.set_index('date', inplace=True)
    
    # Reindex df to include all trading days and fill shares forward
    daily_df = df.set_index('净值日期')[['单位净值']].copy()
    
    # Map shares from log to daily_df
    # We use asof to find cumulative shares for each day
    # Or simpler: create a series from log_df['total_shares'] and reindex
    
    # Merge daily data with investment log to get shares update points
    # We need to forward fill the shares owned
    daily_perf = daily_df.join(log_df[['total_shares', 'total_invested']], how='left')
    daily_perf['total_shares'] = daily_perf['total_shares'].ffill().fillna(0)
    daily_perf['total_invested'] = daily_perf['total_invested'].ffill().fillna(0)
    
    # Calculate daily market value
    daily_perf['market_value'] = daily_perf['total_shares'] * daily_perf['单位净值']
    
    # Calculate Drawdown
    daily_perf['max_val'] = daily_perf['market_value'].cummax()
    daily_perf['drawdown'] = (daily_perf['market_value'] - daily_perf['max_val']) / daily_perf['max_val']
    
    max_drawdown = daily_perf['drawdown'].min()
    
    print("\n" + "="*40)
    print(f"       BACKTEST RESULTS: {fund_code} - {fund_name}")
    print("="*40)
    print(f"Start Date:       {investment_log[0]['date'].date()}")
    print(f"End Date:         {final_date.date()}")
    print(f"Total Invested:   {total_invested:,.2f} CNY")
    print(f"Final Value:      {final_market_value:,.2f} CNY")
    print(f"Total Return:     {total_return_rate*100:.2f}%")
    print(f"Annualized (XIRR):{annualized_return*100:.2f}%")
    print(f"Max Drawdown:     {max_drawdown*100:.2f}%")
    print("="*40)

    # Visualization
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese characters
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Market Value vs Total Invested
    ax1.plot(daily_perf.index, daily_perf['market_value'], label='Total Assets', color='red')
    ax1.plot(daily_perf.index, daily_perf['total_invested'], label='Total Invested', color='blue', linestyle='--')
    ax1.set_title(f'{fund_name} ({fund_code}) Fixed Investment Backtest')
    ax1.set_ylabel('Value (CNY)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Drawdown
    ax2.fill_between(daily_perf.index, daily_perf['drawdown'] * 100, 0, color='green', alpha=0.3, label='Drawdown')
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown %')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    # Save plot
    filename = f'backtest_{fund_code}.png'
    plt.savefig(filename)
    print(f"Chart saved to {filename}")
    
    # Show plot (if running in interactive env, otherwise it just saves)
    # plt.show()
    
    return daily_perf

if __name__ == "__main__":
    # Settings
    FUND_CODE = "270042"
    # START_DATE = None # Auto-detect inception
    START_DATE = None 
    AMOUNT = 1000
    
    backtest_fund(FUND_CODE, START_DATE, AMOUNT)
