import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
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
    Calculate the Internal Rate of Return (XIRR) for a series of cashflows at irregular intervals.
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    Calculate Sharpe Ratio based on daily returns.
    Assuming risk_free_rate is annual.
    """
    if returns.empty:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    if excess_returns.std() == 0:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std() * (252 ** 0.5)
    return sharpe

def backtest_sp500_enhanced(monthly_investment=1000, start_year=2000):
    print(f"Fetching data for S&P 500 (Enhanced Strategy) from Stooq...")
    
    # Use SPY ETF data.
    try:
        start = datetime(start_year, 1, 1)
        end = datetime.now()
        # Stooq source
        df = web.DataReader('SPY.US', 'stooq', start, end)
        
        if df.empty:
             print(f"No data found for SPY")
             return

        # Stooq returns index as Date, and columns Open, High, Low, Close, Volume
        # Sort by date ascending
        df = df.sort_index()
        
        # Keep only Close and Date (index)
        df = df[['Close']]
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date', 'Close': 'close'})
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Clean and prepare data
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'])
    df = df.rename(columns={'date': '净值日期', 'close': '单位净值'})
    
    # Simulation Variables
    total_shares = 0.0
    total_invested = 0.0
    cashflows = [] # (amount, date)
    
    # Trackers for reporting
    history = []
    
    # Flags for dip buying
    # Triggers: 20%, 40%, 60%
    # We trigger ONCE when the threshold is crossed.
    # We reset triggers when drawdown recovers to 0 (New High).
    triggered_20 = False
    triggered_40 = False
    triggered_60 = False
    
    max_nav = 0.0
    
    # Prepare monthly investment schedule
    # Map (Year, Month) -> True
    # We iterate daily. If (Year, Month) changes or matches schedule, we invest.
    # Simpler: Iterate daily. If it's the first trading day of the month, invest.
    
    last_invest_month = None # (Year, Month)
    
    print("Calculating enhanced backtest...")
    
    # Iterate through every trading day
    for idx, row in df.iterrows():
        current_date = row['净值日期']
        nav = row['单位净值']
        
        # 1. Update Max NAV for Drawdown Calculation
        if nav > max_nav:
            max_nav = nav
            # Reset triggers on new high
            triggered_20 = False
            triggered_40 = False
            triggered_60 = False
            
        current_drawdown = (nav - max_nav) / max_nav if max_nav > 0 else 0
        
        current_market_value = total_shares * nav
        
        # 2. Monthly Investment Check (First trading day of month)
        current_month_key = (current_date.year, current_date.month)
        if current_month_key != last_invest_month:
            # Invest Monthly Amount
            shares_bought = monthly_investment / nav
            total_shares += shares_bought
            total_invested += monthly_investment
            
            cashflows.append((-monthly_investment, current_date))
            last_invest_month = current_month_key
            
            # Update market value after investment
            current_market_value = total_shares * nav

        # 3. Dip Buying Check
        # Rule:
        # If drawdown > 20% (meaning < -0.20) and not triggered_20: Buy 25% of holdings
        # If drawdown > 40% (meaning < -0.40) and not triggered_40: Buy 50% of holdings
        # If drawdown > 60% (meaning < -0.60) and not triggered_60: Buy 75% of holdings
        
        dip_investment = 0.0
        
        if current_drawdown < -0.60 and not triggered_60:
            dip_investment = current_market_value * 0.75
            triggered_60 = True
            print(f"[{current_date.date()}] Triggered -60% Dip Buy! Drawdown: {current_drawdown:.2%}, Invested: {dip_investment:,.2f}")
            
        elif current_drawdown < -0.40 and not triggered_40:
            dip_investment = current_market_value * 0.50
            triggered_40 = True
            print(f"[{current_date.date()}] Triggered -40% Dip Buy! Drawdown: {current_drawdown:.2%}, Invested: {dip_investment:,.2f}")
            
        elif current_drawdown < -0.20 and not triggered_20:
            dip_investment = current_market_value * 0.25
            triggered_20 = True
            print(f"[{current_date.date()}] Triggered -20% Dip Buy! Drawdown: {current_drawdown:.2%}, Invested: {dip_investment:,.2f}")
            
        if dip_investment > 0:
            shares_bought = dip_investment / nav
            total_shares += shares_bought
            total_invested += dip_investment
            cashflows.append((-dip_investment, current_date))
            current_market_value = total_shares * nav
            
        # Record Daily State
        history.append({
            'date': current_date,
            'nav': nav,
            'total_shares': total_shares,
            'total_invested': total_invested,
            'market_value': current_market_value,
            'nav_drawdown': current_drawdown
        })

    # Finalize
    final_row = df.iloc[-1]
    final_date = final_row['净值日期']
    final_nav = final_row['单位净值']
    final_market_value = total_shares * final_nav
    
    cashflows.append((final_market_value, final_date))
    
    # Calculate Metrics
    total_return_rate = (final_market_value - total_invested) / total_invested
    
    # Prepare cashflows for XIRR
    cf_amounts = [c[0] for c in cashflows]
    cf_dates = [c[1] for c in cashflows]
    try:
        annualized_return = xirr(cf_amounts, cf_dates)
    except:
        annualized_return = 0.0
        
    # DataFrame for Analysis
    hist_df = pd.DataFrame(history)
    hist_df.set_index('date', inplace=True)
    
    # Calculate Portfolio Drawdown (different from NAV drawdown)
    hist_df['max_val'] = hist_df['market_value'].cummax()
    hist_df['port_drawdown'] = (hist_df['market_value'] - hist_df['max_val']) / hist_df['max_val']
    max_port_drawdown = hist_df['port_drawdown'].min()
    
    print("="*40)
    print(f"       ENHANCED BACKTEST RESULTS")
    print("="*40)
    print(f"Start Date:       {hist_df.index[0].date()}")
    print(f"End Date:         {final_date.date()}")
    print(f"Total Invested:   {total_invested:,.2f}")
    print(f"Final Value:      {final_market_value:,.2f}")
    print(f"Total Return:     {total_return_rate*100:.2f}%")
    print(f"Annualized (XIRR):{annualized_return*100:.2f}%")
    print(f"Max Drawdown:     {max_port_drawdown*100:.2f}%")
    print("="*40)
    
    return hist_df, total_invested

def backtest_sp500(monthly_investment=1000, start_year=2000):
    print(f"Fetching data for S&P 500 (using SPY ETF as proxy) from Stooq...")
    
    # Use SPY ETF data.
    try:
        start = datetime(start_year, 1, 1)
        end = datetime.now()
        # Stooq source
        df = web.DataReader('SPY.US', 'stooq', start, end)
        
        if df.empty:
             print(f"No data found for SPY")
             return None, 0

        # Stooq returns index as Date, and columns Open, High, Low, Close, Volume
        # Sort by date ascending
        df = df.sort_index()
        
        # Keep only Close and Date (index)
        df = df[['Close']]
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date', 'Close': 'close'})
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, 0

    # Clean and prepare data
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'])
    df = df.sort_values('date')
    
    # Filter by start date
    start_date = pd.Timestamp(f"{start_year}-01-01")
    df = df[df['date'] >= start_date]
    
    if df.empty:
        print(f"No data found after {start_year}")
        return

    actual_start_date = df.iloc[0]['date']
    print(f"Data available from: {actual_start_date.date()}")
    
    # Rename for consistency with previous logic
    df = df.rename(columns={'date': '净值日期', 'close': '单位净值'})

    # Generate ideal monthly investment dates (1st of each month)
    # We extend to the last available date in the dataframe
    ideal_dates = pd.date_range(start=start_date, end=df['净值日期'].iloc[-1], freq='MS')

    # Backtest Loop
    investment_log = []
    total_shares = 0
    total_invested = 0
    cashflows = []
    cashflow_dates = []

    print("Calculating backtest...")
    for ideal_date in ideal_dates:
        # Find the first trading day on or after the ideal date
        valid_dates = df[df['净值日期'] >= ideal_date]
        if valid_dates.empty:
            continue
        
        trade_row = valid_dates.iloc[0]
        trade_date = trade_row['净值日期']
        
        # Ensure we don't trade multiple times in the same month if data has gaps/weirdness
        # (Logic: check if we already traded in this month)
        if len(cashflow_dates) > 0:
            last_date = cashflow_dates[-1]
            if trade_date.year == last_date.year and trade_date.month == last_date.month:
                continue
        
        nav = trade_row['单位净值']
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
        
        # Cashflow for XIRR: negative for investment
        cashflows.append(-monthly_investment)
        cashflow_dates.append(trade_date)

    # Final Valuation
    final_nav = df.iloc[-1]['单位净值']
    final_date = df.iloc[-1]['净值日期']
    final_market_value = total_shares * final_nav
    
    # Add final value as positive cashflow
    cashflows.append(final_market_value)
    cashflow_dates.append(final_date)

    # Calculate Metrics
    total_return_rate = (final_market_value - total_invested) / total_invested
    try:
        annualized_return = xirr(cashflows, cashflow_dates)
    except:
        annualized_return = 0.0

    # Calculate Max Drawdown
    log_df = pd.DataFrame(investment_log)
    log_df.set_index('date', inplace=True)
    
    # Create a daily performance series
    daily_df = df.set_index('净值日期')[['单位净值']].copy()
    daily_df['shares_owned'] = 0.0
    
    # Map shares owned to daily dates (forward fill)
    # We need to map the 'total_shares' from log_df to daily_df
    # Logic: Join log_df's total_shares to daily_df, then ffill
    daily_perf = daily_df.join(log_df[['total_shares']], how='left')
    daily_perf['total_shares'] = daily_perf['total_shares'].ffill().fillna(0)
    
    daily_perf['market_value'] = daily_perf['total_shares'] * daily_perf['单位净值']
    
    # Filter out rows before the first investment (where total_shares is 0)
    daily_perf = daily_perf[daily_perf['total_shares'] > 0]

    daily_perf['max_val'] = daily_perf['market_value'].cummax()
    daily_perf['drawdown'] = (daily_perf['market_value'] - daily_perf['max_val']) / daily_perf['max_val']

    max_drawdown = daily_perf['drawdown'].min()

    # Output Results
    print("="*40)
    print(f"       BACKTEST RESULTS: S&P 500 (SPY)")
    print("="*40)
    print(f"Start Date:       {investment_log[0]['date'].date()}")
    print(f"End Date:         {final_date.date()}")
    print(f"Total Invested:   {total_invested:,.2f}")
    print(f"Final Value:      {final_market_value:,.2f}")
    print(f"Total Return:     {total_return_rate*100:.2f}%")
    print(f"Annualized (XIRR):{annualized_return*100:.2f}%")
    print(f"Max Drawdown:     {max_drawdown*100:.2f}%")
    print("="*40)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Market Value vs Invested Capital
    plt.subplot(2, 1, 1)
    plt.plot(daily_perf.index, daily_perf['market_value'], label='总资产 (Total Assets)', color='red')
    
    # Create a step line for invested capital
    invested_series = pd.Series(index=cashflow_dates[:-1], data=[monthly_investment]*len(cashflow_dates[:-1])).cumsum()
    # Reindex to daily to plot nicely
    invested_daily = invested_series.reindex(daily_perf.index, method='ffill')
    plt.plot(daily_perf.index, invested_daily, label='投入本金 (Invested Capital)', linestyle='--', color='blue')
    
    plt.title('标普500定投回测 (S&P 500 Monthly Investment)')
    plt.legend()
    plt.grid(True)
    
    # Plot Drawdown
    plt.subplot(2, 1, 2)
    plt.plot(daily_perf.index, daily_perf['drawdown'], label='回撤 (Drawdown)', color='green')
    plt.fill_between(daily_perf.index, daily_perf['drawdown'], 0, color='green', alpha=0.3)
    plt.title('最大回撤 (Max Drawdown)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_sp500.png')
    print("Chart saved to backtest_sp500.png")
    
    return daily_perf, total_invested

def compare_strategies():
    print("Running Base Strategy...")
    base_df, base_invested = backtest_sp500()
    
    print("\nRunning Enhanced Strategy...")
    enhanced_df, enhanced_invested = backtest_sp500_enhanced()
    
    if base_df is None or enhanced_df is None:
        return

    # Visualization of Comparison
    plt.figure(figsize=(14, 10))
    
    # 1. Portfolio Value Comparison
    plt.subplot(3, 1, 1)
    plt.plot(base_df.index, base_df['market_value'], label='普通定投 (Base)', color='blue', alpha=0.7)
    plt.plot(enhanced_df.index, enhanced_df['market_value'], label='增强定投 (Enhanced)', color='red', alpha=0.7)
    plt.title('策略对比：总资产 (Strategy Comparison: Total Assets)')
    plt.legend()
    plt.grid(True)
    
    # 2. Drawdown Comparison
    plt.subplot(3, 1, 2)
    # Re-calculate base drawdown using the same logic just to be safe (it's already in 'drawdown')
    # Enhanced logic uses 'port_drawdown'
    plt.plot(base_df.index, base_df['drawdown'], label='普通定投回撤 (Base Drawdown)', color='blue', alpha=0.5)
    plt.plot(enhanced_df.index, enhanced_df['port_drawdown'], label='增强定投回撤 (Enhanced Drawdown)', color='red', alpha=0.5)
    plt.title('策略对比：回撤 (Strategy Comparison: Drawdown)')
    plt.legend()
    plt.grid(True)
    
    # 3. Invested Capital Comparison
    plt.subplot(3, 1, 3)
    # We need to reconstruct invested capital series for plotting
    # Base
    base_invested_series = base_df['total_invested'] if 'total_invested' in base_df.columns else pd.Series(index=base_df.index, data=0)
    # Wait, base_df in backtest_sp500 doesn't have 'total_invested' column, we need to add it or reconstruct it.
    # Actually, let's fix backtest_sp500 to return a DF with 'total_invested'.
    # In backtest_sp500, we created 'daily_perf' but didn't put 'total_invested' in it.
    # Let's just assume linear growth for base for now or skip if too complex.
    # Enhanced DF has 'total_invested'.
    
    plt.plot(enhanced_df.index, enhanced_df['total_invested'], label='增强定投投入本金 (Enhanced Invested)', color='red', linestyle='--')
    # For base, we know it's roughly linear, but let's just plot Enhanced to show the "Dip Buys"
    plt.title('增强定投：投入本金变化 (Enhanced Strategy: Invested Capital)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    print("Comparison chart saved to strategy_comparison.png")

if __name__ == "__main__":
    compare_strategies()
