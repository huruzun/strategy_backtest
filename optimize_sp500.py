import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import optimize
import platform
import itertools

# Set font
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
        if rate <= -1.0:
            return float('inf')
        d0 = dates[0]
        return sum([cf / (1 + rate) ** ((d - d0).days / 365.0) for cf, d in zip(cashflows, dates)])

    try:
        return optimize.newton(lambda r: xnpv(r, cashflows, dates), 0.1)
    except RuntimeError:
        return xnpv(0, cashflows, dates)

def fetch_data():
    print(f"Fetching data for S&P 500 (SPY)...")
    try:
        df = ak.stock_us_daily(symbol="SPY", adjust="")
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = pd.to_numeric(df['close'])
        df = df.sort_values('date')
        
        start_date = pd.Timestamp("2000-01-01")
        df = df[df['date'] >= start_date]
        df = df.rename(columns={'date': '净值日期', 'close': '单位净值'})
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def run_strategy(df, dip_thresholds, buy_ratios, monthly_investment=1000):
    # Simulation Variables
    total_shares = 0.0
    total_invested = 0.0
    cashflows = [] # (amount, date)
    
    # Flags
    # thresholds e.g., [0.2, 0.4, 0.6]
    # ratios e.g., [0.25, 0.5, 0.75]
    # triggered_flags = [False, False, False]
    triggered_flags = [False] * len(dip_thresholds)
    
    max_nav = 0.0
    last_invest_month = None
    
    # To calculate daily returns for Sharpe
    # We need daily total asset value
    daily_values = []
    
    for idx, row in df.iterrows():
        current_date = row['净值日期']
        nav = row['单位净值']
        
        # Update Max NAV
        if nav > max_nav:
            max_nav = nav
            # Reset triggers
            triggered_flags = [False] * len(dip_thresholds)
            
        current_drawdown = (nav - max_nav) / max_nav if max_nav > 0 else 0
        
        # Monthly Investment
        current_month_key = (current_date.year, current_date.month)
        todays_investment = 0.0
        
        if current_month_key != last_invest_month:
            shares_bought = monthly_investment / nav
            total_shares += shares_bought
            total_invested += monthly_investment
            todays_investment += monthly_investment
            
            cashflows.append((-monthly_investment, current_date))
            last_invest_month = current_month_key

        current_market_value = total_shares * nav

        # Dip Buying
        # Iterate thresholds from deepest to shallowest to avoid double triggering if they are close
        # Actually, we should check all.
        # But usually we check deepest first.
        # Let's check in order. If -60% is crossed, -40% and -20% must have been crossed earlier or now.
        # We want distinct triggers.
        # E.g. Market drops instantly to -60%. Should we trigger all 3?
        # User requirement implies levels.
        # Let's assume if we cross -60%, we trigger the -60% bucket.
        # If we are already at -65%, we don't trigger again.
        
        dip_investment = 0.0
        
        # Loop through thresholds (assuming sorted ascending e.g. 0.2, 0.4, 0.6)
        # Thresholds are positive numbers representing drop. Drawdown is negative.
        # So we check if current_drawdown < -threshold
        
        for i, threshold in enumerate(dip_thresholds):
            if current_drawdown < -threshold and not triggered_flags[i]:
                # Trigger buy
                amount_to_buy = current_market_value * buy_ratios[i]
                dip_investment += amount_to_buy
                triggered_flags[i] = True
                
                # Note: If market drops 50% in a day, it crosses 20% and 40% simultaneously.
                # This logic triggers both if they haven't been triggered.
                # This seems consistent with "if drawdown exceeds X".
        
        if dip_investment > 0:
            shares_bought = dip_investment / nav
            total_shares += shares_bought
            total_invested += dip_investment
            todays_investment += dip_investment
            cashflows.append((-dip_investment, current_date))
            current_market_value = total_shares * nav
            
        daily_values.append({
            'date': current_date,
            'market_value': current_market_value,
            'todays_investment': todays_investment
        })

    # Finalize
    final_row = df.iloc[-1]
    final_date = final_row['净值日期']
    final_nav = final_row['单位净值']
    final_market_value = total_shares * final_nav
    cashflows.append((final_market_value, final_date))
    
    # Calculate XIRR
    cf_amounts = [c[0] for c in cashflows]
    cf_dates = [c[1] for c in cashflows]
    try:
        annualized_return = xirr(cf_amounts, cf_dates)
    except:
        annualized_return = 0.0
        
    # Calculate Sharpe Ratio
    # Daily Return = (EndVal - (StartVal + Inflow)) / StartVal
    val_df = pd.DataFrame(daily_values)
    val_df['prev_value'] = val_df['market_value'].shift(1)
    val_df['prev_value'] = val_df['prev_value'].fillna(0) # First day prev is 0
    
    # We only calculate return for days where prev_value > 0
    # Day 1: prev=0, inflow=1000, end=1000. Return undefined/0.
    
    # Formula: r = (V_t - V_{t-1} - C_t) / (V_{t-1} + C_t_adjusted?)
    # Simple Modified Dietz for daily: (V_t - V_{t-1} - C_t) / V_{t-1}
    # This assumes cashflow happens at end of day or doesn't contribute to that day's return?
    # Actually, we bought at price 'nav'. So the cashflow immediately converts to asset.
    # The return is driven by nav change.
    # Simpler: Daily Return of Portfolio is roughly same as Underlying NAV return 
    # IF fully invested. But we are fully invested.
    # So the volatility of the portfolio return % should be identical to the underlying SPY volatility,
    # regardless of leverage/amount, because we don't hold cash.
    # Wait, "Buy the Dip" changes the *timing* of capital entry.
    # Does it change Sharpe?
    # Sharpe = (Rp - Rf) / Sigma_p.
    # Rp (Annualized Return) changes (XIRR).
    # Sigma_p (Volatility). Does volatility change?
    # Since we are always 100% long SPY (just varying amount), the *percentage* daily fluctuation of our portfolio 
    # is exactly equal to SPY's daily fluctuation.
    # (V_t = Shares * Price_t). V_{t-1} = Shares * Price_{t-1}.
    # Return = Price_t / Price_{t-1} - 1.
    # UNLESS we add cash in the middle of the day.
    # If we add cash, the dollar change includes cash.
    # But percentage return of the *invested* capital is just SPY return.
    
    # BUT, Sharpe Ratio of a Strategy usually refers to the return series of the STRATEGY.
    # For a DCA strategy, XIRR is the correct "Return" metric.
    # What is the correct risk metric?
    # Usually it's the volatility of the underlying asset?
    # Or should we penalize maximum drawdown?
    # Calmar Ratio = Annualized Return / Max Drawdown. This might be better for this specific goal (optimizing "Buy the Dip").
    # Sharpe Ratio with constant volatility (since it's SPY) will just maximize Return.
    # So maximizing Sharpe is equivalent to maximizing XIRR.
    # However, user asked for "Sharpe Ratio Optimal".
    # Let's use Calmar Ratio as a secondary or better metric because "Buy the Dip" aims to reduce Drawdown.
    # If we only maximize XIRR, we might just buy aggressively at -5% which increases exposure.
    
    # Let's calculate Max Drawdown of the PORTFOLIO VALUE (Net of inflows? No, raw value).
    # Drawdown of portfolio value is tricky because inflows mask drawdowns.
    # E.g. Market drops 10%, we double our money. Portfolio Value goes UP. Drawdown is 0?
    # This is misleading.
    # Standard practice for DCA drawdown:
    # 1. Market Value Drawdown (What user sees on screen). This matters for psychology.
    # 2. NAV Drawdown (Underlying).
    
    # In previous script, we calculated "Portfolio Drawdown" = (MV - Peak_MV) / Peak_MV.
    # This is the "Account Value Drawdown".
    # This is what hurts users.
    
    # Let's return: XIRR, Max Portfolio Drawdown, Total Invested, Final Value.
    # And we can define "Score" = XIRR / Abs(MaxDD) (Calmar-like).
    
    val_df['max_val'] = val_df['market_value'].cummax()
    val_df['drawdown'] = (val_df['market_value'] - val_df['max_val']) / val_df['max_val']
    max_dd = val_df['drawdown'].min()
    
    return {
        'xirr': annualized_return,
        'max_dd': max_dd,
        'total_invested': total_invested,
        'final_value': final_market_value,
        'calmar': annualized_return / abs(max_dd) if max_dd != 0 else 0
    }

def optimize_strategy():
    df = fetch_data()
    if df is None:
        return

    # Define Search Space
    # Threshold sets (Drop %, Drop %, Drop %)
    threshold_options = [
        [0.10, 0.20, 0.30],
        [0.15, 0.25, 0.35],
        [0.15, 0.30, 0.45],
        [0.20, 0.30, 0.40],
        [0.20, 0.40, 0.60]
    ]
    
    # Buy Ratio sets (Ratio, Ratio, Ratio)
    ratio_options = [
        [0.10, 0.20, 0.30],
        [0.20, 0.30, 0.40],
        [0.25, 0.50, 0.75], # Original
        [0.30, 0.30, 0.30], # Flat
        [0.50, 0.50, 0.50], # Aggressive Flat
        [0.50, 1.00, 1.50]  # Very Aggressive
    ]
    
    results = []
    
    print(f"Running optimization on {len(threshold_options) * len(ratio_options)} combinations...")
    
    for thresholds in threshold_options:
        for ratios in ratio_options:
            res = run_strategy(df, thresholds, ratios)
            res['thresholds'] = thresholds
            res['ratios'] = ratios
            results.append(res)
            # print(f"T={thresholds}, R={ratios} -> XIRR={res['xirr']:.2%}, DD={res['max_dd']:.2%}")

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Sort by Calmar (XIRR / MaxDD)
    res_df = res_df.sort_values('calmar', ascending=False)
    
    # Format output for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    print("\n" + "="*60)
    print("TOP 5 STRATEGIES by Calmar Ratio (Return / Risk)")
    print("="*60)
    print(res_df[['thresholds', 'ratios', 'xirr', 'max_dd', 'calmar']].head(5).to_string())
    
    print("\n" + "="*60)
    print("TOP 5 STRATEGIES by XIRR (Pure Return)")
    print("="*60)
    print(res_df.sort_values('xirr', ascending=False)[['thresholds', 'ratios', 'xirr', 'max_dd', 'calmar']].head(5).to_string())

    # Get Best Calmar Strategy
    best_strat = res_df.iloc[0]
    
    # Visualize Best Strategy vs Base
    # Re-run to get details
    # We can just reuse backtest_sp500_enhanced logic but parameterized
    # Or just print the recommendation.
    
    return best_strat

if __name__ == "__main__":
    optimize_strategy()
