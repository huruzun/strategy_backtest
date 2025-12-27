import pandas as pd
import numpy as np
import datetime
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    if len(returns) < 2: return 0
    # 日频夏普比率
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0: return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def run_backtest(df, weights, monthly_investment=1000):
    tickers = list(weights.keys())
    shares = {ticker: 0.0 for ticker in tickers}
    total_invested = 0.0
    history = []
    last_month = None
    last_year = None
    
    for date, row in df.iterrows():
        # 1. Monthly Investment (First trading day of month)
        if date.month != last_month:
            amount = monthly_investment
            total_invested += amount
            for t in tickers:
                invest_amount = amount * weights[t]
                shares[t] += invest_amount / row[t]
            last_month = date.month
            
        # 2. Annual Rebalancing
        if date.year != last_year and last_year is not None:
            current_value = sum(shares[t] * row[t] for t in tickers)
            for t in tickers:
                shares[t] = (current_value * weights[t]) / row[t]
            last_year = date.year
        elif last_year is None:
            last_year = date.year
            
        # 3. Daily Value
        current_value = sum(shares[t] * row[t] for t in tickers)
        history.append(current_value)
    
    history_df = pd.Series(history, index=df.index)
    daily_returns = history_df.pct_change().dropna()
    sharpe = calculate_sharpe_ratio(daily_returns)
    
    total_return = (history[-1] - total_invested) / total_invested
    
    # Max Drawdown
    cummax = history_df.cummax()
    drawdown = (history_df - cummax) / cummax
    max_dd = drawdown.min()
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_dd': max_dd,
        'final_value': history[-1],
        'total_invested': total_invested,
        'history': history_df
    }

def optimize_china_all_weather():
    # 读取数据，指定第一列为索引
    df = pd.read_csv('china_assets_raw.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # 创业板CYB在2010年之前是NaN，为了包含CYB，我们从2010-06-01开始回测（创业板指发布时间）
    df_filtered = df[df.index >= '2010-06-01'].copy().ffill()
    
    print(f"Starting optimization from {df_filtered.index.min().date()} to {df_filtered.index.max().date()}...")
    
    best_sharpe = -1
    best_weights = None
    
    results = []
    
    print("Searching for best weights (Grid Search)...")
    # 权重网格：债券、黄金、股票（股票内部分为沪深300、中证500、创业板）
    for w_bond in [0.4, 0.5, 0.6]: # 债券大头
        for w_gold in [0.05, 0.1, 0.15]: # 黄金补充
            w_stocks_total = round(1.0 - w_bond - w_gold, 2)
            if w_stocks_total < 0.1: continue
            
            # 股票内部配比 (HS300, ZZ500, CYB)
            # 比例组合：(0.4, 0.3, 0.3), (0.6, 0.2, 0.2), (0.3, 0.4, 0.3) 等
            stock_mixes = [
                (0.4, 0.4, 0.2),
                (0.3, 0.3, 0.4),
                (0.6, 0.2, 0.2),
                (0.2, 0.2, 0.6),
                (0.33, 0.33, 0.34)
            ]
            
            for s_hs, s_zz, s_cy in stock_mixes:
                weights = {
                    'HS300': w_stocks_total * s_hs,
                    'ZZ500': w_stocks_total * s_zz,
                    'CYB': w_stocks_total * s_cy,
                    'Bond': w_bond,
                    'Gold': w_gold
                }
                
                res = run_backtest(df_filtered, weights)
                
                if res['sharpe'] > best_sharpe:
                    best_sharpe = res['sharpe']
                    best_weights = weights
    
    print("\n" + "="*40)
    print("中国市场全天候策略优化结果")
    print("="*40)
    print(f"最佳夏普比率: {best_sharpe:.4f}")
    print("最佳资产配置比例:")
    for t, w in best_weights.items():
        print(f"  - {t}: {w*100:.2f}%")
    
    # 使用最佳权重跑一遍完整回测并绘图
    best_res = run_backtest(df_filtered, best_weights)
    
    print(f"\n最佳方案表现 (2010-06 至今):")
    print(f"  总投入本金: {best_res['total_invested']:,.2f}")
    print(f"  期末总资产: {best_res['final_value']:,.2f}")
    print(f"  总收益率: {best_res['total_return']*100:.2f}%")
    print(f"  年化夏普比率: {best_res['sharpe']:.2f}")
    print(f"  最大回撤: {best_res['max_dd']*100:.2f}%")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(best_res['history'].index, best_res['history'], label='最佳全天候配置', color='blue')
    plt.title('中国市场全天候策略最佳配置回测 (2010-06 至今)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    cummax = best_res['history'].cummax()
    drawdown = (best_res['history'] - cummax) / cummax
    plt.plot(drawdown.index, drawdown, color='red')
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.title('回撤曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('china_all_weather_optimized.png')
    print("\nOptimization chart saved to china_all_weather_optimized.png")

if __name__ == "__main__":
    optimize_china_all_weather()
