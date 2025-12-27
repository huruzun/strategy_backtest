import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
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
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0: return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_alpha_beta(strat_returns, bench_returns, risk_free_rate=0.03):
    if len(strat_returns) < 2 or len(bench_returns) < 2:
        return np.nan, np.nan
    
    # 对齐数据并去除空值
    df_comp = pd.concat([strat_returns, bench_returns], axis=1).dropna()
    if len(df_comp) < 10: # 数据太少不计算
        return np.nan, np.nan
        
    s = df_comp.iloc[:, 0]
    b = df_comp.iloc[:, 1]
    
    # 计算 Beta
    covariance = np.cov(s, b)[0][1]
    benchmark_variance = np.var(b)
    if benchmark_variance == 0:
        beta = 1.0
    else:
        beta = covariance / benchmark_variance
        
    # 计算年化收益率
    ann_strat_ret = (1 + s).prod() ** (252 / len(s)) - 1
    ann_bench_ret = (1 + b).prod() ** (252 / len(b)) - 1
    
    # 计算 Alpha (詹森指数)
    # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    alpha = ann_strat_ret - (risk_free_rate + beta * (ann_bench_ret - risk_free_rate))
    
    return alpha, beta

def run_backtest_dynamic(df, target_weights, monthly_investment=1000):
    tickers = list(target_weights.keys())
    shares = {ticker: 0.0 for ticker in tickers}
    total_invested = 0.0
    history = []
    strategy_daily_returns = []
    last_month = None
    last_year = None
    
    for date, row in df.iterrows():
        # 确定当前可用的资产及其权重
        available_tickers = [t for t in tickers if not pd.isna(row[t])]
        if not available_tickers: continue
        
        # 动态调整权重
        current_weights = target_weights.copy()
        if 'CYB' in tickers and pd.isna(row['CYB']):
            missing_w = current_weights['CYB']
            current_weights['CYB'] = 0
            stock_tickers = [t for t in ['HS300', 'ZZ500'] if t in available_tickers]
            if stock_tickers:
                for st in stock_tickers:
                    current_weights[st] += missing_w / len(stock_tickers)
        
        # 计算当天的策略收益率 (不计入定投)
        if len(history) > 0:
            daily_ret = 0
            for t in available_tickers:
                asset_ret = (row[t] - df.loc[:date, t].iloc[-2]) / df.loc[:date, t].iloc[-2]
                daily_ret += current_weights[t] * asset_ret
            strategy_daily_returns.append(daily_ret)

        # 1. Monthly Investment
        if date.month != last_month:
            amount = monthly_investment
            total_invested += amount
            for t in available_tickers:
                invest_amount = amount * current_weights[t]
                shares[t] += invest_amount / row[t]
            last_month = date.month
            
        # 2. Annual Rebalancing
        if date.year != last_year and last_year is not None:
            current_value = sum(shares[t] * row[t] for t in available_tickers if shares[t] > 0)
            for t in available_tickers:
                shares[t] = (current_value * current_weights[t]) / row[t]
            last_year = date.year
        elif last_year is None:
            last_year = date.year
            
        # 3. Daily Value
        current_value = sum(shares[t] * row[t] for t in available_tickers if shares[t] > 0)
        history.append(current_value)
    
    history_df = pd.Series(history, index=df.index[-len(history):])
    # 策略日收益率序列 (长度比 history 少 1)
    strat_ret_series = pd.Series(strategy_daily_returns, index=history_df.index[1:])
    # 基准 (HS300) 日收益率序列
    bench_ret_series = df['HS300'].pct_change().dropna()
    
    sharpe = calculate_sharpe_ratio(strat_ret_series)
    ann_ret = (1 + strat_ret_series).prod() ** (252 / len(strat_ret_series)) - 1
    ann_vol = strat_ret_series.std() * np.sqrt(252)
    alpha, beta = calculate_alpha_beta(strat_ret_series, bench_ret_series)
    
    total_return = (history[-1] - total_invested) / total_invested
    cummax = history_df.cummax()
    drawdown = (history_df - cummax) / cummax
    max_dd = drawdown.min()

    # 计算年度指标
    annual_metrics = []
    years = history_df.index.year.unique()
    for year in years:
        year_history = history_df[history_df.index.year == year]
        # 使用纯策略收益率计算年度表现
        year_strat_rets = strat_ret_series[strat_ret_series.index.year == year]
        year_bench_rets = bench_ret_series[bench_ret_series.index.year == year]
        
        if len(year_strat_rets) == 0: continue
        
        # 年度指标
        year_twr = (1 + year_strat_rets).prod() - 1
        year_vol = year_strat_rets.std() * np.sqrt(252)
        year_sharpe = calculate_sharpe_ratio(year_strat_rets)
        year_alpha, _ = calculate_alpha_beta(year_strat_rets, year_bench_rets)
        
        # 回撤 (使用资产总值的回撤更符合实际体验)
        year_cummax = year_history.cummax()
        year_dd = (year_history - year_cummax) / year_cummax
        year_max_dd = year_dd.min()
        
        annual_metrics.append({
            '年份': year,
            '收益率': f"{year_twr*100:.2f}%",
            '波动率': f"{year_vol*100:.2f}%",
            '最大回撤': f"{year_max_dd*100:.2f}%",
            '夏普': f"{year_sharpe:.2f}",
            'Alpha': f"{year_alpha*100:.2f}%"
        })
    
    return {
        'sharpe': sharpe,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'alpha': alpha,
        'beta': beta,
        'total_return': total_return,
        'max_dd': max_dd,
        'final_value': history[-1],
        'total_invested': total_invested,
        'history': history_df,
        'annual_metrics': pd.DataFrame(annual_metrics)
    }

def main():
    df = pd.read_csv('china_assets_raw.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # 最佳权重（根据之前的网格搜索）
    best_weights = {
        'HS300': 0.05,
        'ZZ500': 0.05,
        'CYB': 0.15,
        'Bond': 0.60,
        'Gold': 0.15
    }
    
    print(f"Running final backtest from {df.index.min().date()} with dynamic allocation...")
    res = run_backtest_dynamic(df, best_weights)
    
    print("\n" + "="*40)
    print("中国市场全天候策略最终报告 (2005-2025)")
    print("="*40)
    print(f"资产配置比例:")
    for t, w in best_weights.items():
        print(f"  - {t}: {w*100:.1f}%")
    print("-" * 20)
    print(f"总投入本金: {res['total_invested']:,.2f}")
    print(f"期末总资产: {res['final_value']:,.2f}")
    print(f"总收益率: {res['total_return']*100:.2f}%")
    print(f"年化收益率: {res['ann_ret']*100:.2f}%")
    print(f"年化波动率: {res['ann_vol']*100:.2f}%")
    print(f"年化夏普比率: {res['sharpe']:.2f}")
    print(f"阿尔法系数 (Alpha): {res['alpha']*100:.2f}%")
    print(f"贝塔系数 (Beta): {res['beta']:.2f}")
    print(f"最大回撤: {res['max_dd']*100:.2f}%")
    print("="*40)
    print("\n年度指标明细:")
    print(res['annual_metrics'].to_string(index=False))
    print("="*40)
    
    # 绘图
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(res['history'].index, res['history'], label='中国全天候策略 (动态分配)', color='blue')
    plt.title('中国市场全天候策略回测 (2005-2025)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    drawdown = (res['history'] - res['history'].cummax()) / res['history'].cummax()
    plt.plot(drawdown.index, drawdown, color='red')
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.title('回撤曲线')
    plt.grid(True)

    # 添加年度指标表格
    plt.subplot(3, 1, 3)
    plt.axis('off')
    metrics_df = res['annual_metrics']
    # 拆分表格，如果行数太多
    table = plt.table(cellText=metrics_df.values,
                      colLabels=metrics_df.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    plt.title('年度表现指标 (基准: 沪深300)', pad=20)
    
    plt.tight_layout()
    plt.savefig('china_all_weather_final.png')
    print("\nFinal chart saved to china_all_weather_final.png")

if __name__ == "__main__":
    main()
