import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# --- 配置页面 ---
st.set_page_config(page_title="China Portfolio Visualizer", layout="wide")

st.title("📊 中国 ETF 基金资产配置回测工具")
st.markdown("模仿 Portfolio Visualizer，基于真实数据进行资产配置分析。")

# --- 缓存数据获取 ---
@st.cache_data(ttl=3600)
def get_etf_list():
    try:
        etf_list = ak.fund_etf_category_sina(symbol="ETF基金")
        return etf_list[['代码', '名称']]
    except:
        # 备选列表
        return pd.DataFrame({
            '代码': [
                'sh510300', 'sh510500', 'sz159915', 'sh510180', 
                'sh511260', 'sh511010', 'sh511880',
                'sh518880', 'sz159934',
                'sz159981', 'sz159985', 'sh510170',
                'sh513100', 'sz159941', 'sh513500'
            ],
            '名称': [
                '沪深300ETF', '中证500ETF', '创业板ETF', '上证180ETF',
                '十年国债ETF', '五年国债ETF', '银华日利货币',
                '黄金ETF', '黄金基金ETF',
                '能源ETF', '豆粕ETF', '商品ETF',
                '纳指ETF(513100)', '纳指ETF(159941)', '标普500ETF'
            ]
        })

@st.cache_data(ttl=3600)
def fetch_etf_data(symbol, start_date, end_date):
    """获取ETF行情数据，优先使用东财接口获取复权数据"""
    try:
        # 提取 6 位数字代码
        code = "".join(filter(str.isdigit, symbol))
        
        # 1. 优先尝试东财接口 (支持前复权 qfq，回测必备)
        try:
            # start_date/end_date 格式通常为 YYYYMMDD
            df = ak.fund_etf_hist_em(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            if df is not None and not df.empty:
                df.rename(columns={'日期': 'date', '收盘': 'close'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df['close'].rename(symbol).sort_index()
        except Exception as e:
            pass # 如果东财失败，尝试新浪
            
        # 2. 备选：新浪接口 (通常不复权)
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df['close'].rename(symbol).sort_index()
            
    except Exception as e:
        st.error(f"获取 {symbol} 数据失败: {e}")
    return None

# --- 侧边栏设置 (Settings) ---
st.sidebar.header("⚙️ 策略设置")
# 动态调整年份范围
current_year = datetime.datetime.now().year
start_year = st.sidebar.number_input("开始年份", 2005, current_year, 2013)
end_year = st.sidebar.number_input("结束年份", 2005, current_year, current_year)
initial_amount = st.sidebar.number_input("初始金额", 1000, 1000000, 10000)
cashflow = st.sidebar.selectbox("定期现金流", ["None", "Monthly Contribution"], index=1)
contribution = 0
if cashflow == "Monthly Contribution":
    contribution = st.sidebar.number_input("每月定资金额", 0, 100000, 1000)

rebalancing = st.sidebar.selectbox("再平衡频率", ["None", "Annual Rebalance", "Monthly Rebalance"], index=1)

# --- 资产选择 (Portfolio Assets) ---
st.header("📂 资产配置")
etf_info = get_etf_list()
# 确保选项不重复且格式正确
etf_options = {f"{row['名称']} ({row['代码']})": row['代码'] for _, row in etf_info.drop_duplicates('代码').iterrows()}

col1, col2 = st.columns([3, 1])

selected_assets = []
weights = []

with col1:
    st.subheader("选择基金")
    num_assets = st.slider("资产数量", 1, 10, 3)
    # 默认值设置，确保一开始就有默认选择
    # 沪深300 (sh510300), 十年国债 (sh511260), 黄金 (sh518880)
    default_tickers = ['sh510300', 'sh511260', 'sh518880']
    
    for i in range(num_assets):
        default_index = 0
        if i < len(default_tickers):
            # 查找默认 ticker 在选项中的索引
            for idx, (label, code) in enumerate(etf_options.items()):
                if code == default_tickers[i]:
                    default_index = idx
                    break
        
        c1, c2 = st.columns([2, 1])
        with c1:
            asset = st.selectbox(f"资产 {i+1}", options=list(etf_options.keys()), index=default_index, key=f"asset_{i}")
        with c2:
            # 默认权重: 60%, 20%, 20%
            default_w = 0
            if i == 0: default_w = 60
            elif i == 1: default_w = 20
            elif i == 2: default_w = 20
            weight = st.number_input(f"权重 (%)", 0, 100, default_w, key=f"weight_{i}")
            
        selected_assets.append(etf_options[asset])
        weights.append(weight / 100.0)

with col2:
    st.subheader("基准选择")
    # 默认寻找沪深300ETF (sh510300) 的索引
    default_bench_idx = 0
    for idx, (label, code) in enumerate(etf_options.items()):
        if "sh510300" in code:
            default_bench_idx = idx
            break
            
    benchmark_asset = st.selectbox("比较基准", options=list(etf_options.keys()), index=default_bench_idx)
    benchmark_code = etf_options[benchmark_asset]

# 校验权重
total_weight = sum(weights)
if abs(total_weight - 1.0) > 1e-6:
    st.warning(f"⚠️ 当前总权重为 {total_weight*100:.1f}%，请确保总权重等于 100%。")

# --- 回测引擎 ---
def run_backtest(asset_data, weights, initial_val, monthly_inv, rebalance_freq):
    # 对齐所有资产的日期
    df = pd.concat(asset_data, axis=1)
    # 先 forward fill 处理停牌，然后再 dropna 处理上市日期不一致
    df = df.ffill().dropna()
    
    if df.empty: return None, 0, None
    
    tickers = df.columns
    shares = {t: 0.0 for t in tickers}
    total_invested = initial_val
    history = []
    nav_history = []
    
    # 初始化
    current_val = initial_val
    for i, t in enumerate(tickers):
        shares[t] = (initial_val * weights[i]) / df.iloc[0][t]
    
    # 初始净值设为 1.0
    current_nav = 1.0
    total_units = initial_val / current_nav
    
    last_month = df.index[0].month
    last_year = df.index[0].year
    
    for date, row in df.iterrows():
        # 1. 计算当日价值 (在定投和再平衡之前，基于前一日持仓)
        current_total_val = sum(shares[t] * row[t] for t in tickers)
        
        # 2. 更新净值 (NAV)
        # 净值 = 当前市值 / 总份额
        current_nav = current_total_val / total_units
        
        # 3. 现金流 (定投) - 在交易日发生
        if monthly_inv > 0 and date.month != last_month:
            # 定投发生在当天行情计算之后（或者说以当天收盘价买入）
            # 买入份额 = 定投金额 / 当前净值
            new_units = monthly_inv / current_nav
            total_units += new_units
            total_invested += monthly_inv
            
            # 实际增加持仓
            for i, t in enumerate(tickers):
                shares[t] += (monthly_inv * weights[i]) / row[t]
            last_month = date.month
            
        # 4. 再平衡
        do_rebalance = False
        if rebalance_freq == "Annual Rebalance" and date.year != last_year:
            do_rebalance = True
            last_year = date.year
        elif rebalance_freq == "Monthly Rebalance" and date.month != last_month:
            do_rebalance = True
            
        if do_rebalance:
            # 再平衡不改变总市值和净值，只改变持仓结构
            current_total_val = sum(shares[t] * row[t] for t in tickers)
            for i, t in enumerate(tickers):
                shares[t] = (current_total_val * weights[i]) / row[t]
        
        # 记录当日状态
        # 注意：为了绘图准确，记录定投后的市值
        final_val_today = sum(shares[t] * row[t] for t in tickers)
        history.append(final_val_today)
        nav_history.append(current_nav)
        
    return pd.Series(history, index=df.index), total_invested, pd.Series(nav_history, index=df.index)

# --- 运行回测 ---
if st.button("🚀 开始分析回测", disabled=(abs(total_weight - 1.0) > 1e-6)):
    with st.spinner("正在获取实时数据并计算..."):
        # 1. 预检所有资产的最早可用日期
        all_symbols = selected_assets + [benchmark_code]
        asset_raw_data = {}
        max_start_date = pd.to_datetime(f"{start_year}-01-01")
        
        for symbol in all_symbols:
            data = fetch_etf_data(symbol, "20050101", f"{end_year}1231")
            if data is not None and not data.empty:
                asset_raw_data[symbol] = data
                # 记录该资产实际开始日期
                if data.index[0] > max_start_date:
                    max_start_date = data.index[0]
        
        # 2. 统一过滤数据，确保从“共同起始日”开始
        st.info(f"💡 自动检测：由于部分资产上市较晚，回测将从共同起始日 **{max_start_date.strftime('%Y-%m-%d')}** 开始。")
        
        asset_series = []
        for symbol in selected_assets:
            if symbol in asset_raw_data:
                asset_series.append(asset_raw_data[symbol][max_start_date:f"{end_year}-12-31"])
        
        benchmark_data = None
        if benchmark_code in asset_raw_data:
            benchmark_data = asset_raw_data[benchmark_code][max_start_date:f"{end_year}-12-31"]

        if len(asset_series) == len(selected_assets) and benchmark_data is not None:
            # 记录计算中间过程
            calc_logs = {
                "回测参数": {
                    "初始金额": initial_amount,
                    "定投金额": contribution,
                    "再平衡": rebalancing,
                    "共同起始日期": max_start_date.strftime('%Y-%m-%d'),
                },
                "投资组合详情": {s: f"{asset_raw_data[s].index[0].date()} 至 {asset_raw_data[s].index[-1].date()}" for s in selected_assets},
                "比较基准详情": {benchmark_code: f"{asset_raw_data[benchmark_code].index[0].date()} 至 {asset_raw_data[benchmark_code].index[-1].date()}"}
            }
            
            result_history, total_inv, nav_history = run_backtest(asset_series, weights, initial_amount, contribution, rebalancing)
            
            if result_history is not None:
                # 基准也跑一个简单的回测
                benchmark_series, bench_total_inv, b_nav_history = run_backtest([benchmark_data], [1.0], initial_amount, contribution, "None")
                
                # --- 展示结果 ---
                st.success("回测完成！")
                
                # 核心指标计算
                def get_metrics(history, invested, nav_series):
                    if history is None or len(history) < 2:
                        return 0, 0, 0, 0, 0, 0, pd.Series()
                    
                    history = pd.to_numeric(history, errors='coerce').ffill()
                    nav_series = pd.to_numeric(nav_series, errors='coerce').ffill()
                    
                    final = history.iloc[-1]
                    # 总收益率依然使用 (最终价值-总投入)/总投入
                    total_ret = (final - invested) / invested if invested != 0 else 0
                    
                    # 波动率、夏普比率、年化收益率应基于 NAV 计算，以排除定投干扰
                    daily_rets = nav_series.pct_change().dropna()
                    
                    if daily_rets.empty:
                        return 0, 0, 0, 0, total_ret, final, daily_rets
                        
                    days = (history.index[-1] - history.index[0]).days
                    # 年化收益率使用 NAV 的年化
                    ann_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (365 / days) - 1 if days > 0 else 0
                    ann_vol = float(daily_rets.std()) * np.sqrt(252)
                    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
                    
                    # 最大回撤基于 NAV 计算
                    cummax = nav_series.cummax()
                    dd = (nav_series - cummax) / cummax
                    max_dd = float(dd.min())
                    return ann_ret, ann_vol, sharpe, max_dd, total_ret, final, daily_rets

                ann_ret, ann_vol, sharpe, max_dd, total_ret, final_val, daily_rets = get_metrics(result_history, total_inv, nav_history)
                b_ann_ret, b_ann_vol, b_sharpe, b_max_dd, b_total_ret, b_final, b_daily_rets = get_metrics(benchmark_series, bench_total_inv, b_nav_history)

                # 计算 Alpha 和 Beta
                common_dates = daily_rets.index.intersection(b_daily_rets.index)
                if len(common_dates) > 10:
                    s_ret = daily_rets.loc[common_dates]
                    b_ret = b_daily_rets.loc[common_dates]
                    covariance = np.cov(s_ret, b_ret)[0][1]
                    b_variance = np.var(b_ret)
                    beta = covariance / b_variance if b_variance != 0 else 1.0
                    alpha = ann_ret - (0.03 + beta * (b_ann_ret - 0.03))
                else:
                    alpha, beta = 0, 1

                # 指标展示
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("最终资产", f"¥{final_val:,.2f}", f"{total_ret*100:.2f}% 总收益")
                m_col2.metric("年化收益", f"{ann_ret*100:.2f}%", f"{(ann_ret - b_ann_ret)*100:.2f}% vs 基准")
                m_col3.metric("年化波动", f"{ann_vol*100:.2f}%", f"基准: {b_ann_vol*100:.2f}%", delta_color="inverse")

                m_col4, m_col5, m_col6 = st.columns(3)
                m_col4.metric("年化夏普", f"{sharpe:.2f}", f"基准: {b_sharpe:.2f}")
                m_col5.metric("最大回撤", f"{max_dd*100:.2f}%", f"基准: {b_max_dd*100:.2f}%", delta_color="inverse")
                m_col6.metric("Alpha / Beta", f"{alpha*100:.1f}% / {beta:.2f}")
                
                # 图表
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, subplot_titles=("资产总值对比 (含定投)", "回撤曲线对比"))
                
                # 组合
                fig.add_trace(go.Scatter(x=result_history.index, y=result_history, name="我的组合", line=dict(color='#1f77b4', width=2)), row=1, col=1)
                # 基准
                fig.add_trace(go.Scatter(x=benchmark_series.index, y=benchmark_series, name=f"基准: {benchmark_asset}", line=dict(color='#7f7f7f', dash='dot')), row=1, col=1)
                
                # 回撤
                dd_series = (result_history - result_history.cummax()) / result_history.cummax()
                b_dd_series = (benchmark_series - benchmark_series.cummax()) / benchmark_series.cummax()
                fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series*100, name="组合回撤", fill='tozeroy', line=dict(color='#d62728', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=b_dd_series.index, y=b_dd_series*100, name="基准回撤", line=dict(color='#7f7f7f', width=1)), row=2, col=1)
                
                fig.update_layout(height=700, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                # 年度数据
                st.subheader("📅 年度表现指标")
                annual_data = []
                for year, group in nav_history.groupby(nav_history.index.year):
                    # 获取该年度的净值序列
                    y_nav = group
                    # 计算该年度收益率: (年末净值 / 年初净值) - 1
                    # 注意：如果跨年，年初净值应该是上一年的最后一个值
                    prev_year_data = nav_history[nav_history.index.year < year]
                    start_nav = prev_year_data.iloc[-1] if not prev_year_data.empty else y_nav.iloc[0]
                    y_ret = (y_nav.iloc[-1] / start_nav) - 1
                    
                    # 基准年度收益
                    by_nav = b_nav_history[b_nav_history.index.year == year]
                    if not by_nav.empty:
                        b_prev_year_data = b_nav_history[b_nav_history.index.year < year]
                        b_start_nav = b_prev_year_data.iloc[-1] if not b_prev_year_data.empty else by_nav.iloc[0]
                        by_ret = (by_nav.iloc[-1] / b_start_nav) - 1
                    else:
                        by_ret = 0
                    
                    # 最大回撤
                    y_dd = ((y_nav - y_nav.cummax()) / y_nav.cummax()).min()
                    
                    # 获取该年度末的总资产
                    y_final_val = result_history[result_history.index.year == year].iloc[-1]
                    
                    annual_data.append({
                        "年份": year, 
                        "期末资产": f"¥{y_final_val:,.2f}",
                        "组合收益": f"{y_ret*100:.2f}%", 
                        "基准收益": f"{by_ret*100:.2f}%",
                        "超额收益": f"{(y_ret - by_ret)*100:.2f}%",
                        "组合最大回撤": f"{y_dd*100:.2f}%"
                    })
                st.table(pd.DataFrame(annual_data))

                # --- 记录与核对验证 ---
                with st.expander("🔍 回测过程详情与验证 (核对计算结果)"):
                    st.write("### 1. 基础配置")
                    st.json(calc_logs)
                    
                    st.write("### 2. 资产数据状态")
                    # 组合资产
                    portfolio_status = [{"资产": s, "角色": "组合成员", "起止日期": calc_logs["投资组合详情"][s]} for s in selected_assets]
                    # 基准资产
                    benchmark_status = [{"资产": benchmark_code, "角色": "比较基准", "起止日期": calc_logs["比较基准详情"][benchmark_code]}]
                    
                    status_df = pd.DataFrame(portfolio_status + benchmark_status)
                    st.table(status_df)
                    
                    st.write("### 3. 累计投资额验证")
                    st.info(f"回测结束时累计投入本金: ¥{total_inv:,.2f} (初始 ¥{initial_amount:,.2f} + 定投)")
            else:
                st.error("所选资产的重叠交易日期不足，无法回测。请调整开始年份。")
        else:
            st.error("部分资产或基准数据获取失败，请重试。")

