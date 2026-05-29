# -*- coding: utf-8 -*-
"""
ema_rsi_10pt.py — EMA 5/12 交叉 + RSI 确认策略（10 点日内盈利）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略来源：
    改编自"10 点盈利策略"（David Tristan），原策略针对外汇市场设计。
    本版本针对币安加密货币期货市场优化，保留核心逻辑并适配 blocks API。

核心逻辑：
    1. 使用两条 EMA（5 周期、12 周期）识别短期趋势方向
    2. 使用 RSI（9 周期）过滤假信号，确认动量方向
    3. 仅在伦敦/纽约交易时段重叠窗口内交易（北京时间 20:00-00:00）
    4. 固定止盈 10 点，固定止损 2 点（基于交叉 K 线的高低点）

适用市场：
    - 高流动性交易对：BTCUSDT、ETHUSDT、BNBUSDT
    - 时间周期：5 分钟 K 线（M5）
    - 市场状态：趋势明确、中等波动

指标说明：
    ✔ EMA(5)  — 快速指数移动平均线，敏感捕捉价格变化
    ✔ EMA(12) — 慢速指数移动平均线，确认趋势方向
    ✔ RSI(9)  — 相对强弱指数，过滤震荡假信号

入场规则：
    【做多】
        1. EMA(5) 向上穿越 EMA(12)
        2. RSI(9) > 50（确认多头动量）
        3. 在交叉确认后的下一根 K 线开盘价入场

    【做空】
        1. EMA(5) 向下穿越 EMA(12)
        2. RSI(9) < 50（确认空头动量）
        3. 在交叉确认后的下一根 K 线开盘价入场

出场规则（由框架统一处理，此处仅标注）：
    - 止盈（Take Profit）：入场价 +10 点（多头）/ -10 点（空头）
    - 止损（Stop Loss）：
        * 多头：交叉 K 线最低价 -2 点
        * 空头：交叉 K 线最高价 +2 点
    - 时间止损：当日 00:00（北京时间）前未平仓则强制离场

风控参数（需在配置文件中设置）：
    - 单笔最大风险：账户权益的 1-2%
    - 每日最大交易次数：建议不超过 5 次
    - 避开重大新闻发布时间（使用财经日历过滤）

回测表现（参考原策略 EUR/USD 数据）：
    - 胜率：~72%
    - 平均盈利：+10 点（+0.94R）
    - 平均亏损：-10.6 点（-1R）
    - 期望值：+0.365R / 每笔交易

注意事项：
    ⚠ 该策略盈亏比 < 1，依赖高胜率盈利
    ⚠ 滑点和手续费对利润影响显著，需选择低点差账户
    ⚠ 震荡市场会产生连续假信号，需配合 ADX 或波动率过滤
    ⚠ 严格执行纪律，避免情绪化加仓或提前平仓

作者：Binance AI 量化策略助手
版本：1.0.0
最后更新：2026-05-16
"""

from cyqnt_trd.blocks import indicators as ind, conditions as cond, entry, strategy
from typing import Tuple
import pandas as pd


def make_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    生成做多/做空信号。

    参数：
        df (pd.DataFrame): OHLCV 数据，必须包含以下列：
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - close_time: K 线收盘时间（毫秒时间戳）

    返回：
        Tuple[pd.Series, pd.Series]: (long_signal, short_signal)
            - long_signal: 布尔序列，True 表示做多入场信号
            - short_signal: 布尔序列，True 表示做空入场信号

    信号逻辑：
        1. 计算 EMA(5) 和 EMA(12)
        2. 检测 EMA 交叉（金叉/死叉）
        3. 计算 RSI(9) 并确认动量方向
        4. 应用交易时段过滤器（仅北京时间 20:00-00:00）
        5. 返回交叉确认后的下一根 K 线作为入场点
    """
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 1：计算技术指标
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # EMA(5) — 快速均线，对价格变化更敏感
    # 用于捕捉短期趋势的早期信号
    ema_fast = ind.ema(df["close"], period=5)

    # EMA(12) — 慢速均线，确认趋势方向
    # 与 EMA(5) 形成交叉信号
    ema_slow = ind.ema(df["close"], period=12)

    # RSI(9) — 相对强弱指数
    # 标准 RSI 周期为 14，这里使用 9 以加快反应速度
    # RSI > 50 表示多头动量，RSI < 50 表示空头动量
    rsi = ind.rsi(df["close"], period=9)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 2：定义入场条件
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 【做多条件】
    # 1. EMA(5) 向上穿越 EMA(12) — 金叉信号
    # 2. RSI(9) > 50 — 确认多头动量充足
    # 使用 entry.all_of() 将多个条件 AND 逻辑组合
    long_condition = entry.all_of([
        cond.ma_cross_above(ema_fast, ema_slow),  # 金叉
        rsi > 50,                                   # RSI 确认多头
    ])

    # 【做空条件】
    # 1. EMA(5) 向下穿越穿越 EMA(12) — 死叉信号
    # 2. RSI(9) < 50 — 确认空头动量充足
    short_condition = entry.all_of([
        cond.ma_cross_below(ema_fast, ema_slow),  # 死叉
        rsi < 50,                                   # RSI 确认空头
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 3：应用交易时段过滤器
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 原策略要求：仅在伦敦/纽约交易时段重叠期间交易
    # 美东时间 08:00-12:00 = 北京时间 20:00-00:00 = UTC 时间 12:00-16:00
    #
    # time_filter 参数说明：
    #   - timestamps_ms: 毫秒时间戳序列（使用 df["close_time"]）
    #   - start_hour: 开始小时（UTC）
    #   - end_hour: 结束小时（UTC）
    #   - tz_offset_hours: 时区偏移（0=UTC，8=北京时间）
    #
    # 这里使用 UTC 时间 12:00-16:00，tz_offset_hours=0
    trading_window = cond.time_filter(
        df["close_time"],
        start_hour=12,    # UTC 12:00
        end_hour=16,      # UTC 16:00
        tz_offset_hours=0  # 使用 UTC 时间
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 4：组合最终信号
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 将入场条件与时段过滤器 AND 逻辑组合
    # 只有在交易时段内且满足技术条件时才产生信号
    long_signal = long_condition & trading_window
    short_signal = short_condition & trading_window

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 步骤 5：信号延迟处理（下一根 K 线入场）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 原策略要求：在交叉确认后的"下一根 K 线"入场
    # 这是因为交叉信号在当前 K 线收盘后才能确认
    # 使用 shift(1) 将信号向后延迟一根 K 线
    long_signal = long_signal.shift(1).fillna(False).astype(bool)
    short_signal = short_signal.shift(1).fillna(False).astype(bool)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 调试信息（开发阶段使用，生产环境可注释）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 取消以下注释可查看信号统计：
    #
    # print(f"[DEBUG] Total bars: {len(df)}")
    # print(f"[DEBUG] Long signals: {long_signal.sum()}")
    # print(f"[DEBUG] Short signals: {short_signal.sum()}")
    # print(f"[DEBUG] Trading window active: {trading_window.sum()} bars")
    # print(f"[DEBUG] EMA fast (last 5): {ema_fast.tail().tolist()}")
    # print(f"[DEBUG] EMA slow (last 5): {ema_slow.tail().tolist()}")
    # print(f"[DEBUG] RSI (last 5): {rsi.tail().tolist()}")

    return long_signal, short_signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 策略注册
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 将策略注册到系统中，使用唯一 ID 标识
# ID 命名规范：小写字母 + 下划线，描述策略核心特征
#
# 注册后可通过以下命令回测：
#   python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
#     --engine python \
#     --strategy ema_rsi_10pt \
#     --strategy-module ema_rsi_10pt \
#     --symbol BTCUSDT \
#     --interval 5m \
#     --limit 1000

strategy.register("ema_rsi_10pt", make_signals)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 附录：止损止盈规则说明（由框架统一处理）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 以下规则需在配置文件或回测参数中设置，不在本文件中实现：
#
# 【止盈规则】
#   - 多头：take_profit_pct = +0.0010 (10 点 = 0.1%)
#   - 空头：take_profit_pct = -0.0010
#
# 【止损规则】
#   - 多头：stop_loss = 交叉 K 线最低价 - 0.0002 (2 点)
#   - 空头：stop_loss = 交叉 K 线最高价 + 0.0002
#
# 【时间止损】
#   - 当日 00:00（北京时间）前未平仓则强制离场
#   - 避免隔夜风险
#
# 【仓位管理】
#   - 建议单笔风险不超过账户权益的 1-2%
#   - 使用 sizing.risk_based_size() 计算仓位
#
# 【交易频率限制】
#   - 每日最多 5 次交易
#   - 避免过度交易导致手续费侵蚀利润
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
