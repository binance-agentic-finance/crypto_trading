"""
计算当前币对合约各个factor的值、看多/看空信号和胜率

使用方法：
    python test_now_factor.py --symbol BTCUSDT --interval 30m --lookback 500
"""

import os
import sys
import argparse
import warnings
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# 抑制 pandas FutureWarning 关于 fillna 的警告
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays on .fillna.*')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 cyqnt_trd 包
try:
    from cyqnt_trd.get_data.get_futures_data import get_and_save_futures_klines
    from cyqnt_trd.blocks.factors.ma_factor import ma_factor
    from cyqnt_trd.blocks.factors.rsi_factor import rsi_factor
    from cyqnt_trd.blocks.factors.stochastic_factor import stochastic_k_factor
    from cyqnt_trd.blocks.factors.cci_factor import cci_factor
    from cyqnt_trd.blocks.factors.adx_factor import adx_factor
    from cyqnt_trd.blocks.factors.ao_factor import ao_factor
    from cyqnt_trd.blocks.factors.momentum_factor import momentum_factor
    from cyqnt_trd.blocks.factors.macd_factor import macd_level_factor
    from cyqnt_trd.blocks.factors.stochastic_tsi_factor import stochastic_tsi_fast_factor
    from cyqnt_trd.blocks.factors.williams_r_factor import williams_r_factor
    from cyqnt_trd.blocks.factors.bbp_factor import bbp_factor
    from cyqnt_trd.blocks.factors.uo_factor import uo_factor
    from cyqnt_trd.blocks.factors.ema_factor import ema_factor, ema_cross_factor
    from cyqnt_trd.blocks.alphas import (
        alpha1_factor, alpha3_factor, alpha7_factor, alpha9_factor,
        alpha11_factor, alpha15_factor, alpha17_factor, alpha21_factor,
        alpha23_factor, alpha25_factor, alpha29_factor, alpha33_factor,
        alpha34_factor, ALPHA_FACTORS
    )
    from cyqnt_trd.backtesting.factor_test import FactorTester
    from cyqnt_trd.utils import set_user
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n提示：请确保已安装 cyqnt_trd package: pip install -e /path/to/crypto_trading")
    sys.exit(1)


def klines_to_dataframe(klines_data: list) -> pd.DataFrame:
    """
    将K线数据列表转换为DataFrame
    
    Args:
        klines_data: K线数据列表，每个元素是一个包含K线信息的列表或字典
    
    Returns:
        DataFrame，包含以下列：
        - open_time: 开盘时间（毫秒时间戳）
        - open_time_str: 开盘时间字符串
        - open_price: 开盘价
        - high_price: 最高价
        - low_price: 最低价
        - close_price: 收盘价
        - volume: 成交量
        - close_time: 收盘时间（毫秒时间戳）
        - quote_volume: 成交额
        - trades: 成交笔数
        - taker_buy_base_volume: 主动买入成交量
        - taker_buy_quote_volume: 主动买入成交额
    """
    if not klines_data:
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(klines_data)
    
    # 标准化列名（根据Binance API返回的格式）
    if len(df.columns) >= 12:
        df.columns = [
            'open_time', 'open_price', 'high_price', 'low_price', 'close_price',
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ]
    elif len(df.columns) >= 11:
        df.columns = [
            'open_time', 'open_price', 'high_price', 'low_price', 'close_price',
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume'
        ]
    
    # 转换数据类型
    numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 
                      'volume', 'quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 转换时间
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df['open_time_str'] = df['open_time'].apply(
            lambda x: datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else ''
        )
    
    # 按时间排序
    if 'open_time' in df.columns:
        df = df.sort_values('open_time').reset_index(drop=True)
    
    return df


def calculate_normalized_alpha_factor(
    data_slice: pd.DataFrame,
    factor_func: Callable,
    factor_name: str,
    min_required: int = 30,
    lookback_periods: int = 30,
    **factor_kwargs
) -> Optional[Dict[str, Any]]:
    """
    计算归一化Alpha因子的通用函数
    
    Args:
        data_slice: 数据切片
        factor_func: 因子计算函数
        factor_name: 因子名称（用于日志）
        min_required: 因子计算所需的最小周期数
        lookback_periods: 归一化回看周期数
        **factor_kwargs: 传递给因子函数的额外参数
        
    Returns:
        包含因子值和看多/看空结果的字典，如果计算失败则返回None
    """
    try:
        if len(data_slice) < min_required + 2:
            return None
        
        available_periods = len(data_slice) - min_required - 1
        if available_periods < 2:
            return None
        
        actual_lookback = min(lookback_periods, max(2, available_periods))
        
        # 计算因子值：当前周期和之前actual_lookback个周期（使用多线程并行计算）
        def compute_factor_value(i):
            """计算单个时间点的因子值"""
            end_idx = len(data_slice) - i
            start_idx = max(0, end_idx - min_required - 1)
            if end_idx <= start_idx:
                return 0.0
            
            period_slice = data_slice.iloc[start_idx:end_idx]
            try:
                # 调用因子函数，传入额外参数
                factor_value = factor_func(period_slice, **factor_kwargs)
                if factor_value is not None:
                    return factor_value
                else:
                    return 0.0
            except Exception:
                return 0.0
        
        # 使用多线程并行计算因子值
        indices = list(range(actual_lookback + 1))
        factor_values = []
        with ThreadPoolExecutor() as executor:
            # 返回顺序保证和原for循环一致
            factor_values = list(executor.map(compute_factor_value, indices))
        
        if len(factor_values) < 2:
            return None
        
        # 归一化
        factor_array = np.array(factor_values)
        factor_min = factor_array.min()
        factor_max = factor_array.max()
        
        if factor_max == factor_min:
            normalized_factors = np.zeros_like(factor_array)
        else:
            # Min-Max归一化到[-1, 1]区间
            normalized_factors = 2 * (factor_array - factor_min) / (factor_max - factor_min) - 1
        
        current_normalized = float(normalized_factors[0])
        prev_normalized = float(normalized_factors[1]) if len(normalized_factors) > 1 else 0.0
        
        # 判断信号：从负转正看多，从正转负看空
        if prev_normalized <= 0 and current_normalized > 0:
            signal = '看多'
        elif prev_normalized >= 0 and current_normalized < 0:
            signal = '看空'
        else:
            signal = '中性'
        
        return {
            'value': current_normalized,
            'signal': signal,
            'raw_value': float(factor_values[0]) if len(factor_values) > 0 else 0.0,
            'prev_normalized': prev_normalized
        }
    except Exception as e:
        print(f"计算归一化{factor_name}因子时出错: {e}")
        return None


def calculate_factor_win_rate(
    data_df: pd.DataFrame,
    factor_func: Callable,
    forward_periods: int = 24,
    min_periods: int = 30,
    factor_name: str = "factor"
) -> Optional[Dict[str, float]]:
    """
    计算因子基于历史数据的胜率（使用FactorTester.test_factor）
    
    Args:
        data_df: 历史数据DataFrame
        factor_func: 因子计算函数，接受数据切片作为参数，返回因子值
        forward_periods: 向前看的周期数（默认24，即未来24个周期）
        min_periods: 最小需要的周期数
        factor_name: 因子名称
        
    Returns:
        包含胜率信息的字典，如果计算失败则返回None
    """
    try:
        if len(data_df) < min_periods + forward_periods + 1:
            return None
        
        # 创建FactorTester实例
        factor_tester = FactorTester(data_df)
        
        # 调用test_factor计算胜率
        test_results = factor_tester.test_factor(
            factor_func=factor_func,
            forward_periods=forward_periods,
            min_periods=min_periods,
            factor_name=factor_name
        )
        
        # 提取需要的胜率信息
        result = {
            'long_win_rate': test_results.get('long_win_rate', 0.0),
            'short_win_rate': test_results.get('short_win_rate', 0.0),
            'overall_win_rate': test_results.get('overall_win_rate', 0.0),
            'long_avg_return': test_results.get('long_avg_return', 0.0),
            'short_avg_return': test_results.get('short_avg_return', 0.0),
            'long_signals': test_results.get('long_signals', 0),
            'short_signals': test_results.get('short_signals', 0),
            'total_samples': test_results.get('total_samples', 0)
        }
        
        return result
        
    except Exception as e:
        print(f"计算因子胜率时出错: {e}")
        return None


def calculate_all_factors(data_df: pd.DataFrame, forward_periods=24) -> Dict[str, Any]:
    """
    计算所有因子的因子值、看多/看空信号和胜率
    
    Args:
        data_df: 历史数据DataFrame
        
    Returns:
        包含所有因子结果的字典
    """
    result = {}
    
    if len(data_df) < 10:
        print("数据量不足，无法计算因子")
        return result
    
    # 使用足够的数据切片（对于alpha因子，需要更多数据）
    min_slice_size = 65
    if len(data_df) >= min_slice_size:
        data_slice = data_df.iloc[-min_slice_size:].copy()
    elif len(data_df) >= 30:
        data_slice = data_df.iloc[-30:].copy()
    else:
        data_slice = data_df.copy()
    
    try:
        # 定义所有需要计算的因子任务
        def calculate_ma_factor():
            """计算MA因子"""
            if len(data_slice) < 6:
                return None
            try:
                ma_factor_value = ma_factor(data_slice, period=5)
                ma_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: ma_factor(d, period=5),
                    forward_periods=forward_periods,
                    min_periods=6,
                    factor_name="MA5因子"
                )
                return ('ma_factor_5', {
                    'value': ma_factor_value,
                    'signal': '看多' if ma_factor_value > 0 else '看空' if ma_factor_value < 0 else '中性',
                    'raw_value': ma_factor_value,
                    'win_rate': ma_win_rate
                })
            except Exception as e:
                print(f"计算MA因子时出错: {e}")
                return None
        
        def calculate_rsi_factor():
            """计算RSI因子"""
            if len(data_slice) < 16:
                return None
            try:
                rsi_factor_value = rsi_factor(data_slice, period=14)
                rsi_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: rsi_factor(d, period=14),
                    forward_periods=forward_periods,
                    min_periods=16,
                    factor_name="RSI14因子"
                )
                return ('rsi_factor_14', {
                    'value': rsi_factor_value,
                    'signal': '看多' if rsi_factor_value > 0 else '看空' if rsi_factor_value < 0 else '中性',
                    'raw_value': rsi_factor_value,
                    'win_rate': rsi_win_rate
                })
            except Exception as e:
                print(f"计算RSI因子时出错: {e}")
                return None
        
        def calculate_stochastic_k_factor():
            """计算Stochastic %K因子"""
            if len(data_slice) < 18:
                return None
            try:
                stoch_value = stochastic_k_factor(data_slice, period=14, k_smooth=3, d_smooth=3)
                stoch_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: stochastic_k_factor(d, period=14, k_smooth=3, d_smooth=3),
                    forward_periods=forward_periods,
                    min_periods=18,
                    factor_name="Stochastic %K(14,3,3)因子"
                )
                return ('stochastic_k_factor_14_3_3', {
                    'value': stoch_value,
                    'signal': '看多' if stoch_value > 0 else '看空' if stoch_value < 0 else '中性',
                    'raw_value': stoch_value,
                    'win_rate': stoch_win_rate
                })
            except Exception as e:
                print(f"计算Stochastic %K因子时出错: {e}")
                return None
        
        def calculate_cci_factor():
            """计算CCI因子"""
            if len(data_slice) < 21:
                return None
            try:
                cci_value = cci_factor(data_slice, period=20)
                cci_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: cci_factor(d, period=20),
                    forward_periods=forward_periods,
                    min_periods=21,
                    factor_name="CCI(20)因子"
                )
                return ('cci_factor_20', {
                    'value': cci_value,
                    'signal': '看多' if cci_value > 0 else '看空' if cci_value < 0 else '中性',
                    'raw_value': cci_value,
                    'win_rate': cci_win_rate
                })
            except Exception as e:
                print(f"计算CCI因子时出错: {e}")
                return None
        
        def calculate_adx_factor():
            """计算ADX因子"""
            if len(data_slice) < 30:
                return None
            try:
                adx_value = adx_factor(data_slice, period=14)
                adx_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: adx_factor(d, period=14),
                    forward_periods=forward_periods,
                    min_periods=30,
                    factor_name="ADX(14)因子"
                )
                return ('adx_factor_14', {
                    'value': adx_value,
                    'signal': '看多' if adx_value > 0 else '看空' if adx_value < 0 else '中性',
                    'raw_value': adx_value,
                    'win_rate': adx_win_rate
                })
            except Exception as e:
                print(f"计算ADX因子时出错: {e}")
                return None
        
        def calculate_ao_factor():
            """计算AO因子"""
            if len(data_slice) < 36:
                return None
            try:
                ao_value = ao_factor(data_slice)
                ao_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=ao_factor,
                    forward_periods=forward_periods,
                    min_periods=36,
                    factor_name="AO因子"
                )
                return ('ao_factor', {
                    'value': ao_value,
                    'signal': '看多' if ao_value > 0 else '看空' if ao_value < 0 else '中性',
                    'raw_value': ao_value,
                    'win_rate': ao_win_rate
                })
            except Exception as e:
                print(f"计算AO因子时出错: {e}")
                return None
        
        def calculate_momentum_factor():
            """计算动量因子"""
            if len(data_slice) < 11:
                return None
            try:
                momentum_value = momentum_factor(data_slice, period=10)
                momentum_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: momentum_factor(d, period=10),
                    forward_periods=forward_periods,
                    min_periods=11,
                    factor_name="Momentum(10)因子"
                )
                return ('momentum_factor_10', {
                    'value': momentum_value,
                    'signal': '看多' if momentum_value > 0 else '看空' if momentum_value < 0 else '中性',
                    'raw_value': momentum_value,
                    'win_rate': momentum_win_rate
                })
            except Exception as e:
                print(f"计算Momentum因子时出错: {e}")
                return None
        
        def calculate_macd_factor():
            """计算MACD因子"""
            if len(data_slice) < 48:
                return None
            try:
                macd_value = macd_level_factor(data_slice, fast_period=12, slow_period=26)
                macd_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: macd_level_factor(d, fast_period=12, slow_period=26),
                    forward_periods=forward_periods,
                    min_periods=48,
                    factor_name="MACD(12,26)因子"
                )
                return ('macd_factor_12_26', {
                    'value': macd_value,
                    'signal': '看多' if macd_value > 0 else '看空' if macd_value < 0 else '中性',
                    'raw_value': macd_value,
                    'win_rate': macd_win_rate
                })
            except Exception as e:
                print(f"计算MACD因子时出错: {e}")
                return None
        
        def calculate_stochastic_tsi_factor():
            """计算Stochastic TSI因子"""
            if len(data_slice) < 35:
                return None
            try:
                stoch_tsi_value = stochastic_tsi_fast_factor(data_slice, r_period=3, s_period=3, tsi_period1=14, tsi_period2=14)
                stoch_tsi_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: stochastic_tsi_fast_factor(d, r_period=3, s_period=3, tsi_period1=14, tsi_period2=14),
                    forward_periods=24,
                    min_periods=35,
                    factor_name="Stochastic TSI Fast(3,3,14,14)因子"
                )
                return ('stochastic_tsi_factor_3_3_14_14', {
                    'value': stoch_tsi_value,
                    'signal': '看多' if stoch_tsi_value > 0 else '看空' if stoch_tsi_value < 0 else '中性',
                    'raw_value': stoch_tsi_value,
                    'win_rate': stoch_tsi_win_rate
                })
            except Exception as e:
                print(f"计算Stochastic TSI因子时出错: {e}")
                return None
        
        def calculate_williams_r_factor():
            """计算Williams %R因子"""
            if len(data_slice) < 15:
                return None
            try:
                williams_r_value = williams_r_factor(data_slice, period=14)
                williams_r_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: williams_r_factor(d, period=14),
                    forward_periods=forward_periods,
                    min_periods=15,
                    factor_name="Williams %R(14)因子"
                )
                return ('williams_r_factor_14', {
                    'value': williams_r_value,
                    'signal': '看多' if williams_r_value > 0 else '看空' if williams_r_value < 0 else '中性',
                    'raw_value': williams_r_value,
                    'win_rate': williams_r_win_rate
                })
            except Exception as e:
                print(f"计算Williams %R因子时出错: {e}")
                return None
        
        def calculate_bbp_factor():
            """计算BBP因子"""
            if len(data_slice) < 14:
                return None
            try:
                bbp_value = bbp_factor(data_slice, period=13)
                bbp_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: bbp_factor(d, period=13),
                    forward_periods=24,
                    min_periods=14,
                    factor_name="BBP因子"
                )
                return ('bbp_factor', {
                    'value': bbp_value,
                    'signal': '看多' if bbp_value > 0 else '看空' if bbp_value < 0 else '中性',
                    'raw_value': bbp_value,
                    'win_rate': bbp_win_rate
                })
            except Exception as e:
                print(f"计算BBP因子时出错: {e}")
                return None
        
        def calculate_uo_factor():
            """计算UO因子"""
            if len(data_slice) < 29:
                return None
            try:
                uo_value = uo_factor(data_slice, period1=7, period2=14, period3=28)
                uo_win_rate = calculate_factor_win_rate(
                    data_df=data_df,
                    factor_func=lambda d: uo_factor(d, period1=7, period2=14, period3=28),
                    forward_periods=forward_periods,
                    min_periods=29,
                    factor_name="UO(7,14,28)因子"
                )
                return ('uo_factor_7_14_28', {
                    'value': uo_value,
                    'signal': '看多' if uo_value > 0 else '看空' if uo_value < 0 else '中性',
                    'raw_value': uo_value,
                    'win_rate': uo_win_rate
                })
            except Exception as e:
                print(f"计算UO因子时出错: {e}")
                return None
        
        def calculate_ema_factors():
            """计算EMA因子（多个周期）"""
            results = {}
            ema_periods = [10, 20, 30, 50, 100, 200]
            
            for period in ema_periods:
                if len(data_slice) < period + 1:
                    continue
                try:
                    ema_value = ema_factor(data_slice, period=period)
                    ema_win_rate = calculate_factor_win_rate(
                        data_df=data_df,
                        factor_func=lambda d, p=period: ema_factor(d, period=p),
                        forward_periods=24,
                        min_periods=period + 1,
                        factor_name=f"EMA({period})因子"
                    )
                    results[f'ema_factor_{period}'] = {
                        'value': ema_value,
                        'signal': '看多' if ema_value > 0 else '看空' if ema_value < 0 else '中性',
                        'raw_value': ema_value,
                        'win_rate': ema_win_rate
                    }
                except Exception as e:
                    print(f"计算EMA({period})因子时出错: {e}")
            
            return results if results else None
        
        def calculate_normalized_alpha(factor_key, factor_func, min_req, alpha_num, **kwargs):
            """计算归一化Alpha因子的通用函数"""
            try:
                normalized_result = calculate_normalized_alpha_factor(
                    data_slice=data_slice,
                    factor_func=factor_func,
                    factor_name=f"Alpha#{alpha_num}",
                    min_required=min_req,
                    lookback_periods=30,
                    **kwargs
                )
                if normalized_result:
                    def normalized_wrapper(d, func=factor_func, req=min_req, num=alpha_num, kw=kwargs):
                        norm_res = calculate_normalized_alpha_factor(
                            data_slice=d,
                            factor_func=func,
                            factor_name=f"Alpha#{num}",
                            min_required=req,
                            lookback_periods=30,
                            **kw
                        )
                        if norm_res:
                            return norm_res['value']
                        return 0.0
                    
                    win_rate = calculate_factor_win_rate(
                        data_df=data_df,
                        factor_func=normalized_wrapper,
                        forward_periods=forward_periods,
                        min_periods=65,
                        factor_name=f"归一化Alpha#{alpha_num}因子"
                    )
                    normalized_result['win_rate'] = win_rate
                    return (f'normalized_{factor_key}', normalized_result)
            except Exception as e:
                print(f"计算{factor_key}因子时出错: {e}")
            return None
        
        # 准备所有因子计算任务
        tasks = []
        
        # MA因子和RSI因子
        if len(data_slice) >= 6:
            tasks.append(calculate_ma_factor)
        if len(data_slice) >= 16:
            tasks.append(calculate_rsi_factor)
        
        # 新增技术指标因子
        if len(data_slice) >= 18:
            tasks.append(calculate_stochastic_k_factor)
        if len(data_slice) >= 21:
            tasks.append(calculate_cci_factor)
        if len(data_slice) >= 30:
            tasks.append(calculate_adx_factor)
        if len(data_slice) >= 36:
            tasks.append(calculate_ao_factor)
        if len(data_slice) >= 11:
            tasks.append(calculate_momentum_factor)
        if len(data_slice) >= 48:
            tasks.append(calculate_macd_factor)
        if len(data_slice) >= 35:
            tasks.append(calculate_stochastic_tsi_factor)
        if len(data_slice) >= 15:
            tasks.append(calculate_williams_r_factor)
        if len(data_slice) >= 14:
            tasks.append(calculate_bbp_factor)
        if len(data_slice) >= 29:
            tasks.append(calculate_uo_factor)
        
        # EMA因子（多个周期）
        def calculate_ema_wrapper():
            ema_results = calculate_ema_factors()
            if ema_results:
                return list(ema_results.items())
            return None
        
        if len(data_slice) >= 11:
            tasks.append(calculate_ema_wrapper)
        
        # 归一化Alpha因子（选择一些常用的）
        alpha_factors_to_add = [
            ('alpha1', alpha1_factor, 30, '1', {'lookback_days': 5, 'stddev_period': 20, 'power': 2.0}),
            ('alpha3', alpha3_factor, 30, '3', {}),
            ('alpha7', alpha7_factor, 30, '7', {}),
            ('alpha9', alpha9_factor, 30, '9', {}),
            ('alpha11', alpha11_factor, 30, '11', {}),
            ('alpha15', alpha15_factor, 30, '15', {}),
            ('alpha17', alpha17_factor, 30, '17', {}),
            ('alpha21', alpha21_factor, 30, '21', {}),
            ('alpha23', alpha23_factor, 30, '23', {}),
            ('alpha25', alpha25_factor, 30, '25', {}),
            ('alpha29', alpha29_factor, 30, '29', {}),
            ('alpha33', alpha33_factor, 30, '33', {}),
            ('alpha34', alpha34_factor, 30, '34', {}),
        ]
        
        for factor_key, factor_func, min_req, alpha_num, kwargs in alpha_factors_to_add:
            # 使用默认参数捕获循环变量，避免闭包问题
            tasks.append(lambda k=factor_key, f=factor_func, r=min_req, n=alpha_num, kw=kwargs: 
                       calculate_normalized_alpha(k, f, r, n, **kw))
        
        # 使用多线程并行计算所有因子
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in futures:
                try:
                    task_result = future.result()
                    if task_result is not None:
                        # 处理EMA因子返回的多个结果
                        if isinstance(task_result, list):
                            for key, value in task_result:
                                result[key] = value
                        else:
                            key, value = task_result
                            result[key] = value
                except Exception as e:
                    print(f"计算因子任务时出错: {e}")
                    
    except Exception as e:
        print(f"计算因子值时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def print_factor_results(factor_results: Dict[str, Any]):
    """
    打印因子结果
    
    Args:
        factor_results: 因子结果字典
    """
    print("\n" + "="*100)
    print("📊 因子分析结果")
    print("="*100)
    
    if not factor_results:
        print("未计算出任何因子结果")
        return
    
    # 按因子类型分组显示
    print("\n【基础因子】")
    print("-"*100)
    
    # MA因子
    if 'ma_factor_5' in factor_results:
        ma_info = factor_results['ma_factor_5']
        signal_emoji = "🟢" if ma_info.get('signal') == '看多' else "🔴" if ma_info.get('signal') == '看空' else "⚪"
        signal_text = ma_info.get('signal', '中性')
        print(f"MA5因子:")
        print(f"  因子值: {ma_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if ma_info.get('win_rate'):
            wr = ma_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # RSI因子
    if 'rsi_factor_14' in factor_results:
        rsi_info = factor_results['rsi_factor_14']
        signal_emoji = "🟢" if rsi_info.get('signal') == '看多' else "🔴" if rsi_info.get('signal') == '看空' else "⚪"
        signal_text = rsi_info.get('signal', '中性')
        print(f"RSI14因子:")
        print(f"  因子值: {rsi_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if rsi_info.get('win_rate'):
            wr = rsi_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # Stochastic %K因子
    if 'stochastic_k_factor_14_3_3' in factor_results:
        stoch_info = factor_results['stochastic_k_factor_14_3_3']
        signal_emoji = "🟢" if stoch_info.get('signal') == '看多' else "🔴" if stoch_info.get('signal') == '看空' else "⚪"
        signal_text = stoch_info.get('signal', '中性')
        print(f"Stochastic %K(14,3,3)因子:")
        print(f"  因子值: {stoch_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if stoch_info.get('win_rate'):
            wr = stoch_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # CCI因子
    if 'cci_factor_20' in factor_results:
        cci_info = factor_results['cci_factor_20']
        signal_emoji = "🟢" if cci_info.get('signal') == '看多' else "🔴" if cci_info.get('signal') == '看空' else "⚪"
        signal_text = cci_info.get('signal', '中性')
        print(f"CCI(20)因子:")
        print(f"  因子值: {cci_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if cci_info.get('win_rate'):
            wr = cci_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # ADX因子
    if 'adx_factor_14' in factor_results:
        adx_info = factor_results['adx_factor_14']
        signal_emoji = "🟢" if adx_info.get('signal') == '看多' else "🔴" if adx_info.get('signal') == '看空' else "⚪"
        signal_text = adx_info.get('signal', '中性')
        print(f"ADX(14)因子:")
        print(f"  因子值: {adx_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if adx_info.get('win_rate'):
            wr = adx_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # AO因子
    if 'ao_factor' in factor_results:
        ao_info = factor_results['ao_factor']
        signal_emoji = "🟢" if ao_info.get('signal') == '看多' else "🔴" if ao_info.get('signal') == '看空' else "⚪"
        signal_text = ao_info.get('signal', '中性')
        print(f"AO因子:")
        print(f"  因子值: {ao_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if ao_info.get('win_rate'):
            wr = ao_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # Momentum因子
    if 'momentum_factor_10' in factor_results:
        momentum_info = factor_results['momentum_factor_10']
        signal_emoji = "🟢" if momentum_info.get('signal') == '看多' else "🔴" if momentum_info.get('signal') == '看空' else "⚪"
        signal_text = momentum_info.get('signal', '中性')
        print(f"Momentum(10)因子:")
        print(f"  因子值: {momentum_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if momentum_info.get('win_rate'):
            wr = momentum_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # MACD因子
    if 'macd_factor_12_26' in factor_results:
        macd_info = factor_results['macd_factor_12_26']
        signal_emoji = "🟢" if macd_info.get('signal') == '看多' else "🔴" if macd_info.get('signal') == '看空' else "⚪"
        signal_text = macd_info.get('signal', '中性')
        print(f"MACD(12,26)因子:")
        print(f"  因子值: {macd_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if macd_info.get('win_rate'):
            wr = macd_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # Stochastic TSI因子
    if 'stochastic_tsi_factor_3_3_14_14' in factor_results:
        stoch_tsi_info = factor_results['stochastic_tsi_factor_3_3_14_14']
        signal_emoji = "🟢" if stoch_tsi_info.get('signal') == '看多' else "🔴" if stoch_tsi_info.get('signal') == '看空' else "⚪"
        signal_text = stoch_tsi_info.get('signal', '中性')
        print(f"Stochastic TSI Fast(3,3,14,14)因子:")
        print(f"  因子值: {stoch_tsi_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if stoch_tsi_info.get('win_rate'):
            wr = stoch_tsi_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # Williams %R因子
    if 'williams_r_factor_14' in factor_results:
        williams_r_info = factor_results['williams_r_factor_14']
        signal_emoji = "🟢" if williams_r_info.get('signal') == '看多' else "🔴" if williams_r_info.get('signal') == '看空' else "⚪"
        signal_text = williams_r_info.get('signal', '中性')
        print(f"Williams %R(14)因子:")
        print(f"  因子值: {williams_r_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if williams_r_info.get('win_rate'):
            wr = williams_r_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # BBP因子
    if 'bbp_factor' in factor_results:
        bbp_info = factor_results['bbp_factor']
        signal_emoji = "🟢" if bbp_info.get('signal') == '看多' else "🔴" if bbp_info.get('signal') == '看空' else "⚪"
        signal_text = bbp_info.get('signal', '中性')
        print(f"BBP因子:")
        print(f"  因子值: {bbp_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if bbp_info.get('win_rate'):
            wr = bbp_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # UO因子
    if 'uo_factor_7_14_28' in factor_results:
        uo_info = factor_results['uo_factor_7_14_28']
        signal_emoji = "🟢" if uo_info.get('signal') == '看多' else "🔴" if uo_info.get('signal') == '看空' else "⚪"
        signal_text = uo_info.get('signal', '中性')
        print(f"UO(7,14,28)因子:")
        print(f"  因子值: {uo_info.get('raw_value', 0):+.6f}")
        print(f"  信号: {signal_emoji} {signal_text}")
        if uo_info.get('win_rate'):
            wr = uo_info['win_rate']
            if wr and isinstance(wr, dict):
                print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
        print()
    
    # EMA因子（多个周期）
    ema_factors = [k for k in factor_results.keys() if k.startswith('ema_factor_')]
    if ema_factors:
        print("\n【EMA因子】")
        print("-"*100)
        # 按周期排序
        ema_factors.sort(key=lambda x: int(x.replace('ema_factor_', '')))
        for factor_key in ema_factors:
            ema_info = factor_results[factor_key]
            period = factor_key.replace('ema_factor_', '')
            signal_emoji = "🟢" if ema_info.get('signal') == '看多' else "🔴" if ema_info.get('signal') == '看空' else "⚪"
            signal_text = ema_info.get('signal', '中性')
            print(f"EMA({period})因子:")
            print(f"  因子值: {ema_info.get('raw_value', 0):+.6f}")
            print(f"  信号: {signal_emoji} {signal_text}")
            if ema_info.get('win_rate'):
                wr = ema_info['win_rate']
                if wr and isinstance(wr, dict):
                    print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                    print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                    print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
            print()
    
    # 归一化Alpha因子
    print("\n【归一化Alpha因子】")
    print("-"*100)
    
    normalized_alpha_factors = [k for k in factor_results.keys() if k.startswith('normalized_alpha')]
    if normalized_alpha_factors:
        # 按Alpha编号排序
        normalized_alpha_factors.sort(key=lambda x: int(x.replace('normalized_alpha', '')))
        
        for factor_key in normalized_alpha_factors:
            alpha_info = factor_results[factor_key]
            alpha_num = factor_key.replace('normalized_alpha', '')
            signal_emoji = "🟢" if alpha_info.get('signal') == '看多' else "🔴" if alpha_info.get('signal') == '看空' else "⚪"
            signal_text = alpha_info.get('signal', '中性')
            
            print(f"归一化Alpha#{alpha_num}:")
            print(f"  原始值: {alpha_info.get('raw_value', 0):+.6f}")
            print(f"  归一化值: {alpha_info.get('value', 0):+.4f}")
            print(f"  信号: {signal_emoji} {signal_text}")
            if alpha_info.get('win_rate'):
                wr = alpha_info['win_rate']
                if wr and isinstance(wr, dict):
                    print(f"  看多胜率: {wr.get('long_win_rate', 0):.2%} (样本={wr.get('long_signals', 0)})")
                    print(f"  看空胜率: {wr.get('short_win_rate', 0):.2%} (样本={wr.get('short_signals', 0)})")
                    print(f"  总体胜率: {wr.get('overall_win_rate', 0):.2%} (总样本={wr.get('total_samples', 0)})")
            print()
    else:
        print("未计算出归一化Alpha因子结果")
    
    print("="*100)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='计算当前币对合约各个factor的值、看多/看空信号和胜率')
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='交易对符号，例如 BTCUSDT, ETHUSDT')
    parser.add_argument('--interval', type=str, default='5m', help='时间间隔，例如 1m, 5m, 30m, 1h, 1d')
    parser.add_argument('--lookback', type=int, default=500, help='回看周期数（获取多少条历史数据）')
    parser.add_argument('--forward', type=int, default=1, help='向前看周期数（用于计算胜率）')
    
    args = parser.parse_args()
    
    # 设置用户（如果需要API认证）
    set_user()
    
    print("="*100)
    print("📊 因子计算工具")
    print("="*100)
    print(f"交易对: {args.symbol}")
    print(f"时间间隔: {args.interval}")
    print(f"回看周期数: {args.lookback}")
    print(f"向前看周期数: {args.forward}")
    print("="*100)
    
    # 计算开始时间和结束时间
    end_time = datetime.now()
    # 根据interval和lookback计算start_time
    interval_durations = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "1w": timedelta(weeks=1),
    }
    interval_delta = interval_durations.get(args.interval, timedelta(hours=1))
    start_time = end_time - interval_delta * args.lookback
    
    print(f"\n正在获取数据...")
    print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 获取数据（使用临时目录，但不保存文件）
    with tempfile.TemporaryDirectory() as temp_dir:
        klines_data = get_and_save_futures_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time,
            output_dir=temp_dir,
            save_csv=False,
            save_json=False
        )
    
        if not klines_data:
            print("❌ 获取数据失败")
            return
        
        print(f"✅ 成功获取 {len(klines_data)} 条数据")
        
        # 转换为DataFrame
        data_df = klines_to_dataframe(klines_data)
    
        if len(data_df) == 0:
            print("❌ 数据为空")
            return
        
        print(f"✅ 数据已转换为DataFrame，共 {len(data_df)} 行")
        print(f"  时间范围: {data_df.iloc[0]['open_time_str']} 至 {data_df.iloc[-1]['open_time_str']}")
        
        # 计算所有因子
        print(f"\n正在计算因子...")
        factor_results = calculate_all_factors(data_df,forward_periods=3)
        
        # 打印结果
        print_factor_results(factor_results)
        
        print("\n✅ 计算完成！")


if __name__ == "__main__":
    main()

