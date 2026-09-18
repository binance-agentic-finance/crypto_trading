"""
实盘交易脚本

结合 RealtimePriceTracker 和订单执行功能，实现基于实时数据的自动交易。

⚠️ 警告：此脚本会执行真实交易，请谨慎使用！
建议：
1. 先在测试网络或使用小额资金测试
2. 仔细检查所有参数
3. 确保策略已经过充分回测
4. 设置合理的止盈止损

使用方法：
    python real_time_trade.py
"""

import os
import sys
import asyncio
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from cyqnt_trd.utils import set_user

# 抑制 pandas FutureWarning 关于 fillna 的警告
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays on .fillna.*')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 cyqnt_trd 包
try:
    from cyqnt_trd.online_trading.realtime_price_tracker import RealtimePriceTracker
    from cyqnt_trd.test_script.test_order import (
        test_futures_order,
        test_spot_order,
        get_futures_balance,
        get_spot_balance,
        show_futures_balances,
        show_spot_balances,
        get_futures_open_orders,
        cancel_futures_order
    )
    from cyqnt_trd.blocks.signals.ma_signal import ma_signal, ma_cross_signal
    from cyqnt_trd.blocks.signals.factor_based_signal import factor_based_signal, normalized_factor_signal
    from cyqnt_trd.blocks.factors.ma_factor import ma_factor
    from cyqnt_trd.blocks.factors.rsi_factor import rsi_factor
    from cyqnt_trd.blocks.alphas import (
        alpha1_factor, alpha3_factor, alpha7_factor, alpha9_factor,
        alpha11_factor, alpha15_factor, alpha17_factor, alpha21_factor,
        alpha23_factor, alpha25_factor, alpha29_factor, alpha33_factor,
        alpha34_factor
    )
    from cyqnt_trd.backtesting.factor_test import FactorTester
    import numpy as np
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n提示：请确保已安装 cyqnt_trd package: pip install -e /path/to/crypto_trading")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimeTradingBot:
    """
    实盘交易机器人
    
    使用 RealtimePriceTracker 获取实时数据，根据交易信号执行真实订单
    """
    
    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        lookback_periods: int = 100,
        market_type: str = "futures",  # "futures" 或 "spot"
        position_size_pct: float = 0.01,  # 每次使用资金的百分比
        take_profit: float = 0.1,  # 止盈10%
        stop_loss: float = 0.05,  # 止损5%
        strategy: str = "ma5",  # 策略类型
        min_order_quantity: float = 0.001,  # 最小下单数量
        ssl_verify: bool = False,
        dry_run: bool = True  # 是否为模拟模式（不实际下单）
    ):
        """
        初始化实盘交易机器人
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            lookback_periods: 历史数据周期数
            market_type: 市场类型，"futures" 或 "spot"
            position_size_pct: 每次交易使用的资金比例（0-1）
            take_profit: 止盈比例（0-1）
            stop_loss: 止损比例（0-1）
            strategy: 策略类型
            min_order_quantity: 最小下单数量
            ssl_verify: SSL证书验证
            dry_run: 是否为模拟模式（True=不实际下单，False=真实下单）
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.market_type = market_type
        self.position_size_pct = position_size_pct
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.strategy = strategy
        self.min_order_quantity = min_order_quantity
        self.dry_run = dry_run
        
        # 创建价格跟踪器
        self.tracker = RealtimePriceTracker(
            symbol=symbol,
            interval=interval,
            lookback_periods=lookback_periods,
            market_type=market_type,
            ssl_verify=ssl_verify
        )
        
        # 交易状态
        self.position = 0.0  # 当前持仓数量
        self.entry_price = 0.0  # 入场价格
        self.entry_index = -1  # 入场索引
        self.entry_time = None  # 入场时间
        self.entry_order_id = None  # 入场订单ID
        
        # 交易记录
        self.completed_trades = []  # 已完成的交易
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.total_profit = 0.0
        
        # 统计信息
        self.start_time = datetime.now()
        self.last_signal = None
        self.last_signal_time = None
        
        # 注册回调
        self.tracker.register_on_new_kline(self._on_new_kline)
        
        # 显示初始状态
        if dry_run:
            logger.warning("="*80)
            logger.warning("⚠️  模拟模式：不会执行真实订单")
            logger.warning("="*80)
        else:
            logger.warning("="*80)
            logger.warning("⚠️  实盘模式：将执行真实订单！")
            logger.warning("="*80)
    
    def _calculate_normalized_alpha_factor(
        self, 
        data_slice,
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
            logger.debug(f"计算归一化{factor_name}因子时出错: {e}")
            return None
    
    def _calculate_factor_win_rate(
        self,
        data_df,
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
            logger.debug(f"计算因子胜率时出错: {e}")
            return None
    
    def _calculate_factor_values(self, data_df) -> Dict[str, Any]:
        """
        计算各种因子的因子值和看多/看空结果（使用多线程并行计算）
        
        Args:
            data_df: 历史数据DataFrame
            
        Returns:
            包含因子值和看多/看空结果的字典
        """
        result = {}
        
        if len(data_df) < 10:
            return result
        
        # 使用足够的数据切片（对于alpha因子，需要更多数据）
        # 至少需要65个周期（30+30+5缓冲）用于归一化alpha因子计算
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
                    ma_win_rate = self._calculate_factor_win_rate(
                        data_df=data_df,
                        factor_func=lambda d: ma_factor(d, period=5),
                        forward_periods=2,
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
                    logger.debug(f"计算MA因子时出错: {e}")
                    return None
            
            def calculate_normalized_alpha1():
                """计算归一化Alpha#1因子"""
                normalized_result = self._calculate_normalized_alpha_factor(
                    data_slice=data_slice,
                    factor_func=alpha1_factor,
                    factor_name="Alpha#1",
                    min_required=30,
                    lookback_periods=30,
                    lookback_days=5,
                    stddev_period=20,
                    power=2.0
                )
                if normalized_result:
                    def normalized_alpha1_wrapper(d):
                        norm_res = self._calculate_normalized_alpha_factor(
                            data_slice=d,
                            factor_func=alpha1_factor,
                            factor_name="Alpha#1",
                            min_required=30,
                            lookback_periods=30,
                            lookback_days=5,
                            stddev_period=20,
                            power=2.0
                        )
                        if norm_res:
                            return norm_res['value']
                        return 0.0
                    
                    alpha1_win_rate = self._calculate_factor_win_rate(
                        data_df=data_df,
                        factor_func=normalized_alpha1_wrapper,
                        forward_periods=2,
                        min_periods=65,
                        factor_name="归一化Alpha#1因子"
                    )
                    normalized_result['win_rate'] = alpha1_win_rate
                    return ('normalized_alpha1', normalized_result)
                return None
            
            def calculate_normalized_alpha15():
                """计算归一化Alpha#15因子"""
                normalized_result = self._calculate_normalized_alpha_factor(
                    data_slice=data_slice,
                    factor_func=alpha15_factor,
                    factor_name="Alpha#15",
                    min_required=30,
                    lookback_periods=30
                )
                if normalized_result:
                    def normalized_alpha15_wrapper(d):
                        norm_res = self._calculate_normalized_alpha_factor(
                            data_slice=d,
                            factor_func=alpha15_factor,
                            factor_name="Alpha#15",
                            min_required=30,
                            lookback_periods=30
                        )
                        if norm_res:
                            return norm_res['value']
                        return 0.0
                    
                    alpha15_win_rate = self._calculate_factor_win_rate(
                        data_df=data_df,
                        factor_func=normalized_alpha15_wrapper,
                        forward_periods=2,
                        min_periods=65,
                        factor_name="归一化Alpha#15因子"
                    )
                    normalized_result['win_rate'] = alpha15_win_rate
                    return ('normalized_alpha15', normalized_result)
                return None
            
            def calculate_normalized_alpha(factor_key, factor_func, min_req, alpha_num):
                """计算归一化Alpha因子的通用函数"""
                try:
                    normalized_result = self._calculate_normalized_alpha_factor(
                        data_slice=data_slice,
                        factor_func=factor_func,
                        factor_name=f"Alpha#{alpha_num}",
                        min_required=min_req,
                        lookback_periods=30
                    )
                    if normalized_result:
                        def normalized_wrapper(d, func=factor_func, req=min_req, num=alpha_num):
                            norm_res = self._calculate_normalized_alpha_factor(
                                data_slice=d,
                                factor_func=func,
                                factor_name=f"Alpha#{num}",
                                min_required=req,
                                lookback_periods=30
                            )
                            if norm_res:
                                return norm_res['value']
                            return 0.0
                        
                        win_rate = self._calculate_factor_win_rate(
                            data_df=data_df,
                            factor_func=normalized_wrapper,
                            forward_periods=2,
                            min_periods=65,
                            factor_name=f"归一化Alpha#{alpha_num}因子"
                        )
                        normalized_result['win_rate'] = win_rate
                        return (f'normalized_{factor_key}', normalized_result)
                except Exception as e:
                    logger.debug(f"计算{factor_key}因子时出错: {e}")
                return None
            
            # 准备所有因子计算任务
            tasks = []
            
            # MA因子
            if len(data_slice) >= 6:
                tasks.append(calculate_ma_factor)
            
            # 归一化Alpha#1和Alpha#15
            tasks.append(calculate_normalized_alpha1)
            tasks.append(calculate_normalized_alpha15)
            
            # 其他归一化Alpha因子
            alpha_factors_to_add = [
                ('alpha3', alpha3_factor, 30, '3'),
                ('alpha7', alpha7_factor, 30, '7'),
                ('alpha9', alpha9_factor, 30, '9'),
                ('alpha11', alpha11_factor, 30, '11'),
                ('alpha17', alpha17_factor, 30, '17'),
                ('alpha21', alpha21_factor, 30, '21'),
                ('alpha23', alpha23_factor, 30, '23'),
                ('alpha25', alpha25_factor, 30, '25'),
                ('alpha29', alpha29_factor, 30, '29'),
                ('alpha33', alpha33_factor, 30, '33'),
                ('alpha34', alpha34_factor, 30, '34'),
            ]
            
            for factor_key, factor_func, min_req, alpha_num in alpha_factors_to_add:
                # 使用默认参数捕获循环变量，避免闭包问题
                tasks.append(lambda k=factor_key, f=factor_func, r=min_req, n=alpha_num: 
                           calculate_normalized_alpha(k, f, r, n))
            
            # 使用多线程并行计算所有因子
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(task) for task in tasks]
                for future in futures:
                    try:
                        task_result = future.result()
                        if task_result is not None:
                            key, value = task_result
                            result[key] = value
                    except Exception as e:
                        logger.debug(f"计算因子任务时出错: {e}")
                    
        except Exception as e:
            logger.debug(f"计算因子值时出错: {e}")
        
        return result
    
    def _calculate_signal(self, data_df) -> Optional[str]:
        """
        根据策略计算交易信号
        
        Args:
            data_df: 历史数据DataFrame
            
        Returns:
            交易信号: 'buy', 'sell', 'hold' 或 None
        """
        if len(data_df) < 10:
            return None
        
        # 使用足够的数据切片
        data_slice = data_df.iloc[-30:].copy() if len(data_df) >= 30 else data_df.copy()
        
        try:
            if self.strategy == "ma5":
                if len(data_slice) >= 6:
                    return ma_signal(
                        data_slice=data_slice,
                        position=self.position,
                        entry_price=self.entry_price,
                        entry_index=self.entry_index,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        period=5
                    )
            
            elif self.strategy == "ma_cross":
                if len(data_slice) >= 22:
                    return ma_cross_signal(
                        data_slice=data_slice,
                        position=self.position,
                        entry_price=self.entry_price,
                        entry_index=self.entry_index,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        check_periods=1,
                        short_period=5,
                        long_period=20
                    )
            
            elif self.strategy == "ma_factor":
                if len(data_slice) >= 6:
                    return factor_based_signal(
                        data_slice=data_slice,
                        position=self.position,
                        entry_price=self.entry_price,
                        entry_index=self.entry_index,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        check_periods=1,
                        factor_func=lambda d: ma_factor(d, period=5),
                        factor_period=5
                    )
            
            elif self.strategy == "rsi_factor":
                if len(data_slice) >= 16:
                    return factor_based_signal(
                        data_slice=data_slice,
                        position=self.position,
                        entry_price=self.entry_price,
                        entry_index=self.entry_index,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        check_periods=1,
                        factor_func=lambda d: rsi_factor(d, period=14),
                        factor_period=14
                    )
            
            elif self.strategy == "alpha1":
                if len(data_slice) >= 26:
                    return factor_based_signal(
                        data_slice=data_slice,
                        position=self.position,
                        entry_price=self.entry_price,
                        entry_index=self.entry_index,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        check_periods=1,
                        factor_func=lambda d: alpha1_factor(d, lookback_days=5, stddev_period=20, power=2.0),
                        factor_period=25
                    )
        except Exception as e:
            logger.debug(f"计算信号时出错: {e}")
            return None
        
        return None
    
    def _get_available_balance(self) -> float:
        """
        获取可用余额
        
        Returns:
            可用余额
        """
        try:
            if self.market_type == "futures":
                result = get_futures_balance("USDT")
                if result.get("success"):
                    balance_info = result.get("balances", {})
                    return balance_info.get("available", 0.0)
            else:
                # 从交易对中提取报价货币
                quote_asset = "USDT"  # 默认
                if self.symbol.endswith("USDT"):
                    quote_asset = "USDT"
                elif self.symbol.endswith("BUSD"):
                    quote_asset = "BUSD"
                elif self.symbol.endswith("USDC"):
                    quote_asset = "USDC"
                
                result = get_spot_balance(quote_asset)
                if result.get("success"):
                    balance_info = result.get("balances", {})
                    return balance_info.get("free", 0.0)
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
        
        return 0.0
    
    def _calculate_order_quantity(self, price: float, side: str) -> float:
        """
        计算订单数量
        
        Args:
            price: 当前价格
            side: 买卖方向，"BUY" 或 "SELL"
            
        Returns:
            订单数量
        """
        if side == "BUY":
            # 买入：使用可用余额的百分比
            available = self._get_available_balance()
            order_value = available * self.position_size_pct
            quantity = order_value / price
            
            # 确保不小于最小下单数量
            if quantity < self.min_order_quantity:
                return 0.0
            
            return quantity
        else:
            # 卖出：使用当前持仓
            return self.position
    
    def _execute_buy_order(self, price: float, time_str: str) -> bool:
        """
        执行买入订单
        
        Args:
            price: 买入价格
            time_str: 时间字符串
            
        Returns:
            是否成功
        """
        quantity = self._calculate_order_quantity(price, "BUY")
        
        if quantity < self.min_order_quantity:
            logger.warning(f"计算出的数量 {quantity} 小于最小下单数量 {self.min_order_quantity}")
            return False
        
        logger.info(f"准备买入: {self.symbol}, 数量: {quantity:.6f}, 价格: {price:.2f}")
        
        if self.dry_run:
            logger.info("🔵 [模拟] 执行买入订单")
            logger.info(f"  时间: {time_str}")
            logger.info(f"  价格: {price:.2f}")
            logger.info(f"  数量: {quantity:.6f}")
            logger.info(f"  金额: {quantity * price:.2f}")
            
            # 模拟更新状态
            self.position = quantity
            self.entry_price = price
            self.entry_time = time_str
            return True
        else:
            # 真实下单
            try:
                if self.market_type == "futures":
                    result = test_futures_order(
                        symbol=self.symbol,
                        side="BUY",
                        order_type="MARKET",
                        quantity=quantity
                    )
                else:
                    result = test_spot_order(
                        symbol=self.symbol,
                        side="BUY",
                        order_type="MARKET",
                        quantity=quantity
                    )
                
                if result.get("success"):
                    order_data = result.get("order", {})
                    executed_qty = float(order_data.get("executedQty", order_data.get("executed_qty", quantity)))
                    avg_price = float(order_data.get("avgPrice", order_data.get("avg_price", price)))
                    order_id = order_data.get("orderId", order_data.get("order_id"))
                    
                    logger.info(f"✅ 买入订单成功")
                    logger.info(f"  订单ID: {order_id}")
                    logger.info(f"  成交数量: {executed_qty:.6f}")
                    logger.info(f"  成交均价: {avg_price:.2f}")
                    
                    # 更新状态
                    self.position = executed_qty
                    self.entry_price = avg_price
                    self.entry_time = time_str
                    self.entry_order_id = order_id
                    
                    return True
                else:
                    logger.error(f"买入订单失败: {result.get('error')}")
                    return False
            except Exception as e:
                logger.error(f"执行买入订单时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    def _execute_sell_order(self, price: float, time_str: str) -> bool:
        """
        执行卖出订单
        
        Args:
            price: 卖出价格
            time_str: 时间字符串
            
        Returns:
            是否成功
        """
        quantity = self._calculate_order_quantity(price, "SELL")
        
        if quantity < self.min_order_quantity:
            logger.warning(f"持仓数量 {quantity} 小于最小下单数量 {self.min_order_quantity}")
            return False
        
        logger.info(f"准备卖出: {self.symbol}, 数量: {quantity:.6f}, 价格: {price:.2f}")
        
        if self.dry_run:
            # 计算盈亏
            profit_amount = (price - self.entry_price) * quantity
            profit_pct = (price - self.entry_price) / self.entry_price * 100
            
            logger.info("🔴 [模拟] 执行卖出订单")
            logger.info(f"  时间: {time_str}")
            logger.info(f"  价格: {price:.2f}")
            logger.info(f"  入场价: {self.entry_price:.2f}")
            logger.info(f"  数量: {quantity:.6f}")
            logger.info(f"  盈亏金额: {profit_amount:+.2f}")
            logger.info(f"  盈亏比例: {profit_pct:+.2f}%")
            
            # 记录交易
            trade_record = {
                'entry_time': self.entry_time,
                'exit_time': time_str,
                'entry_price': self.entry_price,
                'exit_price': price,
                'quantity': quantity,
                'profit_amount': profit_amount,
                'profit_pct': profit_pct
            }
            self.completed_trades.append(trade_record)
            
            # 更新统计
            self.total_trades += 1
            self.total_profit += profit_amount
            if profit_amount > 0:
                self.win_trades += 1
            else:
                self.loss_trades += 1
            
            # 重置持仓
            self.position = 0.0
            self.entry_price = 0.0
            self.entry_time = None
            self.entry_order_id = None
            
            return True
        else:
            # 真实下单
            try:
                if self.market_type == "futures":
                    result = test_futures_order(
                        symbol=self.symbol,
                        side="SELL",
                        order_type="MARKET",
                        quantity=quantity,
                        reduce_only="true"  # 只减仓
                    )
                else:
                    result = test_spot_order(
                        symbol=self.symbol,
                        side="SELL",
                        order_type="MARKET",
                        quantity=quantity
                    )
                
                if result.get("success"):
                    order_data = result.get("order", {})
                    executed_qty = float(order_data.get("executedQty", order_data.get("executed_qty", quantity)))
                    avg_price = float(order_data.get("avgPrice", order_data.get("avg_price", price)))
                    order_id = order_data.get("orderId", order_data.get("order_id"))
                    
                    # 计算盈亏
                    profit_amount = (avg_price - self.entry_price) * executed_qty
                    profit_pct = (avg_price - self.entry_price) / self.entry_price * 100
                    
                    logger.info(f"✅ 卖出订单成功")
                    logger.info(f"  订单ID: {order_id}")
                    logger.info(f"  成交数量: {executed_qty:.6f}")
                    logger.info(f"  成交均价: {avg_price:.2f}")
                    logger.info(f"  盈亏金额: {profit_amount:+.2f}")
                    logger.info(f"  盈亏比例: {profit_pct:+.2f}%")
                    
                    # 记录交易
                    trade_record = {
                        'entry_time': self.entry_time,
                        'exit_time': time_str,
                        'entry_price': self.entry_price,
                        'exit_price': avg_price,
                        'quantity': executed_qty,
                        'profit_amount': profit_amount,
                        'profit_pct': profit_pct,
                        'entry_order_id': self.entry_order_id,
                        'exit_order_id': order_id
                    }
                    self.completed_trades.append(trade_record)
                    
                    # 更新统计
                    self.total_trades += 1
                    self.total_profit += profit_amount
                    if profit_amount > 0:
                        self.win_trades += 1
                    else:
                        self.loss_trades += 1
                    
                    # 重置持仓
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.entry_time = None
                    self.entry_order_id = None
                    
                    return True
                else:
                    logger.error(f"卖出订单失败: {result.get('error')}")
                    return False
            except Exception as e:
                logger.error(f"执行卖出订单时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
    
    def _on_new_kline(self, kline_dict: Dict[str, Any], data_df):
        """
        新K线数据回调函数
        
        Args:
            kline_dict: 新K线数据字典
            data_df: 历史数据DataFrame
        """
        current_price = kline_dict['close_price']
        current_time = kline_dict['open_time_str']
        
        # 计算交易信号
        signal = self._calculate_signal(data_df)
        
        # 显示状态（包含因子值）
        self._display_status(current_time, current_price, signal, data_df)
        
        # 检查止盈止损（如果有持仓）
        if self.position > 0:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= self.take_profit:
                logger.info(f"触发止盈: {profit_pct*100:.2f}% >= {self.take_profit*100:.2f}%")
                self._execute_sell_order(current_price, current_time)
                return
            elif profit_pct <= -self.stop_loss:
                logger.info(f"触发止损: {profit_pct*100:.2f}% <= -{self.stop_loss*100:.2f}%")
                self._execute_sell_order(current_price, current_time)
                return
        
        # 执行交易
        if signal == 'buy' and self.position == 0:
            # 避免频繁交易：检查上次信号时间
            if self.last_signal == 'buy' and self.last_signal_time:
                time_diff = (datetime.now() - self.last_signal_time).total_seconds()
                if time_diff < 60:  # 至少间隔60秒
                    logger.debug("买入信号过于频繁，跳过")
                    return
            
            self._execute_buy_order(current_price, current_time)
            self.last_signal = 'buy'
            self.last_signal_time = datetime.now()
            
        elif signal == 'sell' and self.position > 0:
            self._execute_sell_order(current_price, current_time)
            self.last_signal = 'sell'
            self.last_signal_time = datetime.now()
    
    def _display_status(self, time_str: str, price: float, signal: Optional[str], data_df=None):
        """
        显示当前状态
        
        Args:
            time_str: 时间字符串
            price: 当前价格
            signal: 交易信号
            data_df: 历史数据DataFrame（用于计算因子值）
        """
        # 计算统计信息
        runtime = datetime.now() - self.start_time
        runtime_str = f"{runtime.days}天 {runtime.seconds // 3600}小时 {(runtime.seconds % 3600) // 60}分钟"
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # 获取余额
        available_balance = self._get_available_balance()
        
        # 信号显示
        if signal:
            signal_emoji = "🟢" if signal == 'buy' else "🔴" if signal == 'sell' else "⚪"
            signal_text = f"{signal_emoji} {signal.upper()}"
        else:
            signal_text = "⚪ HOLD"
        
        print(f"\n{'='*80}")
        print(f"📊 实时状态更新")
        print(f"{'='*80}")
        print(f"时间: {time_str}")
        print(f"价格: {price:.2f}")
        print(f"信号: {signal_text}")
        if self.position > 0:
            profit_pct = (price - self.entry_price) / self.entry_price * 100
            print(f"持仓: {self.position:.6f} | 入场价: {self.entry_price:.2f} | 浮动盈亏: {profit_pct:+.2f}%")
        else:
            print(f"持仓: 无")
        
        # 计算并显示因子值
        if data_df is not None:
            try:
                factor_results = self._calculate_factor_values(data_df)
                if factor_results:
                    print(f"{'='*80}")
                    print(f"📈 因子分析:")
                    
                    # MA因子
                    if 'ma_factor_5' in factor_results:
                        ma_info = factor_results['ma_factor_5']
                        signal_emoji = "🟢" if ma_info.get('signal') == '看多' else "🔴" if ma_info.get('signal') == '看空' else "⚪"
                        signal_text = ma_info.get('signal', '中性')
                        win_rate_info = ""
                        if ma_info.get('win_rate'):
                            wr = ma_info['win_rate']
                            if wr and isinstance(wr, dict):
                                if signal_text == '看多' and wr.get('long_win_rate') is not None:
                                    win_rate_info = f" | 看多胜率={wr['long_win_rate']:.2%} (样本={wr.get('long_signals', 0)})"
                                elif signal_text == '看空' and wr.get('short_win_rate') is not None:
                                    win_rate_info = f" | 看空胜率={wr['short_win_rate']:.2%} (样本={wr.get('short_signals', 0)})"
                                if wr.get('overall_win_rate') is not None:
                                    win_rate_info += f" | 总体胜率={wr['overall_win_rate']:.2%}"
                        print(f"  MA5因子: 因子值={ma_info.get('raw_value', 0):+.4f} | {signal_emoji} {signal_text}{win_rate_info}")
                    
                    # # 归一化Alpha#1因子
                    # if 'normalized_alpha1' in factor_results:
                    #     alpha1_info = factor_results['normalized_alpha1']
                    #     signal_emoji = "🟢" if alpha1_info.get('signal') == '看多' else "🔴" if alpha1_info.get('signal') == '看空' else "⚪"
                    #     signal_text = alpha1_info.get('signal', '中性')
                    #     win_rate_info = ""
                    #     if alpha1_info.get('win_rate'):
                    #         wr = alpha1_info['win_rate']
                    #         if wr and isinstance(wr, dict):
                    #             if signal_text == '看多' and wr.get('long_win_rate') is not None:
                    #                 win_rate_info = f" | 看多胜率={wr['long_win_rate']:.2%} (样本={wr.get('long_signals', 0)})"
                    #             elif signal_text == '看空' and wr.get('short_win_rate') is not None:
                    #                 win_rate_info = f" | 看空胜率={wr['short_win_rate']:.2%} (样本={wr.get('short_signals', 0)})"
                    #             if wr.get('overall_win_rate') is not None:
                    #                 win_rate_info += f" | 总体胜率={wr['overall_win_rate']:.2%}"
                    #     print(f"  归一化Alpha#1: 原始值={alpha1_info.get('raw_value', 0):+.6f} | 归一化值={alpha1_info.get('value', 0):+.4f} | {signal_emoji} {signal_text}{win_rate_info}")
                    
                    # # 归一化Alpha#15因子
                    # if 'normalized_alpha15' in factor_results:
                    #     alpha15_info = factor_results['normalized_alpha15']
                    #     signal_emoji = "🟢" if alpha15_info.get('signal') == '看多' else "🔴" if alpha15_info.get('signal') == '看空' else "⚪"
                    #     signal_text = alpha15_info.get('signal', '中性')
                    #     win_rate_info = ""
                    #     if alpha15_info.get('win_rate'):
                    #         wr = alpha15_info['win_rate']
                    #         if wr and isinstance(wr, dict):
                    #             if signal_text == '看多' and wr.get('long_win_rate') is not None:
                    #                 win_rate_info = f" | 看多胜率={wr['long_win_rate']:.2%} (样本={wr.get('long_signals', 0)})"
                    #             elif signal_text == '看空' and wr.get('short_win_rate') is not None:
                    #                 win_rate_info = f" | 看空胜率={wr['short_win_rate']:.2%} (样本={wr.get('short_signals', 0)})"
                    #             if wr.get('overall_win_rate') is not None:
                    #                 win_rate_info += f" | 总体胜率={wr['overall_win_rate']:.2%}"
                    #     print(f"  归一化Alpha#15: 原始值={alpha15_info.get('raw_value', 0):+.6f} | 归一化值={alpha15_info.get('value', 0):+.4f} | {signal_emoji} {signal_text}{win_rate_info}")
                    
                    # 显示其他归一化Alpha因子
                    other_alpha_factors = [
                        'normalized_alpha1', 'normalized_alpha15', 'normalized_alpha3', 'normalized_alpha7', 'normalized_alpha9',
                        'normalized_alpha11', 'normalized_alpha17', 'normalized_alpha21',
                        'normalized_alpha23', 'normalized_alpha25', 'normalized_alpha29',
                        'normalized_alpha33', 'normalized_alpha34'
                    ]
                    
                    for factor_key in other_alpha_factors:
                        if factor_key in factor_results:
                            alpha_info = factor_results[factor_key]
                            alpha_num = factor_key.replace('normalized_alpha', '')
                            signal_emoji = "🟢" if alpha_info.get('signal') == '看多' else "🔴" if alpha_info.get('signal') == '看空' else "⚪"
                            signal_text = alpha_info.get('signal', '中性')
                            win_rate_info = ""
                            if alpha_info.get('win_rate'):
                                wr = alpha_info['win_rate']
                                if wr and isinstance(wr, dict):
                                    if signal_text == '看多' and wr.get('long_win_rate') is not None:
                                        win_rate_info = f" | 看多胜率={wr['long_win_rate']:.2%} (样本={wr.get('long_signals', 0)})"
                                    elif signal_text == '看空' and wr.get('short_win_rate') is not None:
                                        win_rate_info = f" | 看空胜率={wr['short_win_rate']:.2%} (样本={wr.get('short_signals', 0)})"
                                    if wr.get('overall_win_rate') is not None:
                                        win_rate_info += f" | 总体胜率={wr['overall_win_rate']:.2%}"
                            print(f"  归一化Alpha#{alpha_num}: 原始值={alpha_info.get('raw_value', 0):+.6f} | 归一化值={alpha_info.get('value', 0):+.4f} | {signal_emoji} {signal_text}{win_rate_info}")
            except Exception as e:
                logger.debug(f"显示因子分析时出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        print(f"{'='*80}")
        print(f"💰 账户信息:")
        print(f"  可用余额: {available_balance:.2f}")
        print(f"  累计盈亏: {self.total_profit:+.2f}")
        print(f"  运行时间: {runtime_str}")
        print(f"  总交易次数: {self.total_trades} | 盈利: {self.win_trades} | 亏损: {self.loss_trades} | 胜率: {win_rate:.2f}%")
        print(f"{'='*80}\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取交易统计信息
        
        Returns:
            统计信息字典
        """
        runtime = datetime.now() - self.start_time
        win_rate = (self.win_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'total_trades': self.total_trades,
            'win_trades': self.win_trades,
            'loss_trades': self.loss_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit': avg_profit,
            'runtime': str(runtime),
            'completed_trades': self.completed_trades
        }
    
    def print_final_report(self):
        """
        打印最终报告
        """
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("📊 最终交易报告")
        print("="*80)
        print(f"交易对: {self.symbol}")
        print(f"市场类型: {self.market_type}")
        print(f"策略: {self.strategy}")
        print(f"运行时间: {stats['runtime']}")
        print(f"\n📈 交易统计:")
        print(f"  总交易次数: {stats['total_trades']}")
        print(f"  盈利次数: {stats['win_trades']}")
        print(f"  亏损次数: {stats['loss_trades']}")
        print(f"  胜率: {stats['win_rate']:.2f}%")
        print(f"  总盈亏: {stats['total_profit']:+.2f}")
        print(f"  平均盈亏: {stats['avg_profit']:.2f}")
        print("="*80)
        
        # 显示最近10笔交易
        if len(self.completed_trades) > 0:
            print(f"\n最近10笔交易记录:")
            print("-"*80)
            for i, trade in enumerate(self.completed_trades[-10:], 1):
                print(f"{i}. {trade['entry_time']} -> {trade['exit_time']}")
                print(f"   入场: {trade['entry_price']:.2f} | 出场: {trade['exit_price']:.2f}")
                print(f"   盈亏: {trade['profit_amount']:+.2f} ({trade['profit_pct']:+.2f}%)")
            print("="*80)
    
    async def start(self):
        """
        启动实盘交易
        """
        print("="*80)
        print("🚀 实盘交易机器人启动")
        print("="*80)
        print(f"交易对: {self.symbol}")
        print(f"市场类型: {self.market_type}")
        print(f"时间间隔: {self.interval}")
        print(f"策略: {self.strategy}")
        print(f"仓位大小: {self.position_size_pct * 100:.0f}%")
        print(f"止盈: {self.take_profit * 100:.0f}%")
        print(f"止损: {self.stop_loss * 100:.0f}%")
        print(f"模式: {'模拟模式' if self.dry_run else '实盘模式'}")
        print("="*80)
        
        # 显示账户余额
        print("\n账户余额:")
        if self.market_type == "futures":
            show_futures_balances()
        else:
            show_spot_balances()
        
        print("\n等待实时数据...\n")
        
        await self.tracker.run_forever()


async def test_real_time_trading():
    """
    测试实盘交易
    """
    # 创建实盘交易机器人
    # ⚠️ 警告：设置 dry_run=False 将执行真实交易！
    bot = RealTimeTradingBot(
        symbol="ETHUSDT",
        interval="5m",
        lookback_periods=800,
        market_type="futures",  # 或 "spot"
        position_size_pct=0.01,
        take_profit=0.1,
        stop_loss=0.1,
        strategy="ma5",  # 可选: ma5, ma_cross, ma_factor, rsi_factor, alpha1
        min_order_quantity=0.00001,
        ssl_verify=False,
        dry_run=True  # ⚠️ 设置为 False 将执行真实交易！
    )
    
    try:
        # 启动交易
        await bot.start()
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在停止...")
    finally:
        # 打印最终报告
        bot.print_final_report()


def main():
    """
    主函数
    """
    set_user()
    print("="*80)
    print("实盘交易脚本")
    print("="*80)
    print("\n⚠️  重要提示：")
    print("  1. 此脚本会执行真实交易（当 dry_run=False 时）")
    print("  2. 建议先在模拟模式下测试（dry_run=True）")
    print("  3. 确保策略已经过充分回测")
    print("  4. 设置合理的止盈止损")
    print("  5. 确保账户有足够的余额")
    print("  6. 按 Ctrl+C 停止交易")
    print()
    
    # 确认是否继续
    if not os.getenv("AUTO_CONFIRM"):
        response = input("是否继续？(yes/no): ")
        if response.lower() != "yes":
            print("已取消")
            return
    
    try:
        asyncio.run(test_real_time_trading())
    except KeyboardInterrupt:
        print("\n交易已停止")
    except Exception as e:
        print(f"\n交易过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

