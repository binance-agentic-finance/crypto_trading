"""
实时价格跟踪脚本

通过 WebSocket 实时跟踪当前价格数据，并向前追溯 n 个周期，
为实时计算 signal 和计算 strategy 做准备。

使用方法：
    # 方式1: 作为模块运行（推荐）
    cd /Users/user/Desktop/repo/cyqnt_trd
    python -m cyqnt_trd.test_script.realtime_price_tracker
    
    # 方式2: 直接运行脚本
    cd /Users/user/Desktop/repo/cyqnt_trd
    python cyqnt_trd/test_script/realtime_price_tracker.py
"""

import os
import sys
import asyncio
import logging
import ssl
import pandas as pd
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from collections import deque

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 Binance SDK
try:
    from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
        DerivativesTradingUsdsFutures,
        ConfigurationRestAPI,
        ConfigurationWebSocketStreams,
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
        DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
    )
    from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
        KlineCandlestickDataIntervalEnum,
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n提示：请确保已安装 binance-connector-python")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RealtimePriceTracker:
    """
    实时价格跟踪器
    
    通过 WebSocket 实时接收价格数据，并维护一个包含历史 n 个周期的 DataFrame，
    为实时计算 signal 和 strategy 做准备。
    """
    
    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        lookback_periods: int = 100,
        market_type: str = "futures",
        ssl_verify: bool = True
    ):
        """
        初始化实时价格跟踪器
        
        Args:
            symbol: 交易对符号，例如 'BTCUSDT', 'ETHUSDT'
            interval: 时间间隔，例如 '1m', '5m', '1h', '1d'
                     可选值: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            lookback_periods: 向前追溯的周期数（默认100）
            market_type: 市场类型，'futures' 或 'spot'（默认 'futures'）
            ssl_verify: 是否验证 SSL 证书（默认 True）。如果遇到 SSL 证书验证错误，可以设置为 False（仅用于开发/测试）
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.lookback_periods = lookback_periods
        self.market_type = market_type
        self.ssl_verify = ssl_verify
        
        # 数据存储
        self.data_df: Optional[pd.DataFrame] = None
        self.latest_kline: Optional[Dict[str, Any]] = None
        
        # WebSocket 连接
        self.connection = None
        self.stream = None
        self.is_running = False
        
        # 回调函数
        self.on_new_kline_callbacks: list = []
        self.on_data_updated_callbacks: list = []
        
        # 初始化 REST API 客户端（用于获取历史数据）
        if market_type == "futures":
            # Credentials come from the environment ONLY. They used to be
            # hardcoded as os.getenv() *defaults*, which meant a real production
            # futures key/secret sat in this public repository and was used
            # whenever the env vars were unset. Fail loudly instead.
            api_key = os.getenv("API_KEY", "").strip()
            api_secret = os.getenv("API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise RuntimeError(
                    "API_KEY / API_SECRET must be set in the environment. "
                    "They are deliberately not defaulted in code — the previous "
                    "hardcoded pair was exposed publicly and has to be rotated."
                )
            self.rest_config = ConfigurationRestAPI(
                api_key=api_key,
                api_secret=api_secret,
                base_path=os.getenv(
                    "BASE_PATH", DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
                ),
            )
            self.rest_client = DerivativesTradingUsdsFutures(config_rest_api=self.rest_config)
        else:
            raise NotImplementedError("目前只支持期货市场")
        
        # 初始化 WebSocket 客户端
        # 配置 SSL 上下文
        ssl_context = None
        if not ssl_verify:
            # 创建不验证证书的 SSL 上下文（仅用于开发/测试）
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logging.warning("SSL 证书验证已禁用（仅用于开发/测试环境）")
        
        self.ws_config = ConfigurationWebSocketStreams(
            stream_url=os.getenv(
                "STREAM_URL", DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL
            ),
            https_agent=ssl_context
        )
        self.ws_client = DerivativesTradingUsdsFutures(config_ws_streams=self.ws_config)
        
        # 间隔映射
        self.interval_map = {
            "1m": KlineCandlestickDataIntervalEnum.INTERVAL_1m,
            "3m": KlineCandlestickDataIntervalEnum.INTERVAL_3m,
            "5m": KlineCandlestickDataIntervalEnum.INTERVAL_5m,
            "15m": KlineCandlestickDataIntervalEnum.INTERVAL_15m,
            "30m": KlineCandlestickDataIntervalEnum.INTERVAL_30m,
            "1h": KlineCandlestickDataIntervalEnum.INTERVAL_1h,
            "2h": KlineCandlestickDataIntervalEnum.INTERVAL_2h,
            "4h": KlineCandlestickDataIntervalEnum.INTERVAL_4h,
            "6h": KlineCandlestickDataIntervalEnum.INTERVAL_6h,
            "8h": KlineCandlestickDataIntervalEnum.INTERVAL_8h,
            "12h": KlineCandlestickDataIntervalEnum.INTERVAL_12h,
            "1d": KlineCandlestickDataIntervalEnum.INTERVAL_1d,
            "3d": KlineCandlestickDataIntervalEnum.INTERVAL_3d,
            "1w": KlineCandlestickDataIntervalEnum.INTERVAL_1w,
            "1M": KlineCandlestickDataIntervalEnum.INTERVAL_1M,
        }
        
        if interval not in self.interval_map:
            raise ValueError(f"不支持的间隔: {interval}")
    
    def _kline_to_dict(self, kline_data: list) -> Dict[str, Any]:
        """
        将 K 线数据转换为字典格式
        
        Args:
            kline_data: K 线数据列表，格式为 [open_time, open, high, low, close, volume, ...]
        
        Returns:
            字典格式的 K 线数据
        """
        open_time = int(kline_data[0]) if isinstance(kline_data[0], str) else kline_data[0]
        close_time = int(kline_data[6]) if isinstance(kline_data[6], str) else kline_data[6]
        
        return {
            'open_time': open_time,
            'open_time_str': datetime.fromtimestamp(open_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open_price': float(kline_data[1]),
            'high_price': float(kline_data[2]),
            'low_price': float(kline_data[3]),
            'close_price': float(kline_data[4]),
            'volume': float(kline_data[5]),
            'close_time': close_time,
            'close_time_str': datetime.fromtimestamp(close_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'quote_volume': float(kline_data[7]),
            'trades': int(kline_data[8]),
            'taker_buy_base_volume': float(kline_data[9]),
            'taker_buy_quote_volume': float(kline_data[10]),
            'ignore': kline_data[11] if len(kline_data) > 11 else "0"
        }
    
    def _dict_to_dataframe_row(self, kline_dict: Dict[str, Any]) -> pd.Series:
        """
        将 K 线字典转换为 DataFrame 行
        
        Args:
            kline_dict: K 线字典
        
        Returns:
            pandas Series
        """
        return pd.Series({
            'datetime': pd.to_datetime(kline_dict['open_time_str']),
            'open_time': kline_dict['open_time'],
            'open_time_str': kline_dict['open_time_str'],
            'open_price': kline_dict['open_price'],
            'high_price': kline_dict['high_price'],
            'low_price': kline_dict['low_price'],
            'close_price': kline_dict['close_price'],
            'volume': kline_dict['volume'],
            'close_time': kline_dict['close_time'],
            'close_time_str': kline_dict['close_time_str'],
            'quote_volume': kline_dict['quote_volume'],
            'trades': kline_dict['trades'],
            'taker_buy_base_volume': kline_dict['taker_buy_base_volume'],
            'taker_buy_quote_volume': kline_dict['taker_buy_quote_volume'],
            'ignore': kline_dict['ignore']
        })
    
    async def load_historical_data(self) -> bool:
        """
        加载历史数据
        
        Returns:
            是否成功加载
        """
        try:
            logging.info(f"正在加载 {self.symbol} 的历史数据，间隔: {self.interval}, 周期数: {self.lookback_periods}")
            
            interval_enum = self.interval_map[self.interval]
            
            # 查询历史 K 线数据
            response = self.rest_client.rest_api.kline_candlestick_data(
                symbol=self.symbol,
                interval=interval_enum,
                limit=self.lookback_periods
            )
            
            klines_data = response.data()
            
            if not klines_data:
                logging.warning("未获取到历史数据")
                return False
            
            # 转换为 DataFrame
            data_list = []
            for kline in klines_data:
                kline_dict = self._kline_to_dict(kline)
                data_list.append(self._dict_to_dataframe_row(kline_dict))
            
            self.data_df = pd.DataFrame(data_list)
            self.data_df = self.data_df.sort_values('datetime').reset_index(drop=True)
            
            logging.info(f"成功加载 {len(self.data_df)} 条历史数据")
            logging.info(f"数据时间范围: {self.data_df.iloc[0]['open_time_str']} 至 {self.data_df.iloc[-1]['open_time_str']}")
            
            return True
            
        except Exception as e:
            logging.error(f"加载历史数据时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _handle_kline_message(self, data: Any):
        """
        处理 WebSocket 接收到的 K 线消息
        
        Args:
            data: WebSocket 消息数据（可能是字典或 Pydantic 模型对象）
        """
        try:
            # 如果 data 是 Pydantic 模型对象，转换为字典
            if hasattr(data, 'model_dump'):
                data = data.model_dump(by_alias=True)
            elif hasattr(data, 'dict'):
                data = data.dict(by_alias=True)
            elif not isinstance(data, dict):
                # 尝试转换为字典
                data = dict(data) if hasattr(data, '__dict__') else data
            
            # WebSocket 消息格式: {"e": "kline", "k": {...}}
            if 'k' in data:
                kline_info = data['k']
                
                # 如果 kline_info 是 Pydantic 模型对象，转换为字典
                if hasattr(kline_info, 'model_dump'):
                    kline_info = kline_info.model_dump(by_alias=True)
                elif hasattr(kline_info, 'dict'):
                    kline_info = kline_info.dict(by_alias=True)
                elif not isinstance(kline_info, dict):
                    kline_info = dict(kline_info) if hasattr(kline_info, '__dict__') else kline_info
                
                # 检查是否是新的 K 线（is_closed = True）
                is_closed = kline_info.get('x', False)  # x 表示 K 线是否已关闭
                
                if is_closed:
                    # 构建 K 线数据列表（与 REST API 格式一致）
                    # 注意：字段值可能是字符串或数字，需要统一处理
                    kline_data = [
                        int(kline_info.get('t', 0)),  # open_time
                        str(kline_info.get('o', '0')),  # open
                        str(kline_info.get('h', '0')),  # high
                        str(kline_info.get('l', '0')),  # low
                        str(kline_info.get('c', '0')),  # close
                        str(kline_info.get('v', '0')),  # volume
                        int(kline_info.get('T', 0)),  # close_time
                        str(kline_info.get('q', '0')),  # quote_volume
                        int(kline_info.get('n', 0)),  # trades
                        str(kline_info.get('V', '0')),  # taker_buy_base_volume
                        str(kline_info.get('Q', '0')),  # taker_buy_quote_volume
                        "0"  # ignore
                    ]
                    
                    kline_dict = self._kline_to_dict(kline_data)
                    self.latest_kline = kline_dict
                    
                    # 更新 DataFrame
                    new_row = self._dict_to_dataframe_row(kline_dict)
                    
                    if self.data_df is not None:
                        # 检查是否已存在该时间点的数据
                        existing_idx = self.data_df[
                            self.data_df['open_time'] == kline_dict['open_time']
                        ].index
                        
                        if len(existing_idx) > 0:
                            # 更新现有数据
                            self.data_df.loc[existing_idx[0]] = new_row
                            logging.debug(f"更新 K 线数据: {kline_dict['open_time_str']}")
                        else:
                            # 添加新数据
                            self.data_df = pd.concat([self.data_df, new_row.to_frame().T], ignore_index=True)
                            self.data_df = self.data_df.sort_values('datetime').reset_index(drop=True)
                            
                            # 如果数据超过 lookback_periods，删除最旧的数据
                            if len(self.data_df) > self.lookback_periods:
                                self.data_df = self.data_df.tail(self.lookback_periods).reset_index(drop=True)
                            
                            logging.info(f"新增 K 线数据: {kline_dict['open_time_str']}, 价格: {kline_dict['close_price']}")
                    else:
                        # 如果 DataFrame 为空，创建新的
                        self.data_df = new_row.to_frame().T
                    
                    # 调用回调函数
                    for callback in self.on_new_kline_callbacks:
                        try:
                            callback(kline_dict, self.get_data())
                        except Exception as e:
                            logging.error(f"回调函数执行出错: {e}")
                    
                    for callback in self.on_data_updated_callbacks:
                        try:
                            callback(self.get_data())
                        except Exception as e:
                            logging.error(f"数据更新回调函数执行出错: {e}")
                else:
                    # 未关闭的 K 线，更新最新价格（不添加到 DataFrame）
                    open_time = int(kline_info.get('t', 0))
                    close_time = int(kline_info.get('T', 0))
                    self.latest_kline = {
                        'open_time': open_time,
                        'open_time_str': datetime.fromtimestamp(open_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'open_price': float(kline_info.get('o', '0')),
                        'high_price': float(kline_info.get('h', '0')),
                        'low_price': float(kline_info.get('l', '0')),
                        'close_price': float(kline_info.get('c', '0')),
                        'volume': float(kline_info.get('v', '0')),
                        'close_time': close_time,
                        'close_time_str': datetime.fromtimestamp(close_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'quote_volume': float(kline_info.get('q', '0')),
                        'trades': int(kline_info.get('n', 0)),
                        'taker_buy_base_volume': float(kline_info.get('V', '0')),
                        'taker_buy_quote_volume': float(kline_info.get('Q', '0')),
                        'ignore': "0"
                    }
                    logging.debug(f"更新当前 K 线（未关闭）: {self.latest_kline['close_price']}")
                    
        except Exception as e:
            logging.error(f"处理 K 线消息时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        获取当前的数据 DataFrame
        
        Returns:
            包含历史数据的 DataFrame，如果数据未加载则返回 None
        """
        return self.data_df.copy() if self.data_df is not None else None
    
    def get_latest_price(self) -> Optional[float]:
        """
        获取最新价格
        
        Returns:
            最新价格，如果无数据则返回 None
        """
        if self.latest_kline:
            return self.latest_kline['close_price']
        elif self.data_df is not None and len(self.data_df) > 0:
            return self.data_df.iloc[-1]['close_price']
        return None
    
    def register_on_new_kline(self, callback: Callable[[Dict[str, Any], pd.DataFrame], None]):
        """
        注册新 K 线回调函数
        
        Args:
            callback: 回调函数，接收 (kline_dict, data_df) 作为参数
        """
        self.on_new_kline_callbacks.append(callback)
    
    def register_on_data_updated(self, callback: Callable[[pd.DataFrame], None]):
        """
        注册数据更新回调函数
        
        Args:
            callback: 回调函数，接收 data_df 作为参数
        """
        self.on_data_updated_callbacks.append(callback)
    
    async def start(self):
        """
        启动实时跟踪
        """
        if self.is_running:
            logging.warning("实时跟踪已在运行中")
            return
        
        # 先加载历史数据
        if not await self.load_historical_data():
            logging.error("加载历史数据失败，无法启动实时跟踪")
            return
        
        try:
            # 创建 WebSocket 连接
            self.connection = await self.ws_client.websocket_streams.create_connection()
            
            # 检查连接是否成功
            if self.connection is None:
                error_msg = "WebSocket 连接失败：连接对象为 None。可能是 SSL 证书验证问题。"
                if self.ssl_verify:
                    error_msg += " 提示：如果遇到 SSL 证书验证错误，可以在初始化时设置 ssl_verify=False（仅用于开发/测试）"
                logging.error(error_msg)
                self.is_running = False
                return
            
            # 订阅 K 线流
            self.stream = await self.connection.kline_candlestick_streams(
                symbol=self.symbol.lower(),
                interval=self.interval,
            )
            
            # 检查流是否成功创建
            if self.stream is None:
                logging.error("K 线流订阅失败：流对象为 None")
                self.is_running = False
                return
            
            # 设置消息处理
            self.stream.on("message", self._handle_kline_message)
            
            self.is_running = True
            logging.info(f"实时跟踪已启动: {self.symbol} {self.interval}")
            
        except Exception as e:
            error_msg = f"启动实时跟踪时出错: {e}"
            if "SSL" in str(e) or "certificate" in str(e).lower():
                if self.ssl_verify:
                    error_msg += "\n提示：如果遇到 SSL 证书验证错误，可以在初始化时设置 ssl_verify=False（仅用于开发/测试）"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            self.is_running = False
    
    async def stop(self):
        """
        停止实时跟踪
        """
        if not self.is_running:
            return
        
        try:
            if self.stream:
                await self.stream.unsubscribe()
            
            if self.connection:
                await self.connection.close_connection(close_session=True)
            
            self.is_running = False
            logging.info("实时跟踪已停止")
            
        except Exception as e:
            logging.error(f"停止实时跟踪时出错: {e}")
    
    async def run_forever(self):
        """
        运行实时跟踪（持续运行直到中断）
        """
        await self.start()
        
        try:
            # 保持运行
            while self.is_running:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logging.info("收到中断信号，正在停止...")
        finally:
            await self.stop()


async def example_usage():
    """
    示例用法：实时追踪价格并计算信号
    """
    import sys
    import os
    
    # 导入信号函数（优先尝试直接导入，适用于已安装的package）
    try:
        from cyqnt_trd.trading_signal.signal.ma_signal import ma_signal, ma_cross_signal
        from cyqnt_trd.trading_signal.signal.factor_based_signal import factor_based_signal
        from cyqnt_trd.trading_signal.factor.ma_factor import ma_factor
        from cyqnt_trd.trading_signal.factor.rsi_factor import rsi_factor
        from cyqnt_trd.trading_signal.selected_alpha.alpha1 import alpha1_factor
    except ImportError:
        # 如果直接导入失败，尝试添加项目路径（用于开发模式）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 再次尝试导入
        try:
            from cyqnt_trd.trading_signal.signal.ma_signal import ma_signal, ma_cross_signal
            from cyqnt_trd.trading_signal.signal.factor_based_signal import factor_based_signal
            from cyqnt_trd.trading_signal.factor.ma_factor import ma_factor
            from cyqnt_trd.trading_signal.factor.rsi_factor import rsi_factor
            from cyqnt_trd.trading_signal.selected_alpha.alpha1 import alpha1_factor
        except ImportError as e:
            logging.error(f"导入信号模块失败: {e}")
            logging.error("请确保已安装 package 或项目路径正确")
            logging.error("安装方式: pip install -e .")
            return
    
    # 创建跟踪器
    # 如果遇到 SSL 证书验证错误，可以设置 ssl_verify=False（仅用于开发/测试）
    tracker = RealtimePriceTracker(
        symbol="BTCUSDT",
        interval="1m",
        lookback_periods=100,
        ssl_verify=False  # 如果遇到 SSL 证书验证错误，设置为 False
    )
    
    # 交易状态跟踪
    position = 0.0  # 当前持仓
    entry_price = 0.0  # 入场价格
    entry_index = -1  # 入场索引
    
    # 交易记录和统计
    from datetime import datetime
    initial_capital = 10000.0  # 初始资金
    current_capital = initial_capital  # 当前资金（不包括持仓）
    completed_trades = []  # 已完成的交易记录
    total_trades = 0  # 总交易次数
    win_trades = 0  # 盈利交易次数
    loss_trades = 0  # 亏损交易次数
    total_profit = 0.0  # 累计盈亏（金额）
    start_time = datetime.now()  # 程序开始时间
    
    # 信号计算函数
    def calculate_and_display_signals(kline_dict, data_df):
        """计算并显示所有信号"""
        nonlocal position, entry_price, entry_index
        nonlocal initial_capital, current_capital, completed_trades
        nonlocal total_trades, win_trades, loss_trades, total_profit, start_time
        
        if len(data_df) < 10:  # 确保有足够的数据
            print(f"\n[{kline_dict['open_time_str']}] 数据不足，等待更多数据...")
            return
        
        current_price = kline_dict['close_price']
        current_time = kline_dict['open_time_str']
        
        # 使用足够的数据切片（最后30行，确保有足够的历史数据）
        data_slice = data_df.iloc[-30:].copy() if len(data_df) >= 30 else data_df.copy()
        
        # 计算当前总资产（包括持仓价值）
        if position > 0:
            position_value = position * current_price
            total_assets = current_capital + position_value
            floating_profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            total_assets = current_capital
            floating_profit_pct = 0.0
        
        # 计算整体收益率
        total_return_pct = (total_assets - initial_capital) / initial_capital * 100
        
        # 计算运行时间
        runtime = datetime.now() - start_time
        runtime_str = f"{runtime.days}天 {runtime.seconds // 3600}小时 {(runtime.seconds % 3600) // 60}分钟"
        
        # 计算胜率
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"📊 新 K 线数据")
        print(f"{'='*80}")
        print(f"时间: {current_time}")
        print(f"价格: {current_price:.2f}")
        if position > 0:
            print(f"持仓: {position:.4f} | 入场价: {entry_price:.2f} | 浮动盈亏: {floating_profit_pct:+.2f}%")
        else:
            print(f"持仓: 无")
        print(f"{'='*80}")
        print(f"💰 账户统计:")
        print(f"  初始资金: {initial_capital:.2f}")
        print(f"  当前资金: {current_capital:.2f}")
        if position > 0:
            print(f"  持仓价值: {position_value:.2f}")
        print(f"  总资产: {total_assets:.2f}")
        print(f"  累计盈亏: {total_profit:+.2f} ({total_return_pct:+.2f}%)")
        print(f"  运行时间: {runtime_str}")
        print(f"  总交易次数: {total_trades} | 盈利: {win_trades} | 亏损: {loss_trades} | 胜率: {win_rate:.2f}%")
        print(f"{'='*80}")
        print(f"📈 交易信号:")
        
        # 1. MA 信号（MA5）
        try:
            if len(data_slice) >= 6:
                ma5_signal = ma_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,  # 止盈10%
                    stop_loss=0.05,  # 止损5%
                    period=5
                )
                signal_emoji = "🟢" if ma5_signal == 'buy' else "🔴" if ma5_signal == 'sell' else "⚪"
                print(f"  {signal_emoji} MA5信号: {ma5_signal.upper()}")
        except Exception as e:
            print(f"MA5信号计算失败: {e}")
        
        # 2. MA交叉信号（MA5/MA20）
        try:
            if len(data_slice) >= 22:
                ma_cross_sig = ma_cross_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,
                    stop_loss=0.05,
                    check_periods=1,
                    short_period=5,
                    long_period=20
                )
                signal_emoji = "🟢" if ma_cross_sig == 'buy' else "🔴" if ma_cross_sig == 'sell' else "⚪"
                print(f"  {signal_emoji} MA交叉信号(5/20): {ma_cross_sig.upper()}")
        except Exception as e:
            print(f"MA交叉信号计算失败: {e}")
        
        # 3. 基于因子的信号（MA因子）
        try:
            if len(data_slice) >= 6:
                ma_factor_signal = factor_based_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,
                    stop_loss=0.05,
                    check_periods=1,
                    factor_func=lambda d: ma_factor(d, period=5),
                    factor_period=5
                )
                signal_emoji = "🟢" if ma_factor_signal == 'buy' else "🔴" if ma_factor_signal == 'sell' else "⚪"
                print(f"  {signal_emoji} 基于MA因子的信号: {ma_factor_signal.upper()}")
        except Exception as e:
            print(f"基于MA因子的信号计算失败: {e}")
        
        # 4. 基于因子的信号（RSI因子）
        try:
            if len(data_slice) >= 16:
                rsi_factor_signal = factor_based_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,
                    stop_loss=0.05,
                    check_periods=1,
                    factor_func=lambda d: rsi_factor(d, period=14),
                    factor_period=14
                )
                signal_emoji = "🟢" if rsi_factor_signal == 'buy' else "🔴" if rsi_factor_signal == 'sell' else "⚪"
                print(f"  {signal_emoji} 基于RSI因子的信号: {rsi_factor_signal.upper()}")
        except Exception as e:
            print(f"基于RSI因子的信号计算失败: {e}")
        
        # 5. Alpha#1 因子信号
        try:
            if len(data_slice) >= 26:  # alpha1需要至少25行数据
                alpha1_signal = factor_based_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,
                    stop_loss=0.05,
                    check_periods=1,
                    factor_func=lambda d: alpha1_factor(d, lookback_days=5, stddev_period=20, power=2.0),
                    factor_period=25
                )
                signal_emoji = "🟢" if alpha1_signal == 'buy' else "🔴" if alpha1_signal == 'sell' else "⚪"
                print(f"  {signal_emoji} 基于Alpha#1因子的信号: {alpha1_signal.upper()}")
        except Exception as e:
            print(f"基于Alpha#1因子的信号计算失败: {e}")
        
        # 显示因子值（用于参考）
        print(f"\n📊 因子值:")
        try:
            if len(data_slice) >= 6:
                ma5_factor_val = ma_factor(data_slice, period=5)
                print(f"  MA5因子: {ma5_factor_val:.4f}")
        except:
            pass
        
        try:
            if len(data_slice) >= 16:
                rsi_factor_val = rsi_factor(data_slice, period=14)
                print(f"  RSI因子: {rsi_factor_val:.4f}")
        except:
            pass
        
        try:
            if len(data_slice) >= 26:
                alpha1_factor_val = alpha1_factor(data_slice, lookback_days=5, stddev_period=20, power=2.0)
                print(f"  Alpha#1因子: {alpha1_factor_val:.4f}")
        except:
            pass
        
        print(f"{'='*80}\n")
        
        # 更新持仓状态（示例：使用MA5信号进行交易）
        # 注意：这里只是示例，实际交易需要更完善的逻辑
        if len(data_slice) >= 6:
            try:
                ma5_signal = ma_signal(
                    data_slice=data_slice,
                    position=position,
                    entry_price=entry_price,
                    entry_index=entry_index,
                    take_profit=0.1,
                    stop_loss=0.05,
                    period=5
                )
                
                if ma5_signal == 'buy' and position == 0:
                    # 买入
                    # 计算可买入的数量（使用当前资金的90%）
                    buy_amount = current_capital * 0.9
                    position = buy_amount / current_price
                    entry_price = current_price
                    entry_index = len(data_df) - 1
                    
                    # 更新资金（扣除买入金额）
                    current_capital -= buy_amount
                    
                    print(f"\n{'='*80}")
                    print(f"✅ 执行买入")
                    print(f"  价格: {entry_price:.2f}")
                    print(f"  数量: {position:.4f}")
                    print(f"  金额: {buy_amount:.2f}")
                    print(f"  剩余资金: {current_capital:.2f}")
                    print(f"{'='*80}")
                    
                elif ma5_signal == 'sell' and position > 0:
                    # 卖出
                    sell_amount = position * current_price
                    profit_amount = sell_amount - (position * entry_price)
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    
                    # 更新资金（增加卖出金额）
                    current_capital += sell_amount
                    
                    # 记录交易
                    trade_record = {
                        'entry_time': data_df.iloc[entry_index]['open_time_str'] if entry_index >= 0 else 'N/A',
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': position,
                        'profit_amount': profit_amount,
                        'profit_pct': profit_pct
                    }
                    completed_trades.append(trade_record)
                    
                    # 更新统计
                    total_trades += 1
                    total_profit += profit_amount
                    if profit_amount > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    
                    print(f"\n{'='*80}")
                    print(f"✅ 执行卖出")
                    print(f"  价格: {current_price:.2f}")
                    print(f"  入场价: {entry_price:.2f}")
                    print(f"  数量: {position:.4f}")
                    print(f"  盈亏金额: {profit_amount:+.2f}")
                    print(f"  盈亏比例: {profit_pct:+.2f}%")
                    print(f"  当前资金: {current_capital:.2f}")
                    print(f"{'='*80}")
                    
                    # 重置持仓
                    position = 0.0
                    entry_price = 0.0
                    entry_index = -1
            except Exception as e:
                logging.debug(f"更新持仓状态时出错: {e}")
    
    # 注册回调函数
    tracker.register_on_new_kline(calculate_and_display_signals)
    
    print("="*80)
    print("实时价格跟踪和信号计算已启动")
    print(f"交易对: {tracker.symbol}")
    print(f"时间间隔: {tracker.interval}")
    print(f"历史数据周期数: {tracker.lookback_periods}")
    print("="*80)
    print("\n等待新 K 线数据...\n")
    
    # 运行跟踪器
    await tracker.run_forever()


if __name__ == "__main__":
    asyncio.run(example_usage())

