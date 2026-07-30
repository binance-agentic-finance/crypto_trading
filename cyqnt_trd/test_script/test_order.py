import os
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from binance_sdk_spot.spot import Spot, ConfigurationRestAPI, SPOT_REST_API_PROD_URL
from binance_sdk_spot.rest_api.models import NewOrderSideEnum, NewOrderTypeEnum, NewOrderTimeInForceEnum

from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
    ConfigurationRestAPI as FuturesConfigurationRestAPI,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    NewOrderSideEnum as FuturesNewOrderSideEnum,
    NewOrderTimeInForceEnum as FuturesNewOrderTimeInForceEnum,
)

from cyqnt_trd.utils.set_user import set_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API 憑證只從環境變數取,沒有預設值。
# 這兩行原本把真實的正式環境合約金鑰當成 os.getenv 的 fallback,
# 等於把可交易的金鑰放進公開 repo,且在環境變數未設時真的會被拿去用。
def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} must be set in the environment; it is deliberately not "
            "defaulted in code — the previous hardcoded pair was exposed "
            "publicly and has to be rotated."
        )
    return value


DEFAULT_API_KEY = _require("API_KEY")
DEFAULT_API_SECRET = _require("API_SECRET")


def get_spot_balance(asset: Optional[str] = None) -> Dict[str, Any]:
    """
    获取现货账户余额
    
    Args:
        asset: 资产名称，如 "BTC", "USDT"，如果为 None 则返回所有资产
    
    Returns:
        余额信息字典
    """
    try:
        spot_config = ConfigurationRestAPI(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
            base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
        )
        
        spot_client = Spot(config_rest_api=spot_config)
        
        response = spot_client.rest_api.get_account(omit_zero_balances=False)
        data = response.data()
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            account_data = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            account_data = data.dict(by_alias=True)
        elif isinstance(data, dict):
            account_data = data
        else:
            account_data = {"raw_data": str(data)}
        
        # 提取余额信息
        balances = {}
        if 'balances' in account_data:
            for balance in account_data['balances']:
                asset_name = balance.get('asset', '')
                free = float(balance.get('free', 0))
                locked = float(balance.get('locked', 0))
                total = free + locked
                
                if asset is None or asset_name == asset:
                    balances[asset_name] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
        
        return {
            "success": True,
            "balances": balances if asset is None else balances.get(asset, {}),
            "account_data": account_data
        }
        
    except Exception as e:
        logger.error(f"获取现货余额失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


def get_futures_balance(asset: Optional[str] = None) -> Dict[str, Any]:
    """
    获取合约账户余额
    
    Args:
        asset: 资产名称，如 "USDT"，如果为 None 则返回所有资产
    
    Returns:
        余额信息字典
    """
    try:
        # 使用 set_user() 函数获取配置，确保使用正确的 API 密钥
        futures_config = set_user(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
        )
        
        futures_client = DerivativesTradingUsdsFutures(config_rest_api=futures_config)
        
        response = futures_client.rest_api.futures_account_balance_v2()
        data = response.data()
        
        # 处理响应数据 - 合约余额API返回的是列表
        balance_list = []
        if isinstance(data, list):
            # 如果是列表，遍历每个元素并转换为字典
            for item in data:
                if hasattr(item, 'model_dump'):
                    balance_list.append(item.model_dump(by_alias=True))
                elif hasattr(item, 'dict'):
                    balance_list.append(item.dict(by_alias=True))
                elif isinstance(item, dict):
                    balance_list.append(item)
                else:
                    # 如果是Pydantic模型，尝试直接访问属性
                    balance_dict = {}
                    if hasattr(item, 'asset'):
                        balance_dict['asset'] = item.asset
                    if hasattr(item, 'balance'):
                        balance_dict['balance'] = item.balance
                    if hasattr(item, 'available_balance'):
                        balance_dict['availableBalance'] = item.available_balance
                    elif hasattr(item, 'availableBalance'):
                        balance_dict['availableBalance'] = item.availableBalance
                    balance_list.append(balance_dict)
        elif isinstance(data, dict):
            balance_list = [data]
        else:
            # 单个对象
            if hasattr(data, 'model_dump'):
                balance_list = [data.model_dump(by_alias=True)]
            elif hasattr(data, 'dict'):
                balance_list = [data.dict(by_alias=True)]
            else:
                balance_list = [{"raw_data": str(data)}]
        
        # 提取余额信息
        balances = {}
        for balance in balance_list:
            # 尝试多种方式获取asset字段
            if isinstance(balance, dict):
                asset_name = balance.get('asset') or balance.get('Asset') or ''
            else:
                asset_name = getattr(balance, 'asset', None) or getattr(balance, 'Asset', None) or ''
            
            if not asset_name:
                continue  # 跳过没有资产名称的项
            
            # 尝试多种方式获取balance字段
            if isinstance(balance, dict):
                balance_amt = (
                    float(balance.get('balance', 0)) if isinstance(balance.get('balance'), (str, int, float)) else
                    float(balance.get('Balance', 0)) if isinstance(balance.get('Balance'), (str, int, float)) else
                    0.0
                )
            else:
                balance_amt = float(getattr(balance, 'balance', 0) or getattr(balance, 'Balance', 0) or 0)
            
            # 尝试多种方式获取available_balance字段
            if isinstance(balance, dict):
                available = (
                    float(balance.get('availableBalance', balance_amt)) if isinstance(balance.get('availableBalance'), (str, int, float)) else
                    float(balance.get('available_balance', balance_amt)) if isinstance(balance.get('available_balance'), (str, int, float)) else
                    float(balance.get('AvailableBalance', balance_amt)) if isinstance(balance.get('AvailableBalance'), (str, int, float)) else
                    balance_amt
                )
            else:
                available = float(
                    getattr(balance, 'availableBalance', None) or 
                    getattr(balance, 'available_balance', None) or 
                    getattr(balance, 'AvailableBalance', None) or 
                    balance_amt
                )
            
            if asset_name:  # 只处理有效的资产名称
                if asset is None or asset_name == asset:
                    balances[asset_name] = {
                        'balance': balance_amt,
                        'available': available
                    }
        
        return {
            "success": True,
            "balances": balances if asset is None else balances.get(asset, {}),
            "balance_data": balance_list
        }
        
    except Exception as e:
        logger.error(f"获取合约余额失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


def check_spot_balance(symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Tuple[bool, str, float]:
    """
    检查现货账户余额是否足够
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        side: 买卖方向，"BUY" 或 "SELL"
        quantity: 数量
        price: 价格（限价单需要）
    
    Returns:
        (是否足够, 错误信息, 可用余额)
    """
    # 从交易对中提取基础货币和报价货币
    # 常见报价货币列表（按长度从长到短排序，避免误匹配）
    quote_assets = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "DAI", "PAX", "TUSD"]
    
    base_asset = ""
    quote_asset = "USDT"  # 默认报价货币
    
    # 尝试匹配报价货币
    for quote in quote_assets:
        if symbol.endswith(quote):
            base_asset = symbol[:-len(quote)]
            quote_asset = quote
            break
    
    # 如果没匹配到，尝试获取交易对信息（需要API调用，这里简化处理）
    if not base_asset:
        # 假设是USDT交易对
        base_asset = symbol.replace("USDT", "")
        quote_asset = "USDT"
    
    balance_result = get_spot_balance()
    
    if not balance_result.get("success"):
        return False, f"无法获取账户余额: {balance_result.get('error')}", 0.0
    
    balances = balance_result.get("balances", {})
    
    if side.upper() == "BUY":
        # 买入需要报价货币（如USDT）
        required = quantity * (price if price else 1.0)  # 限价单用价格，市价单用1.0（实际会更多）
        available = balances.get(quote_asset, {}).get('free', 0.0)
        
        if available < required:
            return False, f"余额不足: 需要 {required} {quote_asset}, 可用 {available} {quote_asset}", available
        return True, "", available
    
    else:  # SELL
        # 卖出需要基础货币（如BTC）
        available = balances.get(base_asset, {}).get('free', 0.0)
        
        if available < quantity:
            return False, f"余额不足: 需要 {quantity} {base_asset}, 可用 {available} {base_asset}", available
        return True, "", available


def get_futures_position_mode() -> Dict[str, Any]:
    """
    获取合约账户的持仓模式
    
    Returns:
        持仓模式信息字典，包含 dualSidePosition (True表示对冲模式，False表示单向模式)
    """
    try:
        # 使用 set_user() 函数获取配置，确保使用正确的 API 密钥
        futures_config = set_user(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
        )
        
        futures_client = DerivativesTradingUsdsFutures(config_rest_api=futures_config)
        
        response = futures_client.rest_api.get_current_position_mode()
        data = response.data()
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            mode_data = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            mode_data = data.dict(by_alias=True)
        elif isinstance(data, dict):
            mode_data = data
        else:
            mode_data = {"raw_data": str(data)}
        
        # 提取持仓模式
        dual_side = mode_data.get('dualSidePosition', False)
        if isinstance(dual_side, str):
            dual_side = dual_side.lower() in ['true', '1', 'yes']
        
        return {
            "success": True,
            "dual_side_position": bool(dual_side),
            "mode": "Hedge Mode" if dual_side else "One-way Mode",
            "mode_data": mode_data
        }
        
    except Exception as e:
        logger.error(f"获取持仓模式失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "dual_side_position": False,  # 默认假设是单向模式
            "mode": "One-way Mode"
        }


def check_futures_balance(symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Tuple[bool, str, float]:
    """
    检查合约账户余额是否足够
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        side: 买卖方向，"BUY" 或 "SELL"
        quantity: 数量
        price: 价格（限价单需要，用于计算所需保证金）
    
    Returns:
        (是否足够, 错误信息, 可用余额)
    """
    # 合约通常使用USDT作为保证金
    balance_result = get_futures_balance("USDT")
    
    if not balance_result.get("success"):
        return False, f"无法获取账户余额: {balance_result.get('error')}", 0.0
    
    balance_info = balance_result.get("balances", {})
    if not balance_info:
        return False, "未找到USDT余额", 0.0
    
    available = balance_info.get('available', 0.0)
    
    # 简单检查：合约需要保证金，这里简化处理
    # 实际应该根据杠杆和价格计算所需保证金
    estimated_margin = quantity * (price if price else 1.0) * 0.1  # 假设10倍杠杆，需要10%保证金
    
    if available < estimated_margin:
        return False, f"余额可能不足: 估算需要 {estimated_margin} USDT 保证金, 可用 {available} USDT", available
    
    return True, "", available


def test_spot_order(
    symbol: str,
    side: str,
    order_type: str = "MARKET",
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: str = "GTC"
) -> Dict[str, Any]:
    """
    测试现货下单接口
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        side: 买卖方向，"BUY" 或 "SELL"
        order_type: 订单类型，"MARKET" 或 "LIMIT"，默认 "MARKET"
        quantity: 数量（对于 MARKET 订单，可以使用 quantity 或 quoteOrderQty）
        price: 价格（LIMIT 订单必需）
        time_in_force: 有效期，"GTC", "IOC", "FOK"，默认 "GTC"
    
    Returns:
        订单响应数据字典
    """
    try:
        # 创建现货客户端配置
        spot_config = ConfigurationRestAPI(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
            base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
        )
        
        # 初始化现货客户端
        spot_client = Spot(config_rest_api=spot_config)
        
        # 验证参数
        if side.upper() not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
        
        if order_type.upper() == "LIMIT" and price is None:
            raise ValueError("Price is required for LIMIT orders")
        
        if order_type.upper() == "MARKET" and quantity is None:
            raise ValueError("Quantity is required for MARKET orders")
        
        # 检查余额
        logger.info("正在检查账户余额...")
        has_balance, balance_msg, available = check_spot_balance(symbol, side, quantity, price)
        if not has_balance:
            logger.warning(f"余额检查失败: {balance_msg}")
            logger.warning(f"可用余额: {available}")
            # 不直接返回错误，让用户决定是否继续
            # return {
            #     "success": False,
            #     "error": balance_msg,
            #     "available_balance": available
            # }
        else:
            logger.info(f"余额检查通过，可用余额: {available}")
        
        # 准备下单参数
        order_params = {
            "symbol": symbol,
            "side": NewOrderSideEnum[side.upper()].value,
            "type": NewOrderTypeEnum[order_type.upper()].value,
        }
        
        # 添加数量
        if quantity is not None:
            order_params["quantity"] = quantity
        
        # 添加价格（LIMIT 订单）
        if order_type.upper() == "LIMIT" and price is not None:
            order_params["price"] = price
            order_params["time_in_force"] = NewOrderTimeInForceEnum[time_in_force.upper()].value
        
        logger.info(f"正在下单 - 现货 {side} {order_type} {symbol}")
        logger.info(f"订单参数: {order_params}")
        
        # 下单
        response = spot_client.rest_api.new_order(**order_params)
        
        # 获取响应数据
        rate_limits = response.rate_limits
        data = response.data()
        
        logger.info(f"下单成功 - Rate Limits: {rate_limits}")
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            order_result = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            order_result = data.dict(by_alias=True)
        elif isinstance(data, dict):
            order_result = data
        else:
            order_result = {"raw_data": str(data)}
        
        logger.info(f"订单响应: {order_result}")
        
        return {
            "success": True,
            "rate_limits": rate_limits,
            "order": order_result
        }
        
    except Exception as e:
        logger.error(f"现货下单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def test_futures_order(
    symbol: str,
    side: str,
    order_type: str = "MARKET",
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    time_in_force: str = "GTC",
    position_side: Optional[str] = None,
    reduce_only: Optional[str] = None
) -> Dict[str, Any]:
    """
    测试合约下单接口
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        side: 买卖方向，"BUY" 或 "SELL"
        order_type: 订单类型，"MARKET" 或 "LIMIT"，默认 "MARKET"
        quantity: 数量
        price: 价格（LIMIT 订单必需）
        time_in_force: 有效期，"GTC", "IOC", "FOK"，默认 "GTC"
        position_side: 持仓方向，"LONG", "SHORT", "BOTH"（仅对冲模式）
        reduce_only: 是否只减仓，"true" 或 "false"
    
    Returns:
        订单响应数据字典
    """
    try:
        # 创建合约客户端配置
        # 使用 set_user() 函数获取配置，确保使用正确的 API 密钥
        futures_config = set_user(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
        )
        
        # 初始化合约客户端
        futures_client = DerivativesTradingUsdsFutures(config_rest_api=futures_config)
        
        # 验证参数
        if side.upper() not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
        
        if order_type.upper() == "LIMIT" and price is None:
            raise ValueError("Price is required for LIMIT orders")
        
        if quantity is None:
            raise ValueError("Quantity is required")
        
        # 检查余额
        logger.info("正在检查账户余额...")
        has_balance, balance_msg, available = check_futures_balance(symbol, side, quantity, price)
        if not has_balance:
            logger.warning(f"余额检查失败: {balance_msg}")
            logger.warning(f"可用余额: {available}")
            # 不直接返回错误，让用户决定是否继续
            # return {
            #     "success": False,
            #     "error": balance_msg,
            #     "available_balance": available
            # }
        else:
            logger.info(f"余额检查通过，可用余额: {available}")
        
        # 检查持仓模式
        logger.info("正在检查持仓模式...")
        position_mode_result = get_futures_position_mode()
        if position_mode_result.get("success"):
            is_hedge_mode = position_mode_result.get("dual_side_position", False)
            mode_name = position_mode_result.get("mode", "Unknown")
            logger.info(f"当前持仓模式: {mode_name}")
            
            # 如果是对冲模式且未指定position_side，自动设置
            if is_hedge_mode and position_side is None:
                # 对冲模式下，BUY对应LONG，SELL对应SHORT
                auto_position_side = "LONG" if side.upper() == "BUY" else "SHORT"
                logger.info(f"对冲模式自动设置 position_side: {auto_position_side}")
                position_side = auto_position_side
        else:
            logger.warning(f"无法获取持仓模式: {position_mode_result.get('error')}")
            logger.warning("假设为单向模式，如果失败请手动指定 position_side")
        
        # 准备下单参数
        order_params = {
            "symbol": symbol,
            "side": FuturesNewOrderSideEnum[side.upper()].value,
            "type": order_type.upper(),  # 合约使用字符串类型
        }
        
        # 添加数量
        order_params["quantity"] = quantity
        
        # 添加价格（LIMIT 订单）
        if order_type.upper() == "LIMIT" and price is not None:
            order_params["price"] = price
            order_params["time_in_force"] = FuturesNewOrderTimeInForceEnum[time_in_force.upper()].value
        
        # 添加持仓方向（对冲模式必需）
        if position_side is not None:
            from binance_sdk_derivatives_trading_usds_futures.rest_api.models import NewOrderPositionSideEnum
            order_params["position_side"] = NewOrderPositionSideEnum[position_side.upper()].value
        
        # 添加只减仓标志（可选）
        if reduce_only is not None:
            order_params["reduce_only"] = reduce_only
        
        logger.info(f"正在下单 - 合约 {side} {order_type} {symbol}")
        logger.info(f"订单参数: {order_params}")
        
        # 下单
        response = futures_client.rest_api.new_order(**order_params)
        
        # 获取响应数据
        rate_limits = response.rate_limits
        data = response.data()
        
        logger.info(f"下单成功 - Rate Limits: {rate_limits}")
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            order_result = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            order_result = data.dict(by_alias=True)
        elif isinstance(data, dict):
            order_result = data
        else:
            order_result = {"raw_data": str(data)}
        
        logger.info(f"订单响应: {order_result}")
        
        return {
            "success": True,
            "rate_limits": rate_limits,
            "order": order_result
        }
        
    except Exception as e:
        logger.error(f"合约下单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def test_spot_buy_market(symbol: str, quantity: float) -> Dict[str, Any]:
    """测试现货市价买入"""
    logger.info("=" * 80)
    logger.info("测试现货市价买入")
    logger.info("=" * 80)
    return test_spot_order(symbol=symbol, side="BUY", order_type="MARKET", quantity=quantity)


def test_spot_sell_market(symbol: str, quantity: float) -> Dict[str, Any]:
    """测试现货市价卖出"""
    logger.info("=" * 80)
    logger.info("测试现货市价卖出")
    logger.info("=" * 80)
    return test_spot_order(symbol=symbol, side="SELL", order_type="MARKET", quantity=quantity)


def test_spot_buy_limit(symbol: str, quantity: float, price: float) -> Dict[str, Any]:
    """测试现货限价买入"""
    logger.info("=" * 80)
    logger.info("测试现货限价买入")
    logger.info("=" * 80)
    return test_spot_order(symbol=symbol, side="BUY", order_type="LIMIT", quantity=quantity, price=price)


def test_spot_sell_limit(symbol: str, quantity: float, price: float) -> Dict[str, Any]:
    """测试现货限价卖出"""
    logger.info("=" * 80)
    logger.info("测试现货限价卖出")
    logger.info("=" * 80)
    return test_spot_order(symbol=symbol, side="SELL", order_type="LIMIT", quantity=quantity, price=price)


def test_futures_buy_market(symbol: str, quantity: float) -> Dict[str, Any]:
    """测试合约市价买入"""
    logger.info("=" * 80)
    logger.info("测试合约市价买入")
    logger.info("=" * 80)
    return test_futures_order(symbol=symbol, side="BUY", order_type="MARKET", quantity=quantity)


def test_futures_sell_market(symbol: str, quantity: float) -> Dict[str, Any]:
    """测试合约市价卖出"""
    logger.info("=" * 80)
    logger.info("测试合约市价卖出")
    logger.info("=" * 80)
    return test_futures_order(symbol=symbol, side="SELL", order_type="MARKET", quantity=quantity)


def test_futures_buy_limit(symbol: str, quantity: float, price: float) -> Dict[str, Any]:
    """测试合约限价买入"""
    logger.info("=" * 80)
    logger.info("测试合约限价买入")
    logger.info("=" * 80)
    return test_futures_order(symbol=symbol, side="BUY", order_type="LIMIT", quantity=quantity, price=price)


def test_futures_sell_limit(symbol: str, quantity: float, price: float) -> Dict[str, Any]:
    """测试合约限价卖出"""
    logger.info("=" * 80)
    logger.info("测试合约限价卖出")
    logger.info("=" * 80)
    return test_futures_order(symbol=symbol, side="SELL", order_type="LIMIT", quantity=quantity, price=price)


def get_spot_open_orders(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    获取现货当前挂单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"，如果为 None 则返回所有交易对的挂单
    
    Returns:
        挂单列表
    """
    try:
        spot_config = ConfigurationRestAPI(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
            base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
        )
        
        spot_client = Spot(config_rest_api=spot_config)
        
        response = spot_client.rest_api.get_open_orders(symbol=symbol)
        data = response.data()
        
        # 处理响应数据
        orders_list = []
        if isinstance(data, list):
            for order in data:
                if hasattr(order, 'model_dump'):
                    orders_list.append(order.model_dump(by_alias=True))
                elif hasattr(order, 'dict'):
                    orders_list.append(order.dict(by_alias=True))
                elif isinstance(order, dict):
                    orders_list.append(order)
        elif isinstance(data, dict):
            orders_list = [data]
        else:
            if hasattr(data, 'model_dump'):
                orders_list = [data.model_dump(by_alias=True)]
            elif hasattr(data, 'dict'):
                orders_list = [data.dict(by_alias=True)]
        
        return {
            "success": True,
            "orders": orders_list,
            "count": len(orders_list)
        }
        
    except Exception as e:
        logger.error(f"获取现货挂单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "orders": [],
            "count": 0
        }


def get_futures_open_orders(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    获取合约当前挂单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"，如果为 None 则返回所有交易对的挂单
    
    Returns:
        挂单列表
    """
    try:
        # 使用 set_user() 函数获取配置，确保使用正确的 API 密钥
        futures_config = set_user(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
        )
        
        futures_client = DerivativesTradingUsdsFutures(config_rest_api=futures_config)
        
        response = futures_client.rest_api.current_all_open_orders(symbol=symbol)
        data = response.data()
        
        # 处理响应数据
        orders_list = []
        if isinstance(data, list):
            for order in data:
                if hasattr(order, 'model_dump'):
                    orders_list.append(order.model_dump(by_alias=True))
                elif hasattr(order, 'dict'):
                    orders_list.append(order.dict(by_alias=True))
                elif isinstance(order, dict):
                    orders_list.append(order)
        elif isinstance(data, dict):
            orders_list = [data]
        else:
            if hasattr(data, 'model_dump'):
                orders_list = [data.model_dump(by_alias=True)]
            elif hasattr(data, 'dict'):
                orders_list = [data.dict(by_alias=True)]
        
        return {
            "success": True,
            "orders": orders_list,
            "count": len(orders_list)
        }
        
    except Exception as e:
        logger.error(f"获取合约挂单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "orders": [],
            "count": 0
        }


def cancel_spot_order(
    symbol: str,
    order_id: Optional[int] = None,
    orig_client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    撤销现货订单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        order_id: 订单ID
        orig_client_order_id: 客户端订单ID
    
    Returns:
        撤单响应数据字典
    """
    try:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("必须提供 order_id 或 orig_client_order_id 之一")
        
        spot_config = ConfigurationRestAPI(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
            base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
        )
        
        spot_client = Spot(config_rest_api=spot_config)
        
        cancel_params = {"symbol": symbol}
        if order_id is not None:
            cancel_params["order_id"] = order_id
        if orig_client_order_id is not None:
            cancel_params["orig_client_order_id"] = orig_client_order_id
        
        logger.info(f"正在撤销现货订单 - {symbol}")
        logger.info(f"撤单参数: {cancel_params}")
        
        response = spot_client.rest_api.delete_order(**cancel_params)
        
        rate_limits = response.rate_limits
        data = response.data()
        
        logger.info(f"撤单成功 - Rate Limits: {rate_limits}")
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            cancel_result = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            cancel_result = data.dict(by_alias=True)
        elif isinstance(data, dict):
            cancel_result = data
        else:
            cancel_result = {"raw_data": str(data)}
        
        logger.info(f"撤单响应: {cancel_result}")
        
        return {
            "success": True,
            "rate_limits": rate_limits,
            "order": cancel_result
        }
        
    except Exception as e:
        logger.error(f"现货撤单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def cancel_futures_order(
    symbol: str,
    order_id: Optional[int] = None,
    orig_client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    撤销合约订单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
        order_id: 订单ID
        orig_client_order_id: 客户端订单ID
    
    Returns:
        撤单响应数据字典
    """
    try:
        if order_id is None and orig_client_order_id is None:
            raise ValueError("必须提供 order_id 或 orig_client_order_id 之一")
        
        # 使用 set_user() 函数获取配置，确保使用正确的 API 密钥
        futures_config = set_user(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
        )
        
        futures_client = DerivativesTradingUsdsFutures(config_rest_api=futures_config)
        
        cancel_params = {"symbol": symbol}
        if order_id is not None:
            cancel_params["order_id"] = order_id
        if orig_client_order_id is not None:
            cancel_params["orig_client_order_id"] = orig_client_order_id
        
        logger.info(f"正在撤销合约订单 - {symbol}")
        logger.info(f"撤单参数: {cancel_params}")
        
        response = futures_client.rest_api.cancel_order(**cancel_params)
        
        rate_limits = response.rate_limits
        data = response.data()
        
        logger.info(f"撤单成功 - Rate Limits: {rate_limits}")
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            cancel_result = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            cancel_result = data.dict(by_alias=True)
        elif isinstance(data, dict):
            cancel_result = data
        else:
            cancel_result = {"raw_data": str(data)}
        
        logger.info(f"撤单响应: {cancel_result}")
        
        return {
            "success": True,
            "rate_limits": rate_limits,
            "order": cancel_result
        }
        
    except Exception as e:
        logger.error(f"合约撤单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def cancel_all_spot_orders(symbol: str) -> Dict[str, Any]:
    """
    撤销现货某个交易对的所有挂单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"
    
    Returns:
        撤单响应数据字典
    """
    try:
        spot_config = ConfigurationRestAPI(
            api_key=DEFAULT_API_KEY,
            api_secret=DEFAULT_API_SECRET,
            base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
        )
        
        spot_client = Spot(config_rest_api=spot_config)
        
        logger.info(f"正在撤销现货所有挂单 - {symbol}")
        
        response = spot_client.rest_api.delete_open_orders(symbol=symbol)
        
        rate_limits = response.rate_limits
        data = response.data()
        
        logger.info(f"批量撤单成功 - Rate Limits: {rate_limits}")
        
        # 处理响应数据
        if hasattr(data, 'model_dump'):
            cancel_result = data.model_dump(by_alias=True)
        elif hasattr(data, 'dict'):
            cancel_result = data.dict(by_alias=True)
        elif isinstance(data, dict):
            cancel_result = data
        else:
            cancel_result = {"raw_data": str(data)}
        
        logger.info(f"批量撤单响应: {cancel_result}")
        
        return {
            "success": True,
            "rate_limits": rate_limits,
            "result": cancel_result
        }
        
    except Exception as e:
        logger.error(f"现货批量撤单失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def show_spot_open_orders(symbol: Optional[str] = None):
    """
    显示现货当前挂单
    
    Args:
        symbol: 交易对，如 "BTCUSDT"，如果为 None 则显示所有交易对的挂单
    """
    print("\n" + "=" * 80)
    print(f"现货当前挂单" + (f" - {symbol}" if symbol else " (所有交易对)"))
    print("=" * 80)
    
    result = get_spot_open_orders(symbol)
    
    if not result.get("success"):
        print(f"获取挂单失败: {result.get('error')}")
        return
    
    orders = result.get("orders", [])
    
    if not orders:
        print("当前没有挂单")
        return
    
    print(f"共有 {len(orders)} 个挂单：")
    print(f"{'订单ID':>15} {'客户端订单ID':>25} {'交易对':>12} {'方向':>6} {'类型':>8} {'数量':>15} {'价格':>15} {'状态':>10}")
    print("-" * 120)
    
    for order in orders:
        order_id = order.get('orderId', order.get('order_id', ''))
        client_order_id = order.get('clientOrderId', order.get('client_order_id', ''))
        symbol_name = order.get('symbol', '')
        side = order.get('side', '')
        order_type = order.get('type', order.get('order_type', ''))
        quantity = order.get('origQty', order.get('orig_qty', order.get('quantity', 0)))
        price = order.get('price', 0)
        status = order.get('status', '')
        
        print(f"{order_id:>15} {client_order_id:>25} {symbol_name:>12} {side:>6} {order_type:>8} {quantity:>15} {price:>15} {status:>10}")
    
    print("=" * 80)


def get_futures_order_ids(symbol: Optional[str] = None) -> list:
    """
    获取合约当前挂单的订单ID列表
    
    Args:
        symbol: 交易对，如 "BTCUSDT"，如果为 None 则返回所有交易对的挂单ID
    
    Returns:
        订单ID列表
    """
    result = get_futures_open_orders(symbol)
    
    if not result.get("success"):
        logger.warning(f"获取挂单失败: {result.get('error')}")
        return []
    
    orders = result.get("orders", [])
    order_ids = []
    
    for order in orders:
        order_id = order.get('orderId', order.get('order_id', ''))
        if order_id:
            order_ids.append(order_id)
    
    return order_ids


def show_futures_open_orders(symbol: Optional[str] = None) -> list:
    """
    显示合约当前挂单并返回订单ID列表
    
    Args:
        symbol: 交易对，如 "BTCUSDT"，如果为 None 则显示所有交易对的挂单
    
    Returns:
        订单ID列表
    """
    print("\n" + "=" * 80)
    print(f"合约当前挂单" + (f" - {symbol}" if symbol else " (所有交易对)"))
    print("=" * 80)
    
    result = get_futures_open_orders(symbol)
    
    if not result.get("success"):
        print(f"获取挂单失败: {result.get('error')}")
        return []
    
    orders = result.get("orders", [])
    
    if not orders:
        print("当前没有挂单")
        return []
    
    print(f"共有 {len(orders)} 个挂单：")
    print(f"{'订单ID':>15} {'客户端订单ID':>25} {'交易对':>12} {'方向':>6} {'类型':>8} {'数量':>15} {'价格':>15} {'持仓方向':>10} {'状态':>10}")
    print("-" * 130)
    
    order_ids = []
    for order in orders:
        order_id = order.get('orderId', order.get('order_id', ''))
        client_order_id = order.get('clientOrderId', order.get('client_order_id', ''))
        symbol_name = order.get('symbol', '')
        side = order.get('side', '')
        order_type = order.get('type', order.get('order_type', ''))
        quantity = order.get('origQty', order.get('orig_qty', order.get('quantity', 0)))
        price = order.get('price', 0)
        position_side = order.get('positionSide', order.get('position_side', ''))
        status = order.get('status', '')
        
        if order_id:
            order_ids.append(order_id)
        
        print(f"{order_id:>15} {client_order_id:>25} {symbol_name:>12} {side:>6} {order_type:>8} {quantity:>15} {price:>15} {position_side:>10} {status:>10}")
    
    print("=" * 80)
    
    if order_ids:
        print(f"\n订单ID列表: {order_ids}")
        print(f"订单ID数量: {len(order_ids)}")
    
    return order_ids


def show_spot_balances(assets: Optional[list] = None):
    """
    显示现货账户余额
    
    Args:
        assets: 要显示的资产列表，如 ["BTC", "USDT"]，如果为 None 则显示所有非零余额
    """
    print("\n" + "=" * 80)
    print("现货账户余额")
    print("=" * 80)
    
    result = get_spot_balance()
    
    if not result.get("success"):
        print(f"获取余额失败: {result.get('error')}")
        return
    
    balances = result.get("balances", {})
    
    if assets:
        # 显示指定资产
        for asset in assets:
            balance_info = balances.get(asset, {})
            if balance_info:
                free = balance_info.get('free', 0)
                locked = balance_info.get('locked', 0)
                total = balance_info.get('total', 0)
                print(f"{asset:>8} - 可用: {free:>20.8f}, 锁定: {locked:>20.8f}, 总计: {total:>20.8f}")
            else:
                print(f"{asset:>8} - 余额为 0")
    else:
        # 显示所有非零余额
        print(f"{'资产':>8} {'可用余额':>25} {'锁定余额':>25} {'总余额':>25}")
        print("-" * 80)
        for asset, balance_info in sorted(balances.items()):
            free = balance_info.get('free', 0)
            locked = balance_info.get('locked', 0)
            total = balance_info.get('total', 0)
            if total > 0:  # 只显示非零余额
                print(f"{asset:>8} {free:>25.8f} {locked:>25.8f} {total:>25.8f}")
    
    print("=" * 80)


def show_futures_balances(assets: Optional[list] = None):
    """
    显示合约账户余额
    
    Args:
        assets: 要显示的资产列表，如 ["USDT"]，如果为 None 则显示所有非零余额
    """
    print("\n" + "=" * 80)
    print("合约账户余额")
    print("=" * 80)
    
    result = get_futures_balance()
    
    if not result.get("success"):
        print(f"获取余额失败: {result.get('error')}")
        return
    
    balances = result.get("balances", {})
    
    if assets:
        # 显示指定资产
        for asset in assets:
            balance_info = balances.get(asset, {})
            if balance_info:
                balance = balance_info.get('balance', 0)
                available = balance_info.get('available', 0)
                print(f"{asset:>8} - 余额: {balance:>20.8f}, 可用: {available:>20.8f}")
            else:
                print(f"{asset:>8} - 余额为 0")
    else:
        # 显示所有非零余额
        print(f"{'资产':>8} {'余额':>25} {'可用余额':>25}")
        print("-" * 80)
        for asset, balance_info in sorted(balances.items()):
            balance = balance_info.get('balance', 0)
            available = balance_info.get('available', 0)
            if balance > 0:  # 只显示非零余额
                print(f"{asset:>8} {balance:>25.8f} {available:>25.8f}")
    
    print("=" * 80)


def main():
    """主函数 - 运行所有测试"""
    print("\n" + "=" * 80)
    print("Binance 下单接口测试")
    print("=" * 80)
    
    # 显示账户余额
    show_spot_balances()
    show_futures_balances()
    
    # 显示当前挂单
    show_spot_open_orders()
    show_futures_open_orders()
    
    # 测试参数（请根据实际情况修改）
    test_symbol = "BNBUSDT"
    test_quantity = 0.01  # 测试数量（请根据实际情况调整）
    test_price = 500.0  # 测试价格（请根据实际情况调整）
    
    results = []
    
    # 测试现货买卖
    print("\n" + "-" * 80)
    print("现货交易测试")
    print("-" * 80)
    
    # 注意：以下测试会实际下单，请谨慎使用！
    # 建议先使用测试网络或小额资金测试
    
    # 1. 现货市价买入（注释掉以避免实际下单）
    # result = test_spot_buy_market(test_symbol, test_quantity)
    # results.append(("现货市价买入", result))
    
    # 2. 现货市价卖出（注释掉以避免实际下单）
    # result = test_spot_sell_market(test_symbol, test_quantity)
    # results.append(("现货市价卖出", result))
    
    # 3. 现货限价买入（注释掉以避免实际下单）
    # result = test_spot_buy_limit(test_symbol, test_quantity, test_price)
    # results.append(("现货限价买入", result))
    
    # 4. 现货限价卖出（注释掉以避免实际下单）
    # result = test_spot_sell_limit(test_symbol, test_quantity, test_price)
    # results.append(("现货限价卖出", result))
    
    # 测试合约买卖
    print("\n" + "-" * 80)
    print("合约交易测试")
    print("-" * 80)
    
    # 5. 合约市价买入（注释掉以避免实际下单）
    # result = test_futures_buy_market(test_symbol, test_quantity)
    # results.append(("合约市价买入", result))
    
    # 6. 合约市价卖出（注释掉以避免实际下单）
    # result = test_futures_sell_market(test_symbol, test_quantity)
    # results.append(("合约市价卖出", result))
    
    # 7. 合约限价买入（注释掉以避免实际下单）
    # result = test_futures_buy_limit(test_symbol, test_quantity, test_price)
    # results.append(("合约限价买入", result))
    
    # 8. 合约限价卖出（注释掉以避免实际下单）
    # result = test_futures_sell_limit(test_symbol, test_quantity, test_price)
    # results.append(("合约限价卖出", result))
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print("\n注意：所有测试函数已注释，以避免实际下单。")
    print("如需测试，请取消相应函数的注释，并确保：")
    print("1. 使用测试网络或小额资金")
    print("2. 检查交易对、数量和价格参数")
    print("3. 确保账户有足够的余额")
    print("=" * 80)
    
    # 打印结果摘要
    if results:
        print("\n测试结果摘要：")
        for test_name, result in results:
            status = "✓ 成功" if result.get("success") else "✗ 失败"
            print(f"{test_name}: {status}")
            if not result.get("success"):
                print(f"  错误: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # 示例：单独测试某个功能
    # 取消注释以下行来测试特定功能
    
    # 测试现货市价买入
    # test_spot_buy_market("BTCUSDT", 0.001)
    
    # 测试现货市价卖出
    # test_spot_sell_market("BTCUSDT", 0.001)
    
    # 测试现货限价买入
    # test_spot_buy_limit("BTCUSDT", 0.001, 50000.0)
    
    # 测试现货限价卖出
    # test_spot_sell_limit("BTCUSDT", 0.001, 50000.0)
    
    # 测试合约市价买入
    # test_futures_buy_market("BTCUSDT", 0.001)
    
    # 测试合约市价卖出
    # test_futures_sell_market("BTCUSDT", 0.001)
    
    # 测试合约限价买入
    # test_futures_buy_limit("BTCUSDT", 0.001, 50000.0)
    
    # 测试合约限价卖出
    test_futures_sell_limit("BNBUSDT", 0.01, 1000.0)
    
    # 撤单功能示例
    # 查看当前挂单
    # show_spot_open_orders("BTCUSDT")
    # show_futures_open_orders("BNBUSDT")
    
    # 获取合约订单ID列表
    # order_ids = get_futures_order_ids("BNBUSDT")
    # print(f"订单ID列表: {order_ids}")
    
    # 或者显示并获取订单ID
    # order_ids = show_futures_open_orders("BNBUSDT")
    # if order_ids:
    #     print(f"\n获取到的订单ID: {order_ids}")
    #     cancel_futures_order("BNBUSDT", order_id=order_ids[0])

    # 撤销现货订单（通过订单ID）
    # cancel_spot_order("BNBUSDT", order_id=order_ids[0])
    
    # 撤销现货订单（通过客户端订单ID）
    # cancel_spot_order("BTCUSDT", orig_client_order_id="my_order_123")
    
    # 撤销合约订单（通过订单ID）
    # cancel_futures_order("BTCUSDT", order_id=12345678)
    
    # 撤销合约订单（通过客户端订单ID）
    # cancel_futures_order("BTCUSDT", orig_client_order_id="my_order_123")
    
    # 撤销现货某个交易对的所有挂单
    # cancel_all_spot_orders("BTCUSDT")
    
    # 运行主函数
    main()

