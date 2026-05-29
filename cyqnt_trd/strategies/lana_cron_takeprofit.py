"""
拉娜風格 Cron 定時止盈任務

每 15 分鐘執行一次：
1. 掃描持有標的
2. 結合現價、成本、合約數據分析
3. 判斷漲跌概率
4. 觸發分批止盈（高收益標的每次 1-5%）

注意：此腳本需要配合 OpenClaw cron 工具使用
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


# ========== 配置區 ==========
CHECK_INTERVAL_MINUTES = 15
TAKE_PROFIT_THRESHOLDS = {
    "high_gain": {"min_multiple": 2.0, "sell_ratio": 0.05},  # 200%+ 收益，每次賣 5%
    "mid_gain": {"min_multiple": 1.5, "sell_ratio": 0.03},   # 150%+ 收益，每次賣 3%
    "low_gain": {"min_multiple": 1.2, "sell_ratio": 0.01},   # 120%+ 收益，每次賣 1%
}

# ========== 數據加載 ==========
def load_positions():
    """從本地加載持有倉位數據"""
    positions_file = Path(__file__).parent / "data" / "positions.json"
    if not positions_file.exists():
        return []
    
    import json
    with open(positions_file, "r") as f:
        return json.load(f)


def load_market_data(symbol: str, interval: str = "15m", limit: int = 100):
    """從 Binance API 加載市場數據"""
    from cyqnt_trd.blocks import data
    
    df = data.fetch_klines(symbol, interval, limit=limit)
    return df


# ========== AI 分析模組（模擬） ==========
def analyze_trend_probability(df: pd.DataFrame) -> dict:
    """
    分析繼續上漲 vs 下跌的概率
    
    實際實現應調用 Claude API，這裡用技術指標近似
    
    Returns:
        {
            "up_probability": 0.65,  # 上漲概率
            "down_probability": 0.35,  # 下跌概率
            "confidence": 0.78,  # 置信度
            "reason": "MACD 金叉 + 成交量放大"
        }
    """
    from cyqnt_trd.blocks import indicators as ind, conditions as cond
    
    close = df["close"]
    ma20 = ind.sma(close, 20)
    ma60 = ind.sma(close, 60)
    macd_line, signal_line, _ = ind.macd(close, 6, 13, 5)
    rsi = ind.rsi(close, 14)
    
    # 多頭信號
    bullish_signals = 0
    total_signals = 0
    
    # MA 排列
    if ma20.iloc[-1] > ma60.iloc[-1]:
        bullish_signals += 1
    total_signals += 1
    
    # MACD
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        bullish_signals += 1
    total_signals += 1
    
    # RSI
    if 40 <= rsi.iloc[-1] <= 70:
        bullish_signals += 1
    total_signals += 1
    
    # 價格位置
    if close.iloc[-1] > ma20.iloc[-1]:
        bullish_signals += 1
    total_signals += 1
    
    up_prob = bullish_signals / total_signals
    down_prob = 1 - up_prob
    
    reason_parts = []
    if ma20.iloc[-1] > ma60.iloc[-1]:
        reason_parts.append("MA 多頭排列")
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        reason_parts.append("MACD 金叉")
    if 40 <= rsi.iloc[-1] <= 70:
        reason_parts.append("RSI 健康區間")
    
    return {
        "up_probability": round(up_prob, 2),
        "down_probability": round(down_prob, 2),
        "confidence": round(max(up_prob, down_prob), 2),
        "reason": " + ".join(reason_parts) if reason_parts else "信號不明確"
    }


# ========== 止盈決策引擎 ==========
def should_take_profit(position: dict, current_price: float, analysis: dict) -> dict:
    """
    判斷是否應該止盈
    
    Args:
        position: 倉位信息 {symbol, entry_price, quantity, entry_time}
        current_price: 當前價格
        analysis: AI 分析結果
    
    Returns:
        {
            "should_sell": True,
            "sell_ratio": 0.05,
            "reason": "高收益 + 下跌概率 > 上漲概率"
        }
    """
    entry_price = position["entry_price"]
    gain_multiple = (current_price - entry_price) / entry_price
    
    # 判斷收益等級
    gain_tier = None
    for tier, config in TAKE_PROFIT_THRESHOLDS.items():
        if gain_multiple >= config["min_multiple"]:
            gain_tier = tier
            sell_ratio = config["sell_ratio"]
    
    if gain_tier is None:
        return {"should_sell": False, "reason": "收益未達止盈閾值"}
    
    # AI 概率判斷
    up_prob = analysis["up_probability"]
    down_prob = analysis["down_probability"]
    
    if down_prob > up_prob:
        return {
            "should_sell": True,
            "sell_ratio": sell_ratio,
            "reason": f"{gain_tier}收益 ({gain_multiple*100:.1f}%) + 下跌概率 {down_prob*100:.1f}% > 上漲概率 {up_prob*100:.1f}%"
        }
    
    # 如果上漲概率高，繼續持有
    return {
        "should_sell": False,
        "reason": f"上漲概率 {up_prob*100:.1f}% > 下跌概率 {down_prob*100:.1f}%，繼續持有"
    }


# ========== 主執行函數 ==========
def run_takeprofit_check():
    """
    主執行函數 - 每 15 分鐘調用一次
    
    Returns:
        執行報告（可發送到消息通道）
    """
    positions = load_positions()
    report = []
    
    for pos in positions:
        symbol = pos["symbol"]
        
        # 加載市場數據
        df = load_market_data(symbol, interval="15m", limit=100)
        current_price = df["close"].iloc[-1]
        
        # AI 分析
        analysis = analyze_trend_probability(df)
        
        # 止盈決策
        decision = should_take_profit(pos, current_price, analysis)
        
        report.append({
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "gain_multiple": f"{(current_price - pos['entry_price']) / pos['entry_price'] * 100:.1f}%",
            "analysis": analysis,
            "decision": decision
        })
    
    return report


# ========== Cron 任務入口 ==========
if __name__ == "__main__":
    """
    作為 Cron 任務執行時的入口
    
    使用方式（OpenClaw cron）：
    openclaw cron add --name 'lana-takeprofit' --every 15m \
      --message '執行拉娜止盈檢查，發送報告到 Jarvis' \
      --channel jarvis --to '<userId>:thread:<threadId>' --announce
    """
    result = run_takeprofit_check()
    
    # 格式化輸出
    print(f"=== 拉娜止盈檢查報告 {datetime.now(timezone.utc).isoformat()} ===\n")
    
    for item in result:
        print(f"📊 {item['symbol']}")
        print(f"   進場價：${item['entry_price']:.4f}")
        print(f"   現價：${item['current_price']:.4f}")
        print(f"   收益：{item['gain_multiple']}")
        print(f"   分析：{item['analysis']['reason']}")
        print(f"   概率：上漲 {item['analysis']['up_probability']*100:.0f}% / 下跌 {item['analysis']['down_probability']*100:.0f}%")
        print(f"   決策：{item['decision']['reason']}")
        print()


# ========== 測試代碼 ==========
def test_strategy():
    """測試策略邏輯"""
    from cyqnt_trd.blocks import data
    
    # 加載測試數據
    df = data.fetch_klines("BTCUSDT", "15m", limit=100)
    
    # 運行信號生成
    from lana_style_momentum import make_signals
    long, short = make_signals(df)
    
    print(f"長線信號數量：{long.sum()}")
    print(f"短線信號數量：{short.sum()}")
    print(f"最近信號時間：{df.index[long][-1] if long.any() else '無'}")


if __name__ == "__test__":
    test_strategy()
