"""資金費率逼倉反轉 —— 需要四個資料源對在同一條時間軸上的策略

策略邏輯
--------
多數策略只讀一張表:OHLCV 進、多空布林出。這支不行,而那正是它的用意 ——
它的論點是**四個來源彼此矛盾**才成立:

1. 價格在跌            (K 線)
2. 資金費率轉負         (衍生品 —— 空方在付錢給多方)
3. 未平倉量在增加       (衍生品 —— 是**新的**空單,不是回補)
4. 主動賣壓            (衍生品 —— 賣得很急)

單看任何一條都是雜訊。四條同時成立描述的是一個**擁擠的空方**:槓桿賣方
在一個每八小時要付資金費率的位置持續加碼。那正是逼空的溫床。

反向(擁擠的多方)則相反,做空。

風控規則(由框架處理)
--------------------
- 止損:2 ATR
- 止盈:3 ATR
- 最長持倉:24 根

`df` 從哪來
-----------
`make_signals(df)` 收到的是一張表。衍生品欄位(`rate` / `oi_value` /
`buy_vol` / `sell_vol`)有兩種來源:

- `--derivatives-dir` 指到 parquet(既有路徑),或
- `to_panel(build_live_bundle(...))` —— 把所有來源攤平到 bar 時間軸,
  對齊用 `available_time`(一根 bar 只看得到它收盤時已可知的值)

**欄位不存在時不出手**。這裡不做降級,理由值得寫下來:降級版本會把三個
條件當成「不參與否決」,四條剩一條,策略就變成「價格動了 1% 就進場」——
實測同一段資料,有衍生品資料出 0 筆(費率 -0.57~+1.0 bps,離門檻很遠,
沒有擁擠部位可逼),抽掉衍生品出 32 筆。**32 筆不是這支策略的訊號**,
是另一支動能策略掛著同一個 BOT_ID 在下單。

通則:**降級不可以放寬條件**。缺一個讀數不等於讀到 0,更不等於那個條件成立。
論點需要四個來源互相矛盾才成立,少一個就驗證不了,那就不要出手 ——
少一筆單的成本,遠低於用錯誤前提開的一筆單。

誠實邊界
--------
`open_interest` 只有約 30 天的公開視窗,Square 社群欄位沒有時點歷史。
所以這支**今天回測不了**:重播會把當下的快照貼到每一根歷史 bar 上。
它能上 paper / live,edge 在前向收集夠久之前只是假設。
"""

from cyqnt_trd.blocks import derivatives as deriv, entry, indicators as ind, strategy

BOT_ID = "funding_squeeze_panel"

# 論點的每一個門檻都在這裡,讀的人可以一眼看完
CONFIG = {
    "lookback": 12,             # 動能與 OI 變化的回看根數
    "move_pct": 1.0,            # 價格至少要動這麼多(%)
    "funding_bps": 3.0,         # 資金費率極端門檻(bps/8h)
    "oi_build_pct": 1.5,        # 未平倉量至少要增加這麼多(%)
    "taker_ratio": 1.5,         # 主動買賣量比門檻
}

#: 論點需要的四個來源,缺一個就驗證不了 —— 對應 register(needs={"derivatives": True})。
#: 這是「不出手」的判準,不是「降級」的判準:見模組 docstring。
REQUIRED_COLUMNS = ("rate", "oi_value", "buy_vol", "sell_vol")


def make_signals(df):
    """回傳 (long, short) 兩條對齊 df 的布林 Series。"""
    import pandas as pd

    c = CONFIG
    n = int(c["lookback"])
    close = df["close"]
    nothing = pd.Series(False, index=df.index)

    # ---- 0. 宣告的來源到了沒 --------------------------------------------
    # 缺欄位時**不出手**。把缺席的條件當成「不參與否決」會讓四條剩一條,
    # 出來的是另一支動能策略掛著這個 BOT_ID 在下單(實測 0 筆 → 32 筆)。
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        import warnings

        warnings.warn(
            "%s: 缺少 %s,不出手。這支的論點需要四個來源同時成立,"
            "少一個就驗證不了 —— 少一筆單好過用錯誤前提開一筆單。"
            "資料來源見模組 docstring(--derivatives-dir 或 live bundle)。"
            % (BOT_ID, "/".join(missing)),
            RuntimeWarning, stacklevel=2)
        return nothing, nothing.copy()

    # ---- 1. 價格動能 ------------------------------------------------------
    move_pct = (close / close.shift(n) - 1.0) * 100.0
    price_down = move_pct <= -float(c["move_pct"])
    price_up = move_pct >= float(c["move_pct"])

    # ---- 2. 資金費率:誰在付錢給誰 ---------------------------------------
    # funding_rate_state 自己會把原始比率 ×10000 換成 bps,門檻才是 bps ——
    # 傳已經換算過的值會乘兩次,任何微小正費率都變極端。
    fstate = deriv.funding_rate_state(
        df["rate"],
        high_threshold_bps=float(c["funding_bps"]),
        low_threshold_bps=-float(c["funding_bps"]),
    )
    shorts_crowded = fstate == "bearish_squeeze"   # 費率負 = 空方付錢
    longs_crowded = fstate == "bullish_squeeze"    # 費率正 = 多方付錢

    # ---- 3. 未平倉量:是新倉不是回補 -------------------------------------
    # as_percent=True:CONFIG 的門檻是給人讀的百分比,所以明確要求同一個單位。
    # 預設回的是 pct_change() 的小數(0.06 = 6%),把 1.5 拿去比等於要求 150%。
    oi_change = deriv.oi_change_pct(df["oi_value"], periods=n, as_percent=True)
    oi_building = oi_change >= float(c["oi_build_pct"])

    # ---- 4. 主動單方向 ----------------------------------------------------
    tstate = deriv.taker_buy_sell_state(
        df["buy_vol"], df["sell_vol"], threshold=float(c["taker_ratio"]))
    selling_hard = tstate == "aggressive_sell"
    buying_hard = tstate == "aggressive_buy"

    # ---- 組合:做多逼空 / 做空逼多 ---------------------------------------
    long = entry.all_of([price_down, shorts_crowded, oi_building, selling_hard])
    short = entry.all_of([price_up, longs_crowded, oi_building, buying_hard])

    return long.fillna(False), short.fillna(False)


strategy.register(
    BOT_ID, make_signals,
    exit_cfg={"type": "atr_stop_tp", "atr_period": 14,
              "stop_mult": 2.0, "tp_mult": 3.0, "max_bars": 24},
    size=0.1,
    needs={"derivatives": True},
)
