# 拉娜（Lana）AI 交易系統實現計劃

基於 2026-05-17 訪談內容的技術拆解與落地路線圖

---

## 📊 訪談核心要點回顧

### 拉娜的交易特徵
| 特徵 | 描述 | 實現難度 |
|------|------|---------|
| 勝率 | 個位數（<10%） | - |
| 盈虧比 | 極高（吃長尾收益） | ⭐⭐⭐ |
| 主要市場 | 鏈上標的（99%） | ⭐⭐⭐⭐ |
| 交易頻率 | 低頻，重質量 | ⭐⭐ |
| AI 使用 | Claude 4.6（90% 工作）+ Codex 審計 | ⭐⭐⭐ |

### 系統架構
```
┌─────────────────────────────────────────┐
│          思考層（人類 + AI）              │
│  - 市場情緒分析                           │
│  - 策略框架制定                           │
│  - 標的觀察名單生成                       │
│  - Prompt 迭代優化                        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│          執行層（自動化）                 │
│  - 明確止損/買入點位                      │
│  - 100% 自動交易執行                     │
│  - Cron 定時止盈檢查（每 15 分鐘）          │
│  - 工程容錯（斷線重連）                   │
└─────────────────────────────────────────┘
```

---

## 🎯 實現階段

### 第一階段：基礎策略框架（已完成）✅

**文件**：`lana_style_momentum.py`

**功能**：
- [x] 多因子評分進場系統
- [x] 趨勢 + 動能雙重確認
- [x] 波動率過濾（高波動優先）
- [x] 放寬止損範圍（ATR 2.5 倍）
- [x] 分批止盈規則（註解形式）

**技術要點**：
```python
# 使用 ScoringSystem 實現多因子評分
sys = scoring.ScoringSystem()
sys.add_rule("trend", trend_cond, weight=2.0)
sys.add_rule("vol", vol_cond, weight=1.0)
sys.add_rule("counter", counter_cond, weight=-1.0)

# 評分閾值進場
long = sys.signal(threshold=5.0) & is_trending
```

**測試命令**：
```bash
cd /Users/hankchung/Dev/crypto_trading-main
python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
  --engine python \
  --strategy lana_momentum_v1 \
  --strategy-module cyqnt_trd.strategies.lana_style_momentum \
  --symbol BTCUSDT --interval 15m --limit 500
```

---

### 第二階段：Cron 定時止盈任務（已完成）✅

**文件**：`lana_cron_takeprofit.py`

**功能**：
- [x] 每 15 分鐘掃描持有標的
- [x] AI 概率分析（技術指標近似）
- [x] 分批止盈決策（1-5% 每次）
- [x] 執行報告生成

**待實現**：
- [ ] 接入 Claude API 進行真實概率判斷
- [ ] 本地倉位數據持久化（positions.json）
- [ ] 與交易所 API 對接執行賣出
- [ ] 止盈曲線優化算法

**Cron 設置命令**：
```bash
# 添加到 OpenClaw cron（每 15 分鐘執行）
openclaw cron add \
  --name 'lana-takeprofit-check' \
  --every 900000 \
  --message '執行拉娜止盈檢查，掃描持有標的並生成報告' \
  --channel jarvis \
  --to 'hank:thread:main' \
  --announce
```

---

### 第三階段：數據訓練與聰明錢追蹤（待實現）🔜

**目標**：實現拉娜的 Hyperliquid 數據訓練方法

**需要開發**：
1. **鏈上數據抓取模組**
   - [ ] Hyperliquid API 對接
   - [ ] 全量交易數據存儲（SQLite/Parquet）
   - [ ] 實時數據流處理

2. **聰明錢地址分析**
   - [ ] 長期盈利地址識別算法
   - [ ] 錢包行為模式提取
   - [ ] 持倉週期統計
   - [ ] 進出場時機分析

3. **AI 訓練管道**
   - [ ] 特徵工程（技術指標 + 鏈上數據）
   - [ ] 模型訓練（Claude API / 本地 LLM）
   - [ ] 回測驗證
   - [ ] 持續學習機制

**技術棧建議**：
```
數據層：Hyperliquid API → Kafka → ClickHouse
分析層：Python + Pandas + Scikit-learn
AI 層：Claude API（雲端）+ Ollama（本地備份）
存儲層：SQLite（倉位）+ Redis（緩存）
```

---

### 第四階段：情緒標籤抓取（待實現）🔜

**目標**：實現幣安廣場情緒分析

**需要開發**：
1. **幣安廣場 Skills**
   - [ ] 帖子內容抓取
   - [ ] 標籤提取（漲/跌/中性）
   - [ ] 實盤交易組件解析
   - [ ] 話題熱度統計

2. **情緒指標計算**
   - [ ] 多源情緒融合（Twitter + 幣安廣場）
   - [ ] 情緒極性評分
   - [ ] 情緒 - 價格相關性分析

3. **策略集成**
   - [ ] 情緒因子加入評分系統
   - [ ] 極端情緒反向信號
   - [ ] 情緒動能跟隨

**API 需求**：
```python
# 假設的幣安廣場 API（需官方提供）
from binance_square import fetch_posts, fetch_sentiment

posts = fetch_posts(topic="crypto", limit=100)
sentiment = fetch_sentiment(symbol="BTCUSDT")
```

---

### 第五階段：完整系統集成（待實現）🔜

**架構圖**：
```
┌──────────────────────────────────────────────────────────┐
│                     人類交易員                            │
│  - 策略框架制定                                           │
│  - Prompt 迭代優化                                        │
│  - 異常情況干預                                           │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                   AI 思考層（Claude 4.6）                  │
│  - 市場情緒分析（幣安廣場 + Twitter）                      │
│  - 聰明錢行為模式識別                                      │
│  - 標的觀察名單生成                                        │
│  - 止盈概率判斷                                            │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                  AI 執行層（cyqnt_trd.blocks）             │
│  - 多因子評分進場                                          │
│  - 自動止損止盈                                            │
│  - Cron 定時任務（每 15 分鐘）                               │
│  - 倉位管理                                                │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                   數據層                                   │
│  - Binance API（K 線 + 合約數據）                          │
│  - Hyperliquid API（鏈上交易）                            │
│  - 幣安廣場 API（情緒標籤）                                │
│  - 本地數據庫（倉位 + 歷史）                               │
└──────────────────────────────────────────────────────────┘
```

---

## 📋 任務清單

### 立即可做（本週）
- [x] 閱讀 BLOCKS_API.md
- [x] 實現基礎策略框架
- [x] 實現 Cron 止盈任務骨架
- [ ] 設置 OpenClaw cron 定時任務
- [ ] 回測基礎策略（BTCUSDT 15m）
- [ ] 優化評分閾值參數

### 短期（1-2 週）
- [ ] 接入 Claude API 進行概率判斷
- [ ] 實現倉位數據持久化
- [ ] 對接交易所 API 執行交易
- [ ] 添加工程容錯（斷線重連）
- [ ] 實現止盈曲線優化

### 中期（1-2 月）
- [ ] Hyperliquid 數據抓取
- [ ] 聰明錢地址分析模組
- [ ] AI 訓練管道搭建
- [ ] 幣安廣場情緒抓取
- [ ] 多策略並行運行

### 長期（3-6 月）
- [ ] 完整系統集成測試
- [ ] 實盤小額運行（100U 起步）
- [ ] 持續迭代優化
- [ ] 擴展到更多標的

---

## ⚠️ 風險提示

### 技術風險
1. **API 限制**：Binance/Hyperliquid 可能有速率限制
2. **模型封號**：Claude API 封號風險（需本地備份方案）
3. **數據延遲**：實時數據流可能延遲影響決策

### 交易風險
1. **低勝率策略**：需要極強的心理素質
2. **高波動標的**：可能大幅回撤
3. **鏈上風險**：聰明錢可能是操縱行為

### 工程風險
1. **代碼 Bug**：需要 Codex 審計
2. **系統宕機**：需要監控告警
3. **資金安全**：私鑰管理至關重要

---

## 📚 參考資源

### 代碼文件
- `lana_style_momentum.py` - 基礎策略
- `lana_cron_takeprofit.py` - 止盈任務
- `BLOCKS_API.md` - blocks 庫文檔

### 外部資源
- [Binance API 文檔](https://binance-docs.github.io/apidocs/)
- [Hyperliquid API](https://hyperliquid.xyz/)
- [Claude API 文檔](https://docs.anthropic.com/)
- [OpenClaw Cron 工具](https://openclaw.dev/docs/cron)

---

## 🚀 下一步行動

1. **立即**：運行基礎策略回測
   ```bash
   python -m cyqnt_trd.standard_bot.entrypoints.mvp_backtest \
     --strategy lana_momentum_v1 \
     --strategy-module cyqnt_trd.strategies.lana_style_momentum \
     --symbol BTCUSDT --interval 15m --limit 500
   ```

2. **今天**：設置 Cron 定時任務
   ```bash
   openclaw cron add --name 'lana-takeprofit' --every 15m \
     --message '執行止盈檢查' --channel jarvis --to 'hank:thread:main'
   ```

3. **本週**：接入 Claude API 測試概率判斷

4. **下週**：開始小額實盤測試（100U）

---

*最後更新：2026-05-17*
*版本：v0.1*
