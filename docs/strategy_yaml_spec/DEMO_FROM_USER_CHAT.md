# Demo:真實用戶對話 → YAML → 跑出訊號(選幣)

**日期**:2026-08-01 · **spec**:`example_from_user_chat.yaml` · **結論**:**跑通了,但輸出不符合用戶要求 —— 差在 block 詞彙,不在管線。**

## 1. 用的是哪段對話

2026-05-17 與 05-18 兩天重複出現的同一則選幣需求(取自內部對話需求分析,`primary_intent=COIN_SELECTION`)。
**以下為條件摘要,非逐字原文** —— 生產對話內容不進本 repo:

> 想找**做空候選**,掃 Binance 合約:
> - 排除 BTC / ETH / SOL / XRP、TradFi 標的、USDC 計價對
> - 24 小時成交額 > 200 萬美元
> - 散戶多空比偏多 > 60%
> - Supertrend(10,3) 在 H4 / H1 / M15 三個時框於近 2 小時內同時偏空

選它的原因:條件已經被列成清單,是「自然語言 → YAML」最好的壓力測試 —— 條件夠具體,轉不出來的地方會直接現形。

## 2. 轉換結果

| 用戶條件 | YAML | 結果 |
|---|---|---|
| 24h 成交額 > $2m | `universe.filter_quote_volume{min_quote_volume: 2e6}` | ✅ 完全對應 |
| 排除 BTC/ETH/SOL/XRP | `universe.exclude_symbols` | ✅ 完全對應 |
| 找做空候選 | `short_when: value_below(priceChangePercent, -2.0)` | ✅ 完全對應 |
| Supertrend 三時框偏空 | `universe.top_losers{n: 30}` | 🟡 粗代理 |
| 排除 USDC 計價對 | 逐一列名 + `dedupe_by: base_asset` | 🟡 列不完 |
| **散戶多空比偏多 > 60%** | — | ❌ 無法表達 |
| **exclude tradfi** | — | ❌ 無法表達 |

4 條裡 2 條完整、1 條代理、1 條完全轉不出來。

## 3. 實跑

```
$ python -m cyqnt_trd.standard_bot.yaml_pipeline validate docs/strategy_yaml_spec/example_from_user_chat.yaml
OK: spec 'user_short_candidate_screen' is valid and dry-ran successfully on synthetic data.

$ python -m cyqnt_trd.standard_bot.yaml_pipeline run docs/strategy_yaml_spec/example_from_user_chat.yaml
[yaml_pipeline] selection strategy=user_short_candidate_screen market=futures universe=727
  output=cyqnt.signal/v2 kind=selection data_quality=good
  #1  SNDKUSDT   score=6.06e+09  short
  #2  SOXLUSDT   score=2.50e+09  short
  #3  MUUSDT     score=1.89e+09  short
  #4  SKHYUSDT   score=1.36e+09  short
  #5  KORUUSDT   score=1.28e+09  short
```

727 檔真實宇宙、輸出 `cyqnt.signal/v2` selection envelope。**逐項驗證管線邏輯正確**:

| 檢查 | 結果 |
|---|---|
| 5 檔是否真的弱勢 | -9.58 / -7.74 / -8.84 / -7.23 / -9.50 %,全部 < -2% ✅ |
| 是否真在跌幅前 30 | 前 30 名門檻 -7.09%,全部通過 ✅ |
| 是否依成交額由大到小 | 6.06e9 > 2.50e9 > 1.89e9 > 1.36e9 > 1.28e9 ✅ |

## 4. 但輸出是錯的 —— 而且錯得很有代表性

SNDK(SanDisk)、SOXL(半導體 ETF)、MU(美光)、SKHY(SK 海力士)、KORU(韓股 ETF) —— **5 檔全是 TradFi 永續,正是用戶第一條就講明要排除的**。

為什麼會被整碗端走:跌幅前 30 名裡 TradFi 只有 5 檔,但成交額比加密幣大 1–2 個數量級(1.3e9–6.1e9 vs 5e7–9.4e8),用成交額排序時直接獨佔全部 5 個名額。

**若能排除 TradFi,正確的籃子是**:SNXXUSDT / MMTUSDT / MUUUSDT / AAVEUSDT / BEUSDT。

這就是這次 demo 最值得帶走的一句話:**管線是通的,YAML 語法也夠用,卡住的是 universe 層的 block 詞彙 —— 而且一個詞彙缺口就足以讓整個籃子失去意義。**

## 5. 過程中發現的兩個 bug(皆已修)

**(a) `universe.augment_with_funding` 從 YAML 完全無法使用** — **已修**

原死結是不寫 `with:` 會被守衛擋下,照寫 `with: [funding]` 又因函式只收一個參數而失敗。現在 block 接受外部 funding frame,selection plugin 會把 `DataSnapshot.frames` 傳給 YAML interpreter；live 以全市場 `funding_snapshot` 收集後 alias 成 bundle key `funding`。validator、bundle E2E、PIT 與反事實排名測試都已釘住,且不會把單一標的歷史 funding 當成全市場截面。

**(b) validate 的合成 universe frame 缺 `priceChangePercent`** — **已修**

`_synthetic_universe()` 沒有這個欄位,導致 `top_gainers` / `top_losers` / `filter_change_pct` **三個 block 一律無法通過 validate**,錯誤訊息 `DataFrame missing 'priceChangePercent' column` 看起來像是用戶 spec 寫錯,其實是驗證器的假資料不完整 —— 這正好違反 `vocabulary.py` 自己立的原則(「dry-run frame 要跟真實資料同欄位」)。
→ 已補上 -11%~+11% 正負交錯的 `priceChangePercent`(`spec.py`,+11 行)。回歸:原有 4 個範例仍全部 validate 通過;`tests/standard_bot/test_yaml_*` 88 passed,唯一 failure 是缺 `jsonschema` 套件的環境問題(已用 `git stash` 確認改動前就失敗)。

## 6. 建議補的 block(照效益排序)

以 5–7 月 **13,983 筆選幣對話**估算需求量:

1. **`universe.augment_with_contract_meta` + `filter_underlying_type` / `filter_sub_type`**
   一次 `GET /fapi/v1/exchangeInfo` 就能拿到三個現成欄位:
   - `contractType`:`PERPETUAL` | `TRADIFI_PERPETUAL`
   - `underlyingType`:`COIN`(698)| `EQUITY`(131)| `COMMODITY`(8)| `HK_EQUITY`(6)| `KR_EQUITY`(3)| `INDEX`(3)| `PREMARKET`(2)
   - `underlyingSubType`:`DeFi`(153)| `TradFi`(147)| `Alpha`(76)| `Infrastructure`(70)| `AI`(68)| `Layer-1`(64)| `Meme`(55)…

   一個 block 同時解掉四類高頻條件:板塊/賽道 478 筆(3.4%)· 市值大小 707 筆(5.1%)· Alpha 幣 351 筆(2.5%)· 排除/指定 TradFi 或美股 414 筆(3.0%)。**這次 demo 就是被它擋住的。**

2. **`universe.augment_with_indicator`** — 對每個候選抓 K 線、算指標、join 回截面 frame。
   「先掃全市場、再對每個候選跑技術指標」是選幣對話裡最常見的形狀,現在完全做不到(只能用 24h 漲跌幅代理)。

3. **`universe.augment_with_long_short_ratio` / `augment_with_open_interest`** — 多空比與持倉量,用戶語言裡高頻,Binance 有現成端點。

4. **`selection.order: asc|desc`** — 目前 score 只能 descending,做空篩選要繞路。

5. **匯出 `universe.filter_quote_suffix`** — `UniverseFilter` 類別上已有,只是沒有 module-level 版本,YAML 用不到。

## 7. 重現

```bash
python -m cyqnt_trd.standard_bot.yaml_pipeline validate docs/strategy_yaml_spec/example_from_user_chat.yaml
python -m cyqnt_trd.standard_bot.yaml_pipeline run      docs/strategy_yaml_spec/example_from_user_chat.yaml \
    --output-json /tmp/sel_out.json
```

> 需要 python3.11 + pandas/numpy/pyyaml/requests。`run` 會實際打 Binance REST 抓 24h ticker。
> 選幣是單一時點決策,CLI 會明講「這不是回測」——本 repo 沒有截面回測引擎。
