# Demo:真實用戶對話 → YAML → 跑出訊號(選幣)

**日期**:2026-08-01 · **spec**:`example_from_user_chat.yaml` · **結論**:**跑通了,但輸出不符合用戶要求 —— 差在 block 詞彙,不在管線。**

> **2026-08-02 後記:擋住這次 demo 的那個詞彙缺口已經補上了**(§6 第 1 項)。
> `universe.augment_with_contract_meta` + `filter_crypto_only` 兩步加進同一份 spec,
> 同一個凍結截面的籃子就從「5/5 TradFi」變成「5/5 COIN」:
> `HYPEUSDT · BANKUSDT · COTIUSDT · MMTUSDT · UNIUSDT`。
> **本文以下段落刻意保持原狀** —— 它記錄的是 08-01 那次實跑,而那次的籃子仍然是
> `example_from_user_chat.yaml` 現在跑出來的東西(spec 本身沒改,見 §6 第 1 項的理由)。
> 修好後的籃子釘在 `tests/standard_bot/test_universe_contract_meta.py::CRYPTO_ONLY_BASKET`。

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
| 排除 USDC 計價對 | `universe.filter_quote_suffix{suffix: USDC, exclude: true}` | ✅ 完全對應 |
| 找做空候選 | `short_when: value_below(priceChangePercent, -2.0)` | ✅ 完全對應 |
| Supertrend 三時框偏空 | `universe.top_losers{n: 30}` | 🟡 粗代理 |
| **散戶多空比偏多 > 60%** | — | ❌ 無法表達 |
| **exclude tradfi** | — | ❌ 無法表達(08-01 當時;08-02 已補 `universe.filter_crypto_only`,見 §6-1) |

用戶的 4 條需求:成交額那條完整;「排除」那條的三個小項有兩個完整(主流幣逐一列名、
USDC 計價對用 `filter_quote_suffix` 整批排除)、TradFi 那一項完全轉不出來;
Supertrend 只能用跌幅代理;多空比完全轉不出來。

## 3. 實跑

```
$ python -m cyqnt_trd.standard_bot.yaml_pipeline validate docs/strategy_yaml_spec/example_from_user_chat.yaml
OK: spec 'user_short_candidate_screen' is valid and dry-ran successfully on synthetic data.

$ python -m cyqnt_trd.standard_bot.yaml_pipeline run docs/strategy_yaml_spec/example_from_user_chat.yaml \
      --input-json tests/standard_bot/fixtures/universe_cross_section.json
```

`--input-json` 走的是 bundle replay 路徑,stdout 印的是**完整的 `cyqnt.signal-batch/v1` JSON**
(不是摘要行)。裡面那一個 signal 的重點欄位:

| 欄位 | 值 |
|---|---|
| `schema` / `kind` | `cyqnt.signal/v2` / `selection` |
| `universe_size` / `data_quality` | 727 / `good` |
| `candidates` | `SNDKUSDT` 3.785e+09 · `SOXLUSDT` 1.687e+09 · `MUUSDT` 1.227e+09 · `SKHYUSDT` 7.854e+08 · `KORUUSDT` 7.252e+08,rank 1–5,全部 `short` |
| `candidates[0].reason` | `quoteVolume=3.785e+09, rank 1 of 5, order=desc (highest first)` |

> 這裡餵的是**凍結的**真實截面(727 檔,`tests/standard_bot/fixtures/universe_cross_section.json`),
> 所以上面每個數字都可以逐位元重現,並由 `tests/standard_bot/test_selection_fixture_replay.py`
> 的 golden basket 釘住。不加 `--input-json` 會改成即時抓 Binance —— 一樣跑得動,但籃子每小時都不同,
> 文件裡就沒有可對照的數字了(2026-08-01 對線上宇宙實跑得到的是同樣這五檔、同樣順序,
> 只是當天的成交額與跌幅數字不同)。

**逐項驗證管線邏輯正確**:

| 檢查 | 結果 |
|---|---|
| 5 檔是否真的弱勢 | -11.76 / -14.31 / -11.43 / -9.94 / -14.50 %,全部 < -2% ✅ |
| 是否真在跌幅前 30 | 前 30 名門檻 -8.13%,全部通過 ✅ |
| 是否依成交額由大到小 | 3.79e9 > 1.69e9 > 1.23e9 > 7.85e8 > 7.25e8 ✅ |

## 4. 但輸出是錯的 —— 而且錯得很有代表性

SNDK(SanDisk)、SOXL(半導體 ETF)、MU(美光)、SKHY(SK 海力士)、KORU(韓股 ETF) —— **5 檔全是 TradFi 永續,正是用戶第一條就講明要排除的**。

為什麼會被整碗端走:這個截面的跌幅前 30 名裡有 **21 檔**不是加密幣(`underlyingType != COIN`),只有 9 檔是;而且成交額差一到兩個數量級(前 5 名全是 `EQUITY`,7.3e8–3.8e9,最大的 `COIN` 只有 3.4e8)。用成交額排序時它們直接獨佔全部 5 個名額。

**這裡刻意不列出「正確的籃子」。** 本文原本寫著「若能排除 TradFi,正確的籃子是 SNXXUSDT / MMTUSDT / MUUUSDT / AAVEUSDT / BEUSDT」——那一行是錯的,而且錯得跟這次 demo 是同一個病:當時是拿**手維護的代號清單**判斷誰是 TradFi,沒在清單裡的就當成加密幣。實際查 `exchangeInfo`,那五檔裡有**三檔本身就是 TradFi**:

| 代號 | `contractType` | `underlyingType` | `underlyingSubType` |
|---|---|---|---|
| `SNXXUSDT` | `TRADIFI_PERPETUAL` | `EQUITY` | `['TradFi']` |
| `MUUUSDT` | `TRADIFI_PERPETUAL` | `EQUITY` | `['TradFi']` |
| `BEUSDT` | `TRADIFI_PERPETUAL` | `EQUITY` | `['TradFi']` |
| `MMTUSDT` | `PERPETUAL` | `COIN` | `['DeFi']` |
| `AAVEUSDT` | `PERPETUAL` | `COIN` | `['DeFi']` |

正確做法是問交易所,不是維護清單:`underlyingType == "COIN"`。而且籃子本身也不該寫死在文件裡 —— 行情每天在走,寫死的答案明天就會再錯一次;要釘住某一刻的籃子,用 §7 那個凍結截面。

這就是這次 demo 最值得帶走的一句話:**管線是通的,YAML 語法也夠用,卡住的是 universe 層的 block 詞彙 —— 而且一個詞彙缺口就足以讓整個籃子失去意義,連事後人工補救都會補錯。**

## 5. 過程中發現的兩個 bug(皆已修)

**(a) `universe.augment_with_funding` 從 YAML 完全無法使用** — **已修**

原死結是不寫 `with:` 會被守衛擋下,照寫 `with: [funding]` 又因函式只收一個參數而失敗。現在 block 接受外部 funding frame,selection plugin 會把 `DataSnapshot.frames` 傳給 YAML interpreter；live 以全市場 `funding_snapshot` 收集後 alias 成 bundle key `funding`。validator、bundle E2E、PIT 與反事實排名測試都已釘住,且不會把單一標的歷史 funding 當成全市場截面。

**(b) validate 的合成 universe frame 缺 `priceChangePercent`** — **已修**

`_synthetic_universe()` 沒有這個欄位,導致 `top_gainers` / `top_losers` / `filter_change_pct` **三個 block 一律無法通過 validate**,錯誤訊息 `DataFrame missing 'priceChangePercent' column` 看起來像是用戶 spec 寫錯,其實是驗證器的假資料不完整 —— 這正好違反 `vocabulary.py` 自己立的原則(「dry-run frame 要跟真實資料同欄位」)。
→ 已補上 -11%~+11% 正負交錯的 `priceChangePercent`(`spec.py`,+11 行)。回歸:原有 4 個範例仍全部 validate 通過;`tests/standard_bot/test_yaml_*` 88 passed,唯一 failure 是缺 `jsonschema` 套件的環境問題(已用 `git stash` 確認改動前就失敗)。

## 6. 建議補的 block(照效益排序)

以 5–7 月 **13,983 筆選幣對話**估算需求量:

1. ~~**`universe.augment_with_contract_meta` + `filter_underlying_type` / `filter_sub_type`**~~ — **2026-08-02 已完成**

   四個 block 落地:`augment_with_contract_meta`(`with: [contract_meta]`)、
   `filter_underlying_type(include?/exclude?)`、`filter_sub_type(include?/exclude?)`、
   `filter_crypto_only()`。資料節點 `data.contract_meta`(`FORWARD_ONLY`,一次
   `GET /fapi/v1/exchangeInfo`,**不做任何過濾**,連 `status` / `quoteAsset` 都當欄位交出去)。
   凍結截面已用 `scripts/freeze_selection_fixture.py --add-frame contract_meta` 補上這一個
   frame,其餘 frame 與 `decision_time` 逐位元不變 —— 所以本文 §3 的籃子沒有被「重新捕捉」
   洗掉,是可以跟修好後的籃子直接對照的。

   **原本的建議內容保留如下**(那是決定 block 語意的依據,不是待辦):

   一次 `GET /fapi/v1/exchangeInfo` 就能拿到三個現成欄位(以下為 2026-08-01、727 檔 `status=TRADING` 的實測分佈):
   - `contractType`:`PERPETUAL`(573)| `TRADIFI_PERPETUAL`(150)| `CURRENT_QUARTER`(2)| `NEXT_QUARTER`(2)
   - `underlyingType`:`COIN`(575)| `EQUITY`(131)| `COMMODITY`(8)| `HK_EQUITY`(6)| `KR_EQUITY`(3)| `INDEX`(2)| `PREMARKET`(2)
   - `underlyingSubType`:`TradFi`(150)| `DeFi`(117)| `Alpha`(71)| `Infrastructure`(59)| `AI`(57)| `Layer-1`(54)| `Meme`(47)| `USDC`(36)…

   ⚠️ 三種「排除 TradFi」的寫法**不等價**,差 2 檔:`contractType != TRADIFI_PERPETUAL` → 577、`"TradFi" not in underlyingSubType` → 577、`underlyingType == "COIN"` → 575。差集是 `{ALLUSDT, BTCDOMUSDT}`,兩者都是 `underlyingType=INDEX` 的合成指數(一籃子加密資產,不是某個幣)。前兩種黑名單寫法會讓合成指數漏進「加密幣」那一邊,所以 block 要提供的是白名單語意 `underlyingType == "COIN"`。

   一個 block 同時解掉四類高頻條件:板塊/賽道 478 筆(3.4%)· 市值大小 707 筆(5.1%)· Alpha 幣 351 筆(2.5%)· 排除/指定 TradFi 或美股 414 筆(3.0%)。**這次 demo 就是被它擋住的。**

   最後採用的是白名單 `filter_crypto_only`(= `underlying_type == "COIN"`),理由就是上面那個
   ⚠️:三種寫法會給三個答案,而 LLM 會隨機挑一種、挑錯了沒人看得出來。把意圖命名一次,
   選擇就消失了。另外兩個 block 留給「答案真的是比較鬆的那個集合」的情況
   (例如 `filter_underlying_type(include: [COIN, INDEX])` = 連加密指數一起留)。
   實作上有兩個會咬人的細節,都寫在 block docstring 裡:`underlyingSubType` 是**陣列**,
   多標籤幣(如 `FOLKSUSDT` = `['Alpha','DeFi']`)不壓成純量會在組候選 features 時
   raise `truth value of an array ... is ambiguous`,而且只有多標籤幣會炸;
   還有 join 覆蓋率不到 95% 直接 raise,免得 stale/錯 market_type 的 registry 讓
   整張表變 NaN、篩選回空籃子卻沒有任何診斷。

2. **`universe.augment_with_indicator`** — 對每個候選抓 K 線、算指標、join 回截面 frame。
   「先掃全市場、再對每個候選跑技術指標」是選幣對話裡最常見的形狀,現在完全做不到(只能用 24h 漲跌幅代理)。

3. **`universe.augment_with_long_short_ratio` / `augment_with_open_interest`** — 多空比與持倉量,用戶語言裡高頻,Binance 有現成端點。

上面第 2、3 項還沒做(第 1 項見上,已完成)。以下兩項是這份 demo 當初列的待辦,**現在也已經做完了**,留在這裡是為了讓照舊版文件繞路的人知道不必再繞:

4. ~~**`selection.order: asc|desc`**~~ — **已完成**。`order: asc` 可用,預設仍是 `desc`;「跌幅最大」寫 `score: priceChangePercent` + `order: asc` 即可,不必再先 `top_losers` 收窄再換一個天生 descending 的欄位。`min_score` / `max_score` 是絕對邊界、不隨 `order` 翻轉(細節見 `strategy.schema.yaml` 的 `selection.order`)。

5. ~~**匯出 `universe.filter_quote_suffix`**~~ — **已完成**。module-level 函式已在 `blocks/universe.py` 的 `__all__` 裡,YAML 可直接用:`params: { suffix: USDC, exclude: true }`(`suffix` 也收 `[USDT, USDC]` 這種清單)。§2 那條「排除 USDC 計價對」就是用它,不再逐一列名 —— 這個截面有 38 個 USDC 對,舊的手寫名單只列了 4 個。

## 7. 重現

```bash
# 可重現的那條:凍結截面,不觸網,籃子由 test_selection_fixture_replay.py 釘住
python -m cyqnt_trd.standard_bot.yaml_pipeline validate docs/strategy_yaml_spec/example_from_user_chat.yaml
python -m cyqnt_trd.standard_bot.yaml_pipeline run      docs/strategy_yaml_spec/example_from_user_chat.yaml \
    --input-json tests/standard_bot/fixtures/universe_cross_section.json \
    --output-json /tmp/sel_out.json

# 對今天的市場跑(會實際打 Binance REST 抓 24h ticker,籃子與 §3 不同是正常的)
python -m cyqnt_trd.standard_bot.yaml_pipeline run      docs/strategy_yaml_spec/example_from_user_chat.yaml \
    --output-json /tmp/sel_out_live.json
```

> 需要 python3.11 + pandas/numpy/pyyaml/requests。
> 選幣是單一時點決策,CLI 會明講「這不是回測」——本 repo 沒有截面回測引擎。
