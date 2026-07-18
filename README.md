# 台灣天然氣供給量預測：LSTM vs ARIMA

> 使用長短期記憶模型（LSTM）預測月度天然氣供給量，並與傳統 ARIMA 模型進行系統性比較

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-TensorFlow-red)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 專案動機

台灣政府計畫於 2025 年前將天然氣發電佔比提升至 50%，精準的供給量預測對能源政策規劃、備料採購與電網調度至關重要。本專案以深度學習與統計模型進行比較實驗，探討不同方法在有限月度資料下的適用性與侷限。

---

## 資料

| 項目 | 說明 |
|------|------|
| 來源 | 經濟部能源署 |
| 時間範圍 | 2013年10月 – 2023年9月 |
| 資料筆數 | 120 筆（月度） |
| 訓練 / 測試切分 | 90% / 10%（依時序，未隨機打亂） |

---

## 方法

### Baseline：三層堆疊 LSTM（look_back=1）

- 僅以前一期供給量預測下一期
- 架構：3 × LSTM(50) → Dense(1)
- 優化器：Adam，損失函數：MSE

### 改進版：LSTM + 季節特徵（look_back=12）

針對 Baseline 的兩個核心弱點進行改進：

1. **擴展 look_back 至 12**：讓模型看到完整一年歷史，捕捉年週期性
2. **月份 sin/cos 特徵工程**：將 12 個月的週期性連續編碼，幫助模型辨識季節模式
3. **Dropout（0.2）+ EarlyStopping**：防止小樣本過擬合

```
輸入特徵：[供給量(t-12 ~ t-1), month_sin, month_cos]
架構：LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
```

### 比較基準：SARIMA(p=2, d=0, D=1, Q=1, s=12)

以傳統統計模型作為基準，驗證深度學習在此場景的相對優劣。定階方式：透過 SAS `proc arima` 對 nonseasonal / seasonal 差分與 ACF/PACF 交叉比對，確認 AR(2) 為起點；並以**季節差分（D=1, s=12）**取代一般差分，在季節性 lag=12 加入單一 MA 項，取代原本單純的非季節性 ARIMA(2,1,1)。採用 **滾動預測（Walk-Forward Validation）**，每一步以歷史資料重新擬合，確保與 LSTM 在相同測試區間下公平比較。

---

## 實驗結果

| 模型 | Test MAPE | 備註 |
|------|-----------|------|
| Baseline LSTM (look_back=1) | ~6–10% | 原始版本 |
| **改進 LSTM (look_back=12 + 季節特徵)** | **~5–8%** | 本專案核心貢獻 |
| SARIMA(2,0,0)x(0,1,1,12) | ~4–6% | 季節性統計基準，改用季節差分後優於原本的 ARIMA(2,1,1) |

### 分析洞察

**為何 Baseline LSTM 輸給 SARIMA？**

這是本實驗最具洞察力的發現：在僅有 120 筆月度資料的場景下，擁有數百倍參數的 LSTM 反而不如將季節性直接建模進去的 SARIMA。原因有二：

- **資料量不足**：LSTM 的非線性學習優勢需要大量資料支撐，120 筆反而讓模型過擬合
- **look_back 過短**：天然氣需求具有強烈的 12 月週期；look_back=1 等同於讓模型對季節性「視而不見」，而 SARIMA 用季節差分（D=1, s=12）直接處理了這個週期

**改進後的效果**：擴展視窗長度並加入月份週期特徵後，LSTM 的測試 MAPE 顯著下降，逐漸縮小與 SARIMA 的差距，驗證了特徵工程對小樣本深度學習的重要性；不過 SARIMA 在正確納入季節結構後仍維持領先，顯示對於這種強季節性、小樣本的時序資料，把週期性直接寫進模型（差分/季節項）比讓神經網路自己學出來更有效率。

---

## 延伸實驗：更多資料是否有幫助？

資料集後續補齊了 2004/01 起的歷史資料，並持續更新至 2026/05。這讓我們可以做兩件原本做不到的事：

1. 用真正「未來」的 32 個月真實資料（2023/10 ~ 2026/05，模型訓練時完全沒看過）來評估舊模型的**真實泛化能力**，而不只是原本 120 筆內的 90/10 切分。
2. 用 2004/01 ~ 2023/09（237 筆，多了 22 年歷史）重新訓練一個架構完全相同（look_back=12 + 季節特徵）的模型，驗證「資料量增加」是否真的改善預測。

評估方式分兩種：
- **One-step walk-forward**：每一步都用真實的前 12 個月資料預測下一月，測單步預測能力，不會累積誤差
- **Recursive**：只從訓練期最後一個月出發，之後全部用模型自己的預測值遞迴往後推，等同 API 實際部署時超出資料集範圍的行為

| 模型 | 訓練資料 | 內部 90/10 Test MAPE | 真實未來 One-step MAPE | 真實未來 Recursive MAPE |
|------|---------|----------------------|------------------------|--------------------------|
| 舊模型（部署中） | 120 筆（2013/10~2023/09） | ~6–8% | 7.95% | **11.03%** |
| 擴充版 | 237 筆（2004/01~2023/09） | 6.36% | **7.29%** | 13.10% |

**洞察：更多資料不是單純地「越多越好」。**

- 單步預測確實有小幅改善（7.95% → 7.29%），代表更長的歷史有助於模型學習季節模式。
- 但**遞迴多步預測反而變差**（11.03% → 13.10%）。推測原因是 2004~2013 年的供給量級距明顯偏低（年均約 82 萬，近年已達 250 萬以上），模型學到更長期的趨勢後，遞迴外推時容易低估近年持續走高的水準，使誤差累積更快。

**結論**：由於 API 實際部署情境是「遞迴預測」（超出資料集範圍時遞迴外推），目前 **仍以 120 筆訓練的舊模型（`model_lstm.keras`）作為正式部署版本**，擴充版模型（`model_lstm_v2.keras`）保留作為對照實驗，不取代生產模型。

---

## 專案結構

```
.
├── 天然氣供給_LSTM_portfolio.ipynb   # 主要分析 Notebook（含 EDA、建模、評估、模型匯出）
├── 天然氣供給月資料.csv               # 原始資料（經濟部能源署）
├── app.py                            # FastAPI 預測服務（使用 120 筆訓練的正式模型）
├── model_lstm.keras / scaler.pkl     # 正式部署模型（120 筆，look_back=12 + 季節特徵）
├── model_lstm_v2.keras / scaler_v2.pkl  # 延伸實驗模型（237 筆，同架構，僅供對照）
├── evaluate_holdout.py               # 用真實未來 32 個月資料評估舊模型的泛化能力
├── train_extended_model.py           # 訓練並評估擴充版模型（237 筆 vs 120 筆）
├── requirements.txt                  # 套件需求
└── README.md
```

---

## 環境需求

```bash
pip install -r requirements.txt
```

Notebook 以 Google Colab 開發，本機執行時將 `天然氣供給月資料.csv` 放置於同目錄，並移除 `drive.mount()` 相關程式碼即可。隨機種子已固定為 `SEED=42`，結果可重現。

---

## 預測 API

執行完 Notebook 的 **Section 8（模型匯出）** 後，會產生 `model_lstm.keras` 與 `scaler.pkl`，即可啟動 API：

```bash
uvicorn app:app --reload
```

API 啟動時會自動讀取 `天然氣供給月資料.csv` 建立月份查詢表，呼叫端不需再手動傳入歷史資料；只要指定目標月份，超出資料集範圍的月份會自動遞迴預測中間各月。

### 端點

`POST /predict`：單月預測

```json
{
  "target_month": "2024-01"
}
```

回傳：

```json
{
  "target_month": "2024-01",
  "predicted_supply": 1123456.0,
  "unit": "千立方公尺",
  "data_source": "recursive_prediction",
  "actual_supply": null,
  "error_pct": null
}
```

`data_source` 為 `historical_data` 表示目標月份在資料集範圍內（可用來驗證模型），此時 `actual_supply` 會帶出真實值、`error_pct` 是該筆的絕對百分比誤差（APE，即 `|實際值−預測值|/實際值×100%`）；`recursive_prediction` 表示超出範圍、由遞迴預測產生，此時沒有真實值可比對，兩欄位皆為 `null`。

`POST /predict/multi`：多月連續預測

```json
{
  "start_month": "2024-01",
  "steps": 6
}
```

回傳 `predictions` 陣列（每筆格式同 `/predict` 回傳）與 `batch_mape`——僅根據這批預測中落在資料集內（`historical_data`）的月份計算平均 APE（即 MAPE），若整批都是 `recursive_prediction` 則為 `null`。`steps` 上限為 36，建議 ≤ 6 以避免遞迴誤差累積過大。

互動式文件（Swagger UI）請至 `http://localhost:8000/docs`

---

## 技術棧

- **深度學習**：Keras / TensorFlow（LSTM、Dropout、EarlyStopping）
- **統計模型**：statsmodels（SARIMAX、ACF/PACF），定階過程另以 SAS `proc arima` 交叉驗證
- **資料處理**：pandas、numpy、scikit-learn（MinMaxScaler）
- **視覺化**：matplotlib

---

## 未來改進方向

- 加入外部特徵（氣溫、電力需求、國際天然氣價格）
- ~~使用完整 237 筆資料，發揮 LSTM 的非線性學習潛力~~ → 已驗證（見〈延伸實驗〉），單步預測有改善但遞迴預測誤差增加，非單純的資料量問題
- 針對遞迴預測誤差累積：嘗試在訓練時加入 teacher forcing 之外的多步損失（如 scheduled sampling），或限制輸入特徵的資料期間以避免長期趨勢干擾外推
- 嘗試 Temporal Fusion Transformer（TFT）等近代時序模型

---

## 作者

張若芸｜應用統計學研究所  
指導課程：時間序列分析期末專題
