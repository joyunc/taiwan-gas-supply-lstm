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

### 比較基準：ARIMA(2,1,1)

以傳統統計模型作為基準，驗證深度學習在此場景的相對優劣。採用 **滾動預測（Walk-Forward Validation）**，每一步以歷史資料重新擬合，確保與 LSTM 在相同測試區間下公平比較。

---

## 實驗結果

| 模型 | Test MAPE | 備註 |
|------|-----------|------|
| Baseline LSTM (look_back=1) | ~8–10% | 原始版本 |
| **改進 LSTM (look_back=12 + 季節特徵)** | **~6–8%** | 本專案核心貢獻 |
| ARIMA | ~6% | 線性統計基準 |

### 分析洞察

**為何 Baseline LSTM 輸給 ARIMA？**

這是本實驗最具洞察力的發現：在僅有 120 筆月度資料的場景下，擁有數百倍參數的 LSTM 反而不如 ARIMA。原因有二：

- **資料量不足**：LSTM 的非線性學習優勢需要大量資料支撐，120 筆反而讓模型過擬合
- **look_back 過短**：天然氣需求具有強烈的 12 月週期；look_back=1 等同於讓模型對季節性「視而不見」

**改進後的效果**：擴展視窗長度並加入月份週期特徵後，LSTM 的測試 MAPE 顯著下降，逐漸縮小與 ARIMA 的差距，驗證了特徵工程對小樣本深度學習的重要性。

---

## 專案結構

```
.
├── 天然氣供給_LSTM_portfolio.ipynb   # 主要分析 Notebook（含 EDA、建模、評估、模型匯出）
├── 天然氣供給月資料.csv               # 原始資料（經濟部能源署）
├── app.py                            # FastAPI 預測服務
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
  "data_source": "recursive_prediction"
}
```

`data_source` 為 `historical_data` 表示目標月份在資料集範圍內（可用來驗證模型），`recursive_prediction` 表示超出範圍、由遞迴預測產生。

`POST /predict/multi`：多月連續預測

```json
{
  "start_month": "2024-01",
  "steps": 6
}
```

回傳 `predictions` 陣列（每筆格式同 `/predict` 回傳），`steps` 上限為 36，建議 ≤ 6 以避免遞迴誤差累積過大。

互動式文件（Swagger UI）請至 `http://localhost:8000/docs`

---

## 技術棧

- **深度學習**：Keras / TensorFlow（LSTM、Dropout、EarlyStopping）
- **統計模型**：statsmodels（ARIMA、ACF/PACF）
- **資料處理**：pandas、numpy、scikit-learn（MinMaxScaler）
- **視覺化**：matplotlib

---

## 未來改進方向

- 加入外部特徵（氣溫、電力需求、國際天然氣價格）
- 使用完整 237 筆資料，發揮 LSTM 的非線性學習潛力
- 嘗試 Temporal Fusion Transformer（TFT）等近代時序模型

---

## 作者

張若芸｜應用統計學研究所  
指導課程：時間序列分析期末專題
