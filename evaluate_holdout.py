"""
評估現有模型（以 2013/10~2023/09 共 120 筆訓練）在
2023/10~2026/05 新增 32 筆真實資料上的表現。

兩種評估方式：
1. one-step walk-forward：每一步都用「真實」的前 12 個月資料預測下一個月
   （最佳情境，不會累積誤差，測試模型的單步泛化能力）
2. recursive：只用訓練期資料（到 2023/09）出發，之後全部用模型自己的
   預測值遞迴往後推（實際部署情境，測試誤差累積程度）
"""
import math
import pickle

import keras
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime

LOOK_BACK = 12
TRAIN_END = datetime(2023, 9, 1)  # 舊模型訓練資料的最後一個月

model = keras.models.load_model("model_lstm.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("天然氣供給月資料.csv", usecols=[2], skiprows=117)
df.columns = ["supply"]
df.index = pd.date_range(start="2013-10", periods=len(df), freq="MS")
real_data = {dt.strftime("%Y-%m"): float(v) for dt, v in zip(df.index, df["supply"])}


def step(scaled_history, target_month_num):
    features = []
    for i in range(LOOK_BACK):
        m = ((target_month_num - LOOK_BACK + i - 1) % 12 + 12) % 12 + 1
        features.append([
            scaled_history[i],
            math.sin(2 * math.pi * (m - 1) / 12),
            math.cos(2 * math.pi * (m - 1) / 12),
        ])
    X = np.array(features).reshape(1, LOOK_BACK, 3)
    return float(model.predict(X, verbose=0)[0][0])


def predict_next(history_values, target_dt):
    scaled = scaler.transform(np.array(history_values).reshape(-1, 1).astype("float32")).flatten().tolist()
    pred_scaled = step(scaled, target_dt.month)
    return float(scaler.inverse_transform([[pred_scaled]])[0][0])


holdout_start = TRAIN_END + relativedelta(months=1)
holdout_end = max(datetime.strptime(m, "%Y-%m") for m in real_data)

months = []
current = holdout_start
while current <= holdout_end:
    months.append(current)
    current += relativedelta(months=1)

# ---------- 1. one-step walk-forward（每步都用真實歷史值） ----------
rows = []
for dt in months:
    key = dt.strftime("%Y-%m")
    hist_keys = [(dt - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m") for i in range(LOOK_BACK)]
    hist_values = [real_data[k] for k in hist_keys]
    pred = predict_next(hist_values, dt)
    actual = real_data[key]
    ape = abs(pred - actual) / actual * 100
    rows.append((key, actual, pred, ape))

df_walk = pd.DataFrame(rows, columns=["month", "actual", "pred_walkforward", "ape_walkforward"])

# ---------- 2. recursive（只從訓練期出發，之後全用模型預測值遞迴） ----------
supply_recursive = {k: v for k, v in real_data.items() if datetime.strptime(k, "%Y-%m") <= TRAIN_END}
for dt in months:
    key = dt.strftime("%Y-%m")
    hist_keys = [(dt - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m") for i in range(LOOK_BACK)]
    hist_values = [supply_recursive[k] for k in hist_keys]
    pred = predict_next(hist_values, dt)
    supply_recursive[key] = pred

df_walk["pred_recursive"] = [supply_recursive[m] for m in df_walk["month"]]
df_walk["ape_recursive"] = (df_walk["pred_recursive"] - df_walk["actual"]).abs() / df_walk["actual"] * 100

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
print(df_walk.to_string(index=False))
print()
print(f"評估區間：{holdout_start:%Y-%m} ~ {holdout_end:%Y-%m}（{len(months)} 個月，模型訓練時完全未見過）")
print(f"One-step walk-forward MAPE：{df_walk['ape_walkforward'].mean():.2f}%")
print(f"Recursive（實際部署情境）MAPE：{df_walk['ape_recursive'].mean():.2f}%")
