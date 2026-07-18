"""
訓練「擴充版」LSTM：資料範圍改為 2004/01 ~ 2023/09（237 筆），
架構與原本改進模型（look_back=12 + 月份 sin/cos + Dropout + EarlyStopping）完全相同，
唯一變數是訓練資料量（237 筆 vs 舊模型的 120 筆）。

評估分兩層，皆與舊模型（model_lstm.keras, 120 筆訓練）對照：
1. 內部 90/10 切分的 Test MAPE：與 README 記載的舊模型 Test MAPE（~6-8%）同方法比較
2. 真實未來 32 個月（2023/10~2026/05）holdout：與 evaluate_holdout.py 中舊模型的結果
   （one-step 7.95% / recursive 11.03%）比較，兩個模型用同一組從未參與訓練的真實資料評估
"""
import math
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOOK_BACK = 12
TRAIN_END = datetime(2023, 9, 1)

# ---------- 資料 ----------
df_all = pd.read_csv("天然氣供給月資料.csv", usecols=[2], skiprows=1, header=None)
df_all.columns = ["supply"]
df_all.index = pd.date_range(start="2004-01", periods=len(df_all), freq="MS")

df = df_all.loc[:TRAIN_END].copy()  # 訓練池：2004-01 ~ 2023-09
print(f"擴充訓練資料：{len(df)} 筆（{df.index[0]:%Y-%m} ~ {df.index[-1]:%Y-%m}）")

values = df["supply"].values.reshape(-1, 1).astype("float32")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

months = np.array([((df.index[i].month - 1) / 12) for i in range(len(df))])
month_sin = np.sin(2 * np.pi * months)
month_cos = np.cos(2 * np.pi * months)
features = np.column_stack([scaled.flatten(), month_sin, month_cos])


def create_dataset_multifeature(data, look_back=12):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


train_size = int(len(scaled) * 0.9)
allX, allY = create_dataset_multifeature(features, LOOK_BACK)
split_idx = train_size - LOOK_BACK - 1
trainX, trainY = allX[:split_idx], allY[:split_idx]
testX, testY = allX[split_idx:], allY[split_idx:]
n_features = trainX.shape[2]

print(f"訓練集 shape：{trainX.shape}  測試集 shape：{testX.shape}")

# ---------- 模型（與改進版架構相同） ----------
model = Sequential([
    LSTM(64, activation="relu", return_sequences=True, input_shape=(LOOK_BACK, n_features)),
    Dropout(0.2),
    LSTM(32, activation="relu"),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
model.fit(trainX, trainY, epochs=150, batch_size=4, validation_split=0.1, callbacks=[early_stop], verbose=0)


def evaluate(model, trainX, trainY, testX, testY, scaler, label):
    train_pred = scaler.inverse_transform(model.predict(trainX, verbose=0))
    test_pred = scaler.inverse_transform(model.predict(testX, verbose=0))
    train_true = scaler.inverse_transform(trainY.reshape(-1, 1))
    test_true = scaler.inverse_transform(testY.reshape(-1, 1))
    train_rmse = math.sqrt(mean_squared_error(train_true, train_pred))
    test_rmse = math.sqrt(mean_squared_error(test_true, test_pred))
    train_mape = np.mean(np.abs(train_true - train_pred) / train_true) * 100
    test_mape = np.mean(np.abs(test_true - test_pred) / test_true) * 100
    print(f"[{label}] Train RMSE {train_rmse:,.0f} | Train MAPE {train_mape:.2f}% || "
          f"Test RMSE {test_rmse:,.0f} | Test MAPE {test_mape:.2f}%")


evaluate(model, trainX, trainY, testX, testY, scaler, "擴充版 LSTM（2004起，237筆，內部90/10切分）")

model.save("model_lstm_v2.keras")
with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---------- 真實未來 32 個月 holdout（與舊模型同一組資料比較） ----------
real_data_full = {dt.strftime("%Y-%m"): float(v) for dt, v in zip(df_all.index, df_all["supply"])}


def step(scaled_history, target_month_num):
    feats = []
    for i in range(LOOK_BACK):
        m = ((target_month_num - LOOK_BACK + i - 1) % 12 + 12) % 12 + 1
        feats.append([
            scaled_history[i],
            math.sin(2 * math.pi * (m - 1) / 12),
            math.cos(2 * math.pi * (m - 1) / 12),
        ])
    X = np.array(feats).reshape(1, LOOK_BACK, 3)
    return float(model.predict(X, verbose=0)[0][0])


def predict_next(history_values, target_dt):
    s = scaler.transform(np.array(history_values).reshape(-1, 1).astype("float32")).flatten().tolist()
    ps = step(s, target_dt.month)
    return float(scaler.inverse_transform([[ps]])[0][0])


holdout_start = TRAIN_END + relativedelta(months=1)
holdout_end = max(datetime.strptime(m, "%Y-%m") for m in real_data_full)
months_list = []
cur = holdout_start
while cur <= holdout_end:
    months_list.append(cur)
    cur += relativedelta(months=1)

rows = []
for dt in months_list:
    key = dt.strftime("%Y-%m")
    hist_keys = [(dt - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m") for i in range(LOOK_BACK)]
    hist_vals = [real_data_full[k] for k in hist_keys]
    pred = predict_next(hist_vals, dt)
    actual = real_data_full[key]
    rows.append((key, actual, pred, abs(pred - actual) / actual * 100))

df_walk = pd.DataFrame(rows, columns=["month", "actual", "pred_walkforward", "ape_walkforward"])

supply_rec = {k: v for k, v in real_data_full.items() if datetime.strptime(k, "%Y-%m") <= TRAIN_END}
for dt in months_list:
    key = dt.strftime("%Y-%m")
    hist_keys = [(dt - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m") for i in range(LOOK_BACK)]
    hist_vals = [supply_rec[k] for k in hist_keys]
    supply_rec[key] = predict_next(hist_vals, dt)

df_walk["pred_recursive"] = [supply_rec[m] for m in df_walk["month"]]
df_walk["ape_recursive"] = (df_walk["pred_recursive"] - df_walk["actual"]).abs() / df_walk["actual"] * 100

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
print(df_walk.to_string(index=False))
print()
print(f"評估區間：{holdout_start:%Y-%m} ~ {holdout_end:%Y-%m}（{len(months_list)} 個月，訓練時完全未見過）")
print(f"擴充版 One-step walk-forward MAPE：{df_walk['ape_walkforward'].mean():.2f}%")
print(f"擴充版 Recursive（實際部署情境）MAPE：{df_walk['ape_recursive'].mean():.2f}%")
