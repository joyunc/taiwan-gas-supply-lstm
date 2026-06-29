from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pickle
import math
import keras

app = FastAPI(
    title="天然氣供給量預測 API",
    description="使用改進 LSTM 模型（look_back=12 + 季節特徵）預測台灣天然氣供給量",
    version="1.0.0",
)

model = keras.models.load_model("model_lstm.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

LOOK_BACK = 12


def _build_features(scaled_history: list[float], target_month: int) -> np.ndarray:
    """將 12 期縮放後的供給量 + 月份 sin/cos 組成模型輸入。"""
    features = []
    for i in range(LOOK_BACK):
        m = ((target_month - LOOK_BACK + i - 1) % 12 + 12) % 12 + 1
        features.append([
            scaled_history[i],
            math.sin(2 * math.pi * (m - 1) / 12),
            math.cos(2 * math.pi * (m - 1) / 12),
        ])
    return np.array(features).reshape(1, LOOK_BACK, 3)


def _validate_month(v: str) -> str:
    try:
        datetime.strptime(v, "%Y-%m")
    except ValueError:
        raise ValueError("格式應為 YYYY-MM，例如 2024-01")
    return v


# ---------- Schema ----------

class PredictRequest(BaseModel):
    supply_history: list[float]
    target_month: str

    @field_validator("supply_history")
    @classmethod
    def check_length(cls, v):
        if len(v) != LOOK_BACK:
            raise ValueError(f"supply_history 需要剛好 {LOOK_BACK} 筆（最近 12 個月）")
        return v

    @field_validator("target_month")
    @classmethod
    def check_format(cls, v):
        return _validate_month(v)


class PredictResponse(BaseModel):
    predicted_supply: float
    unit: str
    target_month: str


class MultiPredictRequest(BaseModel):
    supply_history: list[float]
    start_month: str
    steps: int

    @field_validator("supply_history")
    @classmethod
    def check_length(cls, v):
        if len(v) != LOOK_BACK:
            raise ValueError(f"supply_history 需要剛好 {LOOK_BACK} 筆（最近 12 個月）")
        return v

    @field_validator("start_month")
    @classmethod
    def check_format(cls, v):
        return _validate_month(v)

    @field_validator("steps")
    @classmethod
    def check_steps(cls, v):
        if not (1 <= v <= 36):
            raise ValueError("steps 範圍為 1～36")
        return v


class MultiPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    note: str


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "天然氣供給量預測 API，請至 /docs 查看使用說明"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """單步預測：給定前 12 個月的實際數值，預測下一個月。"""
    month = datetime.strptime(req.target_month, "%Y-%m").month
    scaled = scaler.transform(
        np.array(req.supply_history).reshape(-1, 1).astype("float32")
    ).flatten().tolist()

    X = _build_features(scaled, month)
    pred = float(scaler.inverse_transform(model.predict(X, verbose=0))[0][0])

    return PredictResponse(
        predicted_supply=round(pred, 0),
        unit="千立方公尺",
        target_month=req.target_month,
    )


@app.post("/predict/multi", response_model=MultiPredictResponse)
def predict_multi(req: MultiPredictRequest):
    """多步遞迴預測：從 start_month 開始連續預測 steps 個月。

    每一步使用前一步的預測值作為輸入，誤差會隨步數累積。
    建議 steps ≤ 6，超過半年準確度會明顯下降。
    """
    history = scaler.transform(
        np.array(req.supply_history).reshape(-1, 1).astype("float32")
    ).flatten().tolist()

    results = []
    current = datetime.strptime(req.start_month, "%Y-%m")

    for _ in range(req.steps):
        X = _build_features(history, current.month)
        pred_scaled = float(model.predict(X, verbose=0)[0][0])
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])

        results.append(PredictResponse(
            predicted_supply=round(pred, 0),
            unit="千立方公尺",
            target_month=current.strftime("%Y-%m"),
        ))

        history = history[1:] + [pred_scaled]
        current += relativedelta(months=1)

    return MultiPredictResponse(
        predictions=results,
        note="遞迴預測：每步使用上一步預測值作為輸入，步數越多誤差累積越大",
    )
