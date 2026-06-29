from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from datetime import datetime
import numpy as np
import pickle
import math
import keras

app = FastAPI(
    title="天然氣供給量預測 API",
    description="使用改進 LSTM 模型（look_back=12 + 季節特徵）預測台灣下一個月天然氣供給量",
    version="1.0.0",
)

model = keras.models.load_model("model_lstm.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

LOOK_BACK = 12


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
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError:
            raise ValueError("target_month 格式應為 YYYY-MM，例如 2024-01")
        return v


class PredictResponse(BaseModel):
    predicted_supply: float
    unit: str
    target_month: str


@app.get("/")
def root():
    return {"message": "天然氣供給量預測 API，請至 /docs 查看使用說明"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    month = datetime.strptime(req.target_month, "%Y-%m").month

    supply = np.array(req.supply_history).reshape(-1, 1).astype("float32")
    scaled = scaler.transform(supply).flatten()

    features = []
    for i in range(LOOK_BACK):
        m = ((month - LOOK_BACK + i - 1) % 12 + 12) % 12 + 1
        features.append([
            scaled[i],
            math.sin(2 * math.pi * (m - 1) / 12),
            math.cos(2 * math.pi * (m - 1) / 12),
        ])

    X = np.array(features).reshape(1, LOOK_BACK, 3)
    pred = float(scaler.inverse_transform(model.predict(X, verbose=0))[0][0])

    return PredictResponse(
        predicted_supply=round(pred, 0),
        unit="千立方公尺",
        target_month=req.target_month,
    )
