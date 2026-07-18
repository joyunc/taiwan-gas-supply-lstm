from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pickle
import math
import keras

app = FastAPI(
    title="天然氣供給量預測 API",
    description="使用改進 LSTM 模型（look_back=12 + 季節特徵）預測台灣天然氣供給量",
    version="2.0.0",
)

model = keras.models.load_model("model_lstm.keras")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

LOOK_BACK = 12

# 啟動時載入資料集，建立月份 → 供給量的查詢表
_df = pd.read_csv("天然氣供給月資料.csv", usecols=[2], skiprows=117)
_df.columns = ["supply"]
_df.index = pd.date_range(start="2013-10", periods=len(_df), freq="MS")
real_data: dict[str, float] = {
    dt.strftime("%Y-%m"): float(val)
    for dt, val in zip(_df.index, _df["supply"])
}
DATA_START = min(datetime.strptime(m, "%Y-%m") for m in real_data)
DATA_END   = max(datetime.strptime(m, "%Y-%m") for m in real_data)


def _step(scaled_history: list[float], target_month_num: int) -> float:
    """單步預測，回傳縮放後的值。"""
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


def _build_supply(up_to: datetime) -> dict[str, float]:
    """
    從真實資料出發，遞迴預測直到 up_to 月份，
    回傳包含真實值與預測值的完整查詢表。
    """
    supply = dict(real_data)

    current = DATA_END + relativedelta(months=1)
    while current <= up_to:
        key = current.strftime("%Y-%m")
        prev = [
            supply[(current - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m")]
            for i in range(LOOK_BACK)
        ]
        scaled_prev = scaler.transform(
            np.array(prev).reshape(-1, 1).astype("float32")
        ).flatten().tolist()
        pred_scaled = _step(scaled_prev, current.month)
        supply[key] = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        current += relativedelta(months=1)

    return supply


def _predict(target_month_str: str) -> tuple[float, str]:
    """
    預測指定月份，自動判斷是否需要遞迴填補中間月份。
    回傳 (預測值, 資料來源說明)。
    """
    target_dt = datetime.strptime(target_month_str, "%Y-%m")
    earliest = DATA_START + relativedelta(months=LOOK_BACK)

    if target_dt < earliest:
        raise HTTPException(
            status_code=400,
            detail=f"目標月份最早為 {earliest.strftime('%Y-%m')}（需要 {LOOK_BACK} 個月歷史資料）",
        )

    supply = _build_supply(target_dt)

    if target_dt > DATA_END:
        # 已在 _build_supply 中計算，直接取出
        return round(supply[target_month_str], 0), "recursive_prediction"

    # 目標月份在資料集內：用真實歷史資料做預測（可用來驗證模型）
    prev_12 = [
        supply[(target_dt - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m")]
        for i in range(LOOK_BACK)
    ]
    scaled_12 = scaler.transform(
        np.array(prev_12).reshape(-1, 1).astype("float32")
    ).flatten().tolist()
    pred_scaled = _step(scaled_12, target_dt.month)
    pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
    return round(pred, 0), "historical_data"


def _actual_and_error(target_month_str: str, pred: float, source: str) -> tuple[Optional[float], Optional[float]]:
    """若目標月份在資料集內（historical_data），回傳真實值與 APE（絕對百分比誤差）。"""
    if source != "historical_data":
        return None, None
    actual = real_data[target_month_str]
    return actual, round(abs(pred - actual) / actual * 100, 2)


# ---------- Schema ----------

def _validate_month(v: str) -> str:
    try:
        datetime.strptime(v, "%Y-%m")
    except ValueError:
        raise ValueError("格式應為 YYYY-MM，例如 2024-01")
    return v


class PredictRequest(BaseModel):
    target_month: str

    @field_validator("target_month")
    @classmethod
    def check_format(cls, v):
        return _validate_month(v)


class PredictResponse(BaseModel):
    target_month: str
    predicted_supply: float
    unit: str
    data_source: str
    actual_supply: Optional[float] = None
    error_pct: Optional[float] = None


class MultiPredictRequest(BaseModel):
    start_month: str
    steps: int

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
    batch_mape: Optional[float] = None
    note: str


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "message": "天然氣供給量預測 API，請至 /docs 查看使用說明",
        "dataset_range": f"{DATA_START.strftime('%Y-%m')} ~ {DATA_END.strftime('%Y-%m')}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    單步預測：只需指定目標月份，API 自動從資料集取得歷史資料。
    超出資料集範圍的月份會自動遞迴預測中間各月。
    """
    pred, source = _predict(req.target_month)
    actual, error_pct = _actual_and_error(req.target_month, pred, source)
    return PredictResponse(
        target_month=req.target_month,
        predicted_supply=pred,
        unit="千立方公尺",
        data_source=source,
        actual_supply=actual,
        error_pct=error_pct,
    )


@app.post("/predict/multi", response_model=MultiPredictResponse)
def predict_multi(req: MultiPredictRequest):
    """
    多步預測：從 start_month 開始連續預測 steps 個月。
    建議 steps ≤ 6，步數越多遞迴誤差累積越大。
    """
    # 一次性建立供給量查詢表至最遠需要的月份
    end_dt = datetime.strptime(req.start_month, "%Y-%m") + relativedelta(months=req.steps - 1)
    supply = _build_supply(end_dt)

    results = []
    current = datetime.strptime(req.start_month, "%Y-%m")
    earliest = DATA_START + relativedelta(months=LOOK_BACK)

    for _ in range(req.steps):
        if current < earliest:
            raise HTTPException(
                status_code=400,
                detail=f"月份不得早於 {earliest.strftime('%Y-%m')}",
            )

        if current > DATA_END:
            pred = round(supply[current.strftime("%Y-%m")], 0)
            source = "recursive_prediction"
        else:
            prev_12 = [
                supply[(current - relativedelta(months=LOOK_BACK - i)).strftime("%Y-%m")]
                for i in range(LOOK_BACK)
            ]
            scaled_12 = scaler.transform(
                np.array(prev_12).reshape(-1, 1).astype("float32")
            ).flatten().tolist()
            pred_scaled = _step(scaled_12, current.month)
            pred = round(float(scaler.inverse_transform([[pred_scaled]])[0][0]), 0)
            source = "historical_data"

        actual, error_pct = _actual_and_error(current.strftime("%Y-%m"), pred, source)
        results.append(PredictResponse(
            target_month=current.strftime("%Y-%m"),
            predicted_supply=pred,
            unit="千立方公尺",
            data_source=source,
            actual_supply=actual,
            error_pct=error_pct,
        ))
        current += relativedelta(months=1)

    errors = [r.error_pct for r in results if r.error_pct is not None]
    batch_mape = round(sum(errors) / len(errors), 2) if errors else None

    return MultiPredictResponse(
        predictions=results,
        batch_mape=batch_mape,
        note="data_source='recursive_prediction' 表示超出資料集範圍，誤差會隨步數累積；"
             "batch_mape 僅根據落在資料集內（historical_data）的月份計算",
    )
