from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# API DE MODELOS PREDICTIVOS - VERSIÓN ESTABLE (V2)
# ============================================================

app = FastAPI(title="API Modelos Predictivos Transporte Minero - V2")

# ------------------------------------------------------------
# CARGA DE MODELOS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_empty = joblib.load(os.path.join(MODEL_DIR, "model_EmptyStopTime_V2_Tuned.pkl"))
model_load = joblib.load(os.path.join(MODEL_DIR, "model_LoadStopTime_V2_Tuned.pkl"))
model_clf = joblib.load(os.path.join(MODEL_DIR, "model_EfficientCycle.pkl"))

print(f" Modelos cargados correctamente desde: {MODEL_DIR}")

# ------------------------------------------------------------
# ESQUEMAS DE ENTRADA
# ------------------------------------------------------------
class RegressionData(BaseModel):
    Payload: float
    FuelBurned: float
    CycleTime: float
    EmptyTravelTime: float
    LoadTime: float
    LoadTravelTime: float
    Hour_1: bool = False
    Hour_2: bool = False
    Hour_3: bool = False
    Hour_4: bool = False
    Hour_5: bool = False
    Hour_6: bool = False
    Hour_7: bool = False
    Hour_8: bool = False
    Hour_9: bool = False
    Hour_10: bool = False
    Hour_11: bool = False
    Hour_12: bool = False
    Hour_13: bool = False
    Hour_14: bool = False
    Hour_15: bool = False
    Hour_16: bool = False
    Hour_17: bool = False
    Hour_18: bool = False
    Hour_19: bool = False
    Hour_20: bool = False
    Hour_21: bool = False
    Hour_22: bool = False
    Hour_23: bool = False
    Shift_Night: bool = False


# ------------------------------------------------------------
# ENDPOINT PRINCIPAL
# ------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "✅ API funcionando correctamente.",
        "endpoints": [
            "/predict_empty_stop",
            "/predict_load_stop",
            "/predict_efficiency",
        ],
    }


# ------------------------------------------------------------
# MODELO 1 - EmptyStopTime
# ------------------------------------------------------------
@app.post("/predict_empty_stop")
def predict_empty_stop(data: RegressionData):
    try:
        X = np.array(list(data.dict().values())).reshape(1, -1)
        pred = model_empty.predict(X)[0]
        return {"EmptyStopTime_pred": round(float(pred), 2)}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------
# MODELO 2 - LoadStopTime
# ------------------------------------------------------------
@app.post("/predict_load_stop")
def predict_load_stop(data: RegressionData):
    try:
        X = np.array(list(data.dict().values())).reshape(1, -1)
        pred = model_load.predict(X)[0]
        return {"LoadStopTime_pred": round(float(pred), 2)}
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------
# MODELO 3 - EfficientCycle (Clasificación)
# ------------------------------------------------------------
@app.post("/predict_efficiency")
async def predict_efficiency(request: Request):
    try:
        data = await request.json()
        X = pd.DataFrame([data])

        expected = [
            "Payload", "FuelBurned", "DistanceTravelled", "EmptyTravelTime", "EmptyTravelDistance",
            "LoadTime", "LoadTravelTime", "LoadTravelDistance", "TotalStopTime",
            "Hour_1", "Hour_2", "Hour_3", "Hour_4", "Hour_5", "Hour_6", "Hour_7",
            "Hour_8", "Hour_9", "Hour_10", "Hour_11", "Hour_12", "Hour_13", "Hour_14",
            "Hour_15", "Hour_16", "Hour_17", "Hour_18", "Hour_19", "Hour_20", "Hour_21",
            "Hour_22", "Hour_23", "Shift_Night"
        ]

        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X[expected]

        pred = model_clf.predict(X)[0]
        prob = model_clf.predict_proba(X)[0][1]
        label = "Eficiente" if pred == 1 else "Ineficiente"

        return {
            "EfficientCycle_pred": int(pred),
            "EfficientCycle_prob": round(float(prob), 3),
            "Label": label,
        }
    except Exception as e:
        return {"error": str(e)}