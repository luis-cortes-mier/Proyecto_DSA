from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# =======================================================
# Inicialización de la API
# =======================================================
app = FastAPI(title="API Modelos Predictivos Transporte Minero")

# =======================================================
# Cargar modelos (rutas absolutas seguras)
# =======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_empty = joblib.load(os.path.join(MODEL_DIR, "model_EmptyStopTime_V2_Tuned.pkl"))
model_load = joblib.load(os.path.join(MODEL_DIR, "model_LoadStopTime_V2_Tuned.pkl"))
model_clf = joblib.load(os.path.join(MODEL_DIR, "model_EfficientCycle.pkl"))

print("✅ Modelos cargados correctamente desde:", MODEL_DIR)

# =======================================================
# Definir esquemas (entradas) para los modelos de regresión
# =======================================================
class RegressionData(BaseModel):
    Payload: float
    FuelBurned: float
    CycleTime: float
    EmptyTravelTime: float
    LoadTime: float
    LoadTravelTime: float
    Hour_1: bool
    Hour_2: bool
    Hour_3: bool
    Hour_4: bool
    Hour_5: bool
    Hour_6: bool
    Hour_7: bool
    Hour_8: bool
    Hour_9: bool
    Hour_10: bool
    Hour_11: bool
    Hour_12: bool
    Hour_13: bool
    Hour_14: bool
    Hour_15: bool
    Hour_16: bool
    Hour_17: bool
    Hour_18: bool
    Hour_19: bool
    Hour_20: bool
    Hour_21: bool
    Hour_22: bool
    Hour_23: bool
    Shift_Night: bool


# =======================================================
# Rutas base
# =======================================================
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


# =======================================================
# 1️⃣ Modelo EmptyStopTime
# =======================================================
@app.post("/predict_empty_stop")
def predict_empty_stop(data: RegressionData):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    pred = model_empty.predict(X)[0]
    return {"EmptyStopTime_pred": round(float(pred), 2)}


# =======================================================
# 2️⃣ Modelo LoadStopTime
# =======================================================
@app.post("/predict_load_stop")
def predict_load_stop(data: RegressionData):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    pred = model_load.predict(X)[0]
    return {"LoadStopTime_pred": round(float(pred), 2)}


# =======================================================
# 3️⃣ Modelo EfficientCycle (clasificación, 33 columnas)
# =======================================================
@app.post("/predict_efficiency")
async def predict_efficiency(request: Request):
    data = await request.json()  # recibe el JSON crudo del dashboard
    X = pd.DataFrame([data])

    # Variables esperadas (33 columnas)
    expected = [
        "Payload", "FuelBurned", "DistanceTravelled", "EmptyTravelTime", "EmptyTravelDistance",
        "LoadTime", "LoadTravelTime", "LoadTravelDistance", "TotalStopTime",
        "Hour_1", "Hour_2", "Hour_3", "Hour_4", "Hour_5", "Hour_6", "Hour_7",
        "Hour_8", "Hour_9", "Hour_10", "Hour_11", "Hour_12", "Hour_13", "Hour_14",
        "Hour_15", "Hour_16", "Hour_17", "Hour_18", "Hour_19", "Hour_20", "Hour_21",
        "Hour_22", "Hour_23", "Shift_Night"
    ]

    # Alinear columnas con el modelo
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[expected]

    # --- Predicción ---
    pred = model_clf.predict(X)[0]
    prob = model_clf.predict_proba(X)[0][1]
    label = "Eficiente" if pred == 1 else "Ineficiente"

    return {
        "EfficientCycle_pred": int(pred),
        "EfficientCycle_prob": round(float(prob), 3),
        "Label": label
    }
