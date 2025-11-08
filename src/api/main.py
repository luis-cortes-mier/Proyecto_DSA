from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ===============================
# Inicialización de la API
# ===============================
app = FastAPI(title="API Modelos Predictivos Transporte Minero")

# ===============================
# Cargar modelos (rutas absolutas seguras)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_empty = joblib.load(os.path.join(MODEL_DIR, "model_EmptyStopTime_V2_Tuned.pkl"))
model_load = joblib.load(os.path.join(MODEL_DIR, "model_LoadStopTime_V2_Tuned.pkl"))
model_clf = joblib.load(os.path.join(MODEL_DIR, "model_EfficientCycle.pkl"))

print("✅ Modelos cargados correctamente desde:", MODEL_DIR)

# ===============================
# Definir esquemas (entradas)
# ===============================

# Variables compartidas por los modelos de regresión
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

# Clasificación (usa mismas dummies)
class EfficiencyData(RegressionData):
    pass

# ===============================
# Rutas base
# ===============================
@app.get("/")
def home():
    return {"message": "✅ API funcionando correctamente. Usa /predict_empty_stop, /predict_load_stop o /predict_efficiency"}

# ===============================
# 1️⃣ Modelo EmptyStopTime
# ===============================
@app.post("/predict_empty_stop")
def predict_empty_stop(data: RegressionData):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    pred = model_empty.predict(X)[0]
    return {"EmptyStopTime_pred": round(float(pred), 2)}

# ===============================
# 2️⃣ Modelo LoadStopTime
# ===============================
@app.post("/predict_load_stop")
def predict_load_stop(data: RegressionData):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    pred = model_load.predict(X)[0]
    return {"LoadStopTime_pred": round(float(pred), 2)}

# ===============================
# 3️⃣ Modelo EfficientCycle (clasificación)
# ===============================
@app.post("/predict_efficiency")
def predict_efficiency(data: EfficiencyData):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    pred = model_clf.predict(X)[0]
    proba = model_clf.predict_proba(X)[0][1]
    label = "Eficiente ✅" if pred == 1 else "Ineficiente ⚠️"
    return {
        "EfficientCycle_pred": int(pred),
        "EfficientCycle_prob": round(float(proba), 3),
        "Label": label
    }
