# ==========================================
# Entrenamiento y empaquetado de modelos
# Proyecto DSA - Transporte Minero
# ==========================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score
import joblib

# Configuración de MLflow
mlflow.set_tracking_uri("http://18.208.126.149:5000")
mlflow.set_experiment("Proyecto_DSA_Experimentos")

# Cargar dataset procesado
df = pd.read_csv("data/df_final.csv")
print("Dataset cargado:", df.shape)

# División train/test
df = df.sort_values("DateTime").reset_index(drop=True)
n = int(len(df) * 0.8)
train_df, test_df = df.iloc[:n], df.iloc[n:]

# -------- Modelo 1: EmptyStopTime --------
features_reg = [c for c in df.columns if c not in [
    "Equipment", "DateTime", "StartLatLong", "DestinationLatLong",
    "CycleTime", "EfficientCycle", "EmptyStopTime", "LoadStopTime"
] + [c for c in df.columns if c.startswith("CycleClass_")]]

X_tr, X_te = train_df[features_reg].fillna(0), test_df[features_reg].fillna(0)
y_tr, y_te = train_df["EmptyStopTime"], test_df["EmptyStopTime"]

with mlflow.start_run(run_name="reg_EmptyStopTime_Final"):
    model_empty = RandomForestRegressor(
        n_estimators=214, max_depth=15, max_features=1.0,
        min_samples_leaf=1, min_samples_split=6, random_state=42, n_jobs=-1
    )
    model_empty.fit(X_tr, y_tr)
    preds = model_empty.predict(X_te)
    mae, rmse = mean_absolute_error(y_te, preds), np.sqrt(mean_squared_error(y_te, preds))
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse})
    mlflow.sklearn.log_model(model_empty, artifact_path="model_EmptyStopTime_Final")

# -------- Modelo 2: LoadStopTime --------
y_tr, y_te = train_df["LoadStopTime"], test_df["LoadStopTime"]

with mlflow.start_run(run_name="reg_LoadStopTime_Final"):
    model_load = RandomForestRegressor(
        n_estimators=892, max_depth=8, max_features=0.8,
        min_samples_leaf=3, min_samples_split=4, random_state=42, n_jobs=-1
    )
    model_load.fit(X_tr, y_tr)
    preds = model_load.predict(X_te)
    mae, rmse = mean_absolute_error(y_te, preds), np.sqrt(mean_squared_error(y_te, preds))
    mlflow.log_metrics({"MAE": mae, "RMSE": rmse})
    mlflow.sklearn.log_model(model_load, artifact_path="model_LoadStopTime_Final")

# -------- Modelo 3: Clasificación --------
features_clf = [
    "Payload", "FuelBurned", "DistanceTravelled", "EmptyTravelTime", "EmptyTravelDistance",
    "LoadTime", "LoadTravelTime", "LoadTravelDistance", "TotalStopTime",
    "Hour_1", "Hour_2", "Hour_3", "Hour_4", "Hour_5", "Hour_6", "Hour_7",
    "Hour_8", "Hour_9", "Hour_10", "Hour_11", "Hour_12", "Hour_13", "Hour_14",
    "Hour_15", "Hour_16", "Hour_17", "Hour_18", "Hour_19", "Hour_20", "Hour_21",
    "Hour_22", "Hour_23", "Shift_Night"
]

X_tr, X_te = train_df[features_clf].fillna(0), test_df[features_clf].fillna(0)
y_tr, y_te = train_df["EfficientCycle"], test_df["EfficientCycle"]

with mlflow.start_run(run_name="clf_EfficientCycle_Final"):
    model_clf = GradientBoostingClassifier(random_state=42)
    model_clf.fit(X_tr, y_tr)
    f1, auc = f1_score(y_te, model_clf.predict(X_te)), roc_auc_score(y_te, model_clf.predict_proba(X_te)[:, 1])
    mlflow.log_metrics({"F1": f1, "ROC_AUC": auc})
    mlflow.sklearn.log_model(model_clf, artifact_path="model_EfficientCycle_Final")

# -------- Guardar modelos --------
joblib.dump(model_empty, "models/model_EmptyStopTime_Final.pkl")
joblib.dump(model_load, "models/model_LoadStopTime_Final.pkl")
joblib.dump(model_clf, "models/model_EfficientCycle_Final.pkl")

print(" Modelos guardados correctamente en carpeta /models")
