import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# ========================================================
# CONFIGURACIÓN INICIAL
# ========================================================
API_URL = "http://127.0.0.1:8500"  # Puerto FastAPI
DATA_PATH = "/Users/luiscortes/Desktop/Proyecto_DSA/data/df_final.csv"  # Ruta dataset

# ========================================================
# CARGA DE DATOS
# ========================================================
try:
    df = pd.read_csv(DATA_PATH)
    print(" Datos cargados correctamente:", df.shape)
except Exception as e:
    print(f" Error al cargar datos: {e}")
    df = pd.DataFrame()

# ========================================================
# PREPARACIÓN DE DATOS
# ========================================================
if not df.empty:
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["Shift"] = df["Shift_Night"].apply(lambda x: "Night" if x else "Day")

    # Reconstruir columna de hora numérica
    hour_cols = [c for c in df.columns if c.startswith("Hour_")]
    if hour_cols:
        df["Hour"] = (
            df[hour_cols]
            .apply(lambda row: next((int(c.split("_")[1]) for c in hour_cols if row[c]), 0), axis=1)
        )
    else:
        df["Hour"] = 0

    # Limpieza
    df = df.dropna(subset=["CycleTime", "EmptyStopTime", "LoadStopTime"])
    df["Hour"] = df["Hour"].astype(int)
    df = df[df["Hour"] > 0]

    total = len(df)
    pct_ef = round((df["EfficientCycle"].sum() / total) * 100, 2)
    det_vacio = len(df[df["EmptyStopTime"] > 6])
    det_cargado = len(df[df["LoadStopTime"] > 6])
else:
    pct_ef, det_vacio, det_cargado, total = 0, 0, 0, 0

# ========================================================
# DASH APP
# ========================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Tablero Predictivo - Transporte Minero"

# ========================================================
# LAYOUT
# ========================================================
app.layout = dbc.Container([
    html.H2("Calidad de ciclos y detenciones improductivas de camiones 793", className="text-center text-danger mt-3"),
    html.Hr(),

    # Indicadores principales
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Ciclos Eficientes (%)"),
            dbc.CardBody(html.H3(f"{pct_ef}%", className="text-center text-success"))
        ], color="dark", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Detenciones Camion Vacío > 6 min"),
            dbc.CardBody(html.H3(det_vacio, className="text-center text-warning"))
        ], color="secondary", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Detenciones Camion Cargado > 6 min"),
            dbc.CardBody(html.H3(det_cargado, className="text-center text-danger"))
        ], color="dark", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Total de Ciclos Registrados"),
            dbc.CardBody(html.H3(total, className="text-center text-info"))
        ], color="dark", inverse=True), width=3)
    ], justify="center"),

    html.Br(),

    # --- Nueva gráfica circular de eficiencia ---
    dbc.Card([
        dbc.CardHeader("Distribución de Ciclos Eficientes vs. Ineficientes", className="bg-secondary text-white"),
        dbc.CardBody([
            dcc.Graph(
                id="efficiency_pie",
                figure=px.pie(
                    values=[df["EfficientCycle"].sum(), len(df) - df["EfficientCycle"].sum()],
                    names=["Eficientes", "Ineficientes"],
                    color=["Eficientes", "Ineficientes"],
                    color_discrete_map={"Eficientes": "#2ECC40", "Ineficientes": "#FF4136"},
                    hole=0.4,
                    title="Proporción de Ciclos Eficientes e Ineficientes"
                ).update_layout(
                    paper_bgcolor="#222222",
                    plot_bgcolor="#2c2c2c",
                    font_color="white",
                    title_font_size=16
                )
            )
        ])
    ], color="dark", outline=True),

    html.Br(),

    # ========================================================
    # GRÁFICAS DE TENDENCIA
    # ========================================================
    dbc.Card([
        dbc.CardHeader("Tendencia de tiempos de ciclo y detenciones por hora", className="bg-secondary text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Filtrar por turno:", className="text-white"),
                    dcc.Dropdown(
                        id="turn_filter",
                        options=[
                            {"label": "Todos", "value": "All"},
                            {"label": "Día", "value": "Day"},
                            {"label": "Noche", "value": "Night"}
                        ],
                        value="All",
                        clearable=False,
                        style={"color": "black"}
                    )
                ], width=3)
            ]),
            html.Br(),
            dcc.Graph(id="cycle-hour-graph"),
            html.Br(),
            dcc.Graph(id="stop-comparison-graph")
        ])
    ], color="dark", outline=True),

    html.Br(),

    # ========================================================
    # PREDICCIONES
    # ========================================================
    html.H4("Predicciones de Modelos", className="text-danger"),

    # Modelo EmptyStopTime
    dbc.Card([
        dbc.CardHeader("Modelo 1 - Tiempo detenido en vacío (min)"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Input(id="payload_empty", placeholder="Payload", type="number")),
                dbc.Col(dcc.Input(id="fuel_empty", placeholder="FuelBurned", type="number")),
                dbc.Col(dcc.Input(id="cycle_empty", placeholder="CycleTime", type="number")),
                dbc.Col(dcc.Input(id="emptytravel", placeholder="EmptyTravelTime", type="number")),
                dbc.Col(dcc.Input(id="loadtime_empty", placeholder="LoadTime", type="number")),
                dbc.Col(dcc.Input(id="loadtravel_empty", placeholder="LoadTravelTime", type="number")),
                dbc.Col(dcc.Input(id="hour_empty", placeholder="Hour (1-23)", type="number")),
                dbc.Col(dcc.Dropdown(
                    id="shift_empty",
                    options=[{"label": "Día", "value": False}, {"label": "Noche", "value": True}],
                    value=False, clearable=False))
            ]),
            html.Br(),
            dbc.Button("Predecir ciclo vacío", id="predict_empty_btn", color="primary"),
            html.Br(), html.Div(id="empty_output", className="text-info")
        ])
    ], color="dark", outline=True),

    html.Br(),

    # Modelo LoadStopTime
    dbc.Card([
        dbc.CardHeader("Modelo 2 - Tiempo detenido en cargue (min)"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Input(id="payload_load", placeholder="Payload", type="number")),
                dbc.Col(dcc.Input(id="fuel_load", placeholder="FuelBurned", type="number")),
                dbc.Col(dcc.Input(id="cycle_load", placeholder="CycleTime", type="number")),
                dbc.Col(dcc.Input(id="emptytravel_load", placeholder="EmptyTravelTime", type="number")),
                dbc.Col(dcc.Input(id="loadtime_load", placeholder="LoadTime", type="number")),
                dbc.Col(dcc.Input(id="loadtravel_load", placeholder="LoadTravelTime", type="number")),
                dbc.Col(dcc.Input(id="hour_load", placeholder="Hour (1-23)", type="number")),
                dbc.Col(dcc.Dropdown(
                    id="shift_load",
                    options=[{"label": "Día", "value": False}, {"label": "Noche", "value": True}],
                    value=False, clearable=False))
            ]),
            html.Br(),
            dbc.Button("Predecir ciclo cargado", id="predict_load_btn", color="primary"),
            html.Br(), html.Div(id="load_output", className="text-info")
        ])
    ], color="dark", outline=True),

    html.Br(),

    # Modelo EfficientCycle
    dbc.Card([
        dbc.CardHeader("Modelo 3 - Clasificación de Ciclo Eficiente / Ineficiente"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Input(id="payload_eff", placeholder="Payload", type="number")),
                dbc.Col(dcc.Input(id="fuel_eff", placeholder="FuelBurned", type="number")),
                dbc.Col(dcc.Input(id="dist_eff", placeholder="DistanceTravelled", type="number")),
                dbc.Col(dcc.Input(id="emptytime_eff", placeholder="EmptyTravelTime", type="number")),
                dbc.Col(dcc.Input(id="emptydist_eff", placeholder="EmptyTravelDistance", type="number")),
                dbc.Col(dcc.Input(id="loadtime_eff", placeholder="LoadTime", type="number")),
                dbc.Col(dcc.Input(id="loadtravel_eff", placeholder="LoadTravelTime", type="number")),
                dbc.Col(dcc.Input(id="loadtraveldist_eff", placeholder="LoadTravelDistance", type="number")),
                dbc.Col(dcc.Input(id="totalstop_eff", placeholder="TotalStopTime", type="number")),
                dbc.Col(dcc.Input(id="hour_eff", placeholder="Hour (1-23)", type="number")),
                dbc.Col(dcc.Dropdown(
                    id="shift_eff",
                    options=[{"label": "Día", "value": False}, {"label": "Noche", "value": True}],
                    value=False, clearable=False))
            ]),
            html.Br(),
            dbc.Button("Ejecutar predicción de eficiencia", id="predict_efficiency_btn", color="danger"),
            html.Br(), html.Div(id="efficiency_output", className="text-warning")
        ])
    ], color="dark", outline=True)
], fluid=True)

# ========================================================
# CALLBACKS
# ========================================================
@app.callback(
    [Output("cycle-hour-graph", "figure"),
     Output("stop-comparison-graph", "figure")],
    Input("turn_filter", "value")
)
def update_graphs(turn):
    try:
        if df.empty or "Hour" not in df.columns:
            return px.line(title="Sin datos"), px.bar(title="Sin datos")

        filtered = df if turn == "All" else df[df["Shift"] == turn]
        if filtered.empty:
            return px.line(title="Sin datos"), px.bar(title="Sin datos")

        cycle_data = filtered.groupby("Hour", as_index=False)["CycleTime"].mean()
        stops = filtered.groupby("Hour", as_index=False)[["EmptyStopTime", "LoadStopTime"]].mean()

        fig1 = px.line(
            cycle_data, x="Hour", y="CycleTime",
            title="Tendencia del CycleTime promedio por hora",
            markers=True, color_discrete_sequence=["#FF4136"]
        )

        fig2 = px.bar(
            stops, x="Hour", y=["EmptyStopTime", "LoadStopTime"],
            title="Promedio de detenciones por hora (Vacío vs Cargue)",
            barmode="group", color_discrete_sequence=["#AAAAAA", "#FF0000"]
        )

        for fig in [fig1, fig2]:
            fig.update_layout(
                paper_bgcolor="#222222",
                plot_bgcolor="#2c2c2c",
                font_color="white",
                title_font_size=16,
                xaxis_title="Hora del día",
                yaxis_title="Tiempo (min)",
                xaxis=dict(gridcolor="#444444"),
                yaxis=dict(gridcolor="#444444")
            )
        return fig1, fig2
    except Exception as e:
        print(f"⚠️ Error en update_graphs: {e}")
        return px.line(title="Error al graficar"), px.bar(title="Error al graficar")

# ========================================================
# FUNCIONES DE PREDICCIÓN
# ========================================================
def build_regression_json(payload, fuel, cycle, emptytravel, loadtime, loadtravel, hour, shift):
    return {
        "Payload": payload, "FuelBurned": fuel, "CycleTime": cycle,
        "EmptyTravelTime": emptytravel, "LoadTime": loadtime, "LoadTravelTime": loadtravel,
        **{f"Hour_{i}": (i == hour) for i in range(1, 24)},
        "Shift_Night": shift
    }
# --- Predicciones (callbacks) ---
@app.callback(Output("empty_output", "children"), Input("predict_empty_btn", "n_clicks"),
              State("payload_empty", "value"), State("fuel_empty", "value"),
              State("cycle_empty", "value"), State("emptytravel", "value"),
              State("loadtime_empty", "value"), State("loadtravel_empty", "value"),
              State("hour_empty", "value"), State("shift_empty", "value"))
def predict_empty(n, payload, fuel, cycle, emptytravel, loadtime, loadtravel, hour, shift):
    if not n:
        return ""
    try:
        features = build_regression_json(payload, fuel, cycle, emptytravel, loadtime, loadtravel, hour, shift)
        res = requests.post(f"{API_URL}/predict_empty_stop", json=features)
        return f"Predicción: {res.json()['EmptyStopTime_pred']} min"
    except Exception as e:
        return f"⚠️ Error: {e}"

@app.callback(Output("load_output", "children"), Input("predict_load_btn", "n_clicks"),
              State("payload_load", "value"), State("fuel_load", "value"),
              State("cycle_load", "value"), State("emptytravel_load", "value"),
              State("loadtime_load", "value"), State("loadtravel_load", "value"),
              State("hour_load", "value"), State("shift_load", "value"))
def predict_load(n, payload, fuel, cycle, emptytravel, loadtime, loadtravel, hour, shift):
    if not n:
        return ""
    try:
        features = build_regression_json(payload, fuel, cycle, emptytravel, loadtime, loadtravel, hour, shift)
        res = requests.post(f"{API_URL}/predict_load_stop", json=features)
        return f" Predicción: {res.json()['LoadStopTime_pred']} min"
    except Exception as e:
        return f"⚠️ Error: {e}"

@app.callback(Output("efficiency_output", "children"),
              Input("predict_efficiency_btn", "n_clicks"),
              State("payload_eff", "value"), State("fuel_eff", "value"),
              State("dist_eff", "value"), State("emptytime_eff", "value"),
              State("emptydist_eff", "value"), State("loadtime_eff", "value"),
              State("loadtravel_eff", "value"), State("loadtraveldist_eff", "value"),
              State("totalstop_eff", "value"), State("hour_eff", "value"),
              State("shift_eff", "value"))
def predict_efficiency(n, payload, fuel, dist, emptytime, emptydist, loadtime, loadtravel, loadtraveldist, totalstop, hour, shift):
    if not n:
        return ""
    try:
        features = {
            "Payload": payload, "FuelBurned": fuel, "DistanceTravelled": dist,
            "EmptyTravelTime": emptytime, "EmptyTravelDistance": emptydist,
            "LoadTime": loadtime, "LoadTravelTime": loadtravel,
            "LoadTravelDistance": loadtraveldist, "TotalStopTime": totalstop,
            **{f"Hour_{i}": (i == hour) for i in range(1, 24)},
            "Shift_Night": shift
        }
        res = requests.post(f"{API_URL}/predict_efficiency", json=features)
        js = res.json()
        return f"⚙️ Predicción: {js['Label']} (Probabilidad: {js['EfficientCycle_prob']:.2f})"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ========================================================
# EJECUCIÓN
# ========================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
