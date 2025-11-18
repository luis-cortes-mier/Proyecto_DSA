"""
Paquete principal del proyecto DSA - Transporte Minero
Contiene los módulos de API, dashboard y entrenamiento de modelos.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
MLRUNS_DIR = os.path.join(os.path.dirname(BASE_DIR), "mlruns")

__all__ = ["BASE_DIR", "DATA_DIR", "MODELS_DIR", "MLRUNS_DIR"]
