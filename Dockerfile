# ===============================
#  DOCKERFILE - Proyecto DSA
# Transporte Minero Predictivo
# ===============================

# Imagen base
FROM python:3.10-slim

# Configuración básica
WORKDIR /app

# Evita creación de archivos .pyc y buffering de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#  Copia de archivos
COPY . /app

#  Copiar modelos y dataset al contenedor
COPY models/*.pkl /app/models/
COPY data/df_final.csv /app/data/df_final.csv

#  Instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#  Exponer los puertos de la API y el Dashboard
EXPOSE 8000 8050

#  Variables de entorno importantes
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV API_URL=http://api:8000

#  Comando por defecto (para cuando no se use docker-compose)
CMD ["bash"]