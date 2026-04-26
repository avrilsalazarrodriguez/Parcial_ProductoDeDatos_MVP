FROM python:3.11-slim

WORKDIR /app

# Dejamos /app en el PYTHONPATH para que Streamlit pueda importar backend/ y src/.
ENV PYTHONPATH=/app

# Instalamos curl para que ECS pueda revisar el healthcheck de Streamlit.
RUN apt-get update && apt-get install -y curl libgomp1 && rm -rf /var/lib/apt/lists/*

# Copiamos las dependencias para aprovechar cache de Docker.
COPY frontend/requirements.txt ./frontend/requirements.txt

# Instalamos las librerías necesarias para la app y conexión a AWS/RDS.
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Aqui copiamos la aplicación Streamlit.
COPY frontend/ ./frontend/

# Copiamos módulos backend que usará la app.
COPY backend/ ./backend/

# Copiamos el modelo entrenado y artefactos existentes.
COPY artifacts/ ./artifacts/

# Copiamos datos preparados y predicciones existentes.
COPY data/ ./data/

# Copiamos el código src existente para reutilizar lógica del proyecto.
COPY src/ ./src/

EXPOSE 8501

# ECS revisa que Streamlit esté vivo antes de mandar tráfico.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Arrancamos Streamlit escuchando en todas las interfaces dentro del contenedor.
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]


