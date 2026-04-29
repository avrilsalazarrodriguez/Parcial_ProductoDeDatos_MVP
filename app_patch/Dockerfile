FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

COPY frontend/ ./frontend/
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
COPY data/ ./data/
COPY src/ ./src/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD [
    "streamlit", "run", "frontend/app.py",
    "--server.address=0.0.0.0",
    "--server.port=8501",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
    "--browser.gatherUsageStats=false"
]
