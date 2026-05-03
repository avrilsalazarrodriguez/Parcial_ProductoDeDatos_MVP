"""
Tabla de eventos de uso para observabilidad del MVP.

Esta tabla permite contar acciones de la app:
- inferencia individual
- upload de archivo
- batch CFO generado
"""

from backend.db import get_engine
from sqlalchemy import text

DDL = """
CREATE TABLE IF NOT EXISTS app_usage_events (
    event_id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    shop_id INTEGER,
    item_id INTEGER,
    records_count INTEGER,
    status VARCHAR(50) NOT NULL DEFAULT 'success',
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

with get_engine().begin() as conn:
    conn.execute(text(DDL))

print("Tabla app_usage_events lista.")
