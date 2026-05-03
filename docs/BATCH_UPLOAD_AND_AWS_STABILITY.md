# Batch upload + estabilidad AWS

## Problema observado

El 502 puede aparecer aunque la app se vea fluida porque Streamlit mantiene una conexión WebSocket con el navegador. Si el contenedor se queda sin recursos, tarda demasiado o el ALB cierra una conexión inactiva, puede aparecer `Connection failed with status 502`.

## Solución aplicada

1. Mantener navegación lazy/optimizada.
2. Subir recursos de la task ECS/Fargate.
3. Subir idle timeout del ALB.
4. Reintegrar la inferencia batch por CSV cargado sin tocar las demás pestañas.

## CSVs de prueba

Este paquete trae tres CSVs genéricos en `examples/batch_upload/`:

- `batch_upload_example_mixed.csv`
- `batch_upload_example_recent_demand.csv`
- `batch_upload_example_inactive.csv`

Para crear CSVs 100% compatibles con el modelo real de tu repo, ejecuta:

```bash
PYTHONPATH=. uv run python scripts/85_generate_batch_upload_examples.py
```

Esto genera:

```text
data/examples/batch_upload/batch_upload_mixed_catalog_sample.csv
data/examples/batch_upload/batch_upload_recent_demand_sample.csv
data/examples/batch_upload/batch_upload_inactive_sample.csv
```

## Uso en la app

En Batch CFO, baja a `Batch por archivo cargado`, sube uno de los CSVs y presiona:

```text
Ejecutar inferencia sobre archivo cargado
```

Si el CSV de ejemplo no trae todas las columnas del modelo, activa:

```text
Modo prueba: rellenar columnas faltantes con 0
```

Para inferencia real, usa el CSV generado por `85_generate_batch_upload_examples.py` o uno con todas las columnas requeridas por el modelo.
