# Guía de patch para `frontend/app.py`

No reemplaces toda la app. Tu compañera ya agregó funcionalidad, logs, RDS, S3 y uso de `model.joblib`. Este patch solo conecta los outputs de ModelOps S3.

## 1. Agrega estos imports

En la parte superior de `frontend/app.py`, junto a los otros imports, agrega:

```python
import os

from frontend.modelops_sections import (
    get_batch_df_with_modelops_fallback,
    render_modelops_debug_block,
    render_modelops_evaluation_block,
    render_modelops_registry_block,
    render_modelops_summary_block,
)
```

## 2. Agrega la bandera de activación

Después de las rutas `MODEL_PATH`, `VALID_PATH`, etc., agrega:

```python
USE_MODELOPS_S3 = os.getenv("USE_MODELOPS_S3", "true").lower() == "true"
```

## 3. Cambia `batch_df`

Busca la sección donde se construye el batch actual:

```python
batch_df = build_batch_table(test_pairs, submission)
```

Déjala así:

```python
batch_df = build_batch_table(test_pairs, submission)

if USE_MODELOPS_S3:
    batch_df = get_batch_df_with_modelops_fallback(batch_df)
```

Esto conserva el fallback local, pero usa S3 si existe `modelops/latest`.

## 4. Agrega resumen ModelOps en `tab1`

Dentro de `with tab1:`, después de las métricas o de la primera tabla, agrega:

```python
if USE_MODELOPS_S3:
    st.divider()
    render_modelops_summary_block()
```

## 5. Agrega evaluación ModelOps en `tab4`

Dentro de `with tab4:`, después de la evaluación local, agrega:

```python
if USE_MODELOPS_S3:
    st.divider()
    render_modelops_evaluation_block()
```

## 6. Agrega registry/debug en `tab5` o `tab6`

En `tab5` o `tab6`, al final, agrega:

```python
if USE_MODELOPS_S3:
    st.divider()
    render_modelops_registry_block()

    with st.expander("Debug ModelOps S3"):
        render_modelops_debug_block()
```

## 7. Variables que necesita ECS

La Task Definition debe tener:

```text
USE_MODELOPS_S3=true
MODEL_BUCKET=<bucket de pfs-modelops>
MODELOPS_PREFIX=modelops/latest
APP_EXPORT_BUCKET=<bucket de pfs-modelops>
APP_EXPORT_PREFIX=app/batch_exports
AWS_REGION=us-east-1
```

## 8. Qué no cambiar

No quites:

- inferencia individual con `model.joblib`;
- feedback en RDS;
- historial de `batch_exports`;
- logs que ya agregó tu compañera;
- conexión a Secrets Manager.
