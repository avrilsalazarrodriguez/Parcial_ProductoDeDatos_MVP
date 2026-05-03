# Patch de app para nombres amigables y catálogo de modelos

Hay dos formas de aplicar este overlay.

## Opción A — reemplazo completo recomendado

Copia la app nueva:

```bash
cp frontend/app_friendly_model_zoo.py frontend/app.py
```

La app nueva conserva las vistas principales y agrega:

- filtros `id — nombre` para tienda y producto;
- columnas `shop_name`, `item_name`, `item_category_name`, `category_group`;
- limpieza de valores `None`, `unknown` y `recency=99`;
- catálogo de modelos entrenados;
- debug de ModelOps S3.

## Opción B — patch manual

Si prefieres no reemplazar `frontend/app.py`, importa:

```python
from frontend.friendly_display import (
    ensure_friendly_columns,
    first_int_from_label,
    friendly_forecast_columns,
    make_options,
    render_forecast_filters,
    render_table,
)
from frontend.model_zoo_catalog_section import render_model_zoo_catalog
```

Después, cuando construyas `batch_df`, aplica:

```python
batch_df = ensure_friendly_columns(batch_df)
```

Cambia los selectbox de tienda:

```python
selected_shop_label = st.selectbox(
    "Tienda",
    options=make_options(batch_df, "shop_label", include_all=False),
)
selected_shop = first_int_from_label(selected_shop_label)
```

Cambia las tablas de forecast:

```python
render_table(filtered_batch, max_rows=1000)
```

Agrega el catálogo:

```python
with st.expander("Catálogo de modelos entrenados"):
    render_model_zoo_catalog()
```
