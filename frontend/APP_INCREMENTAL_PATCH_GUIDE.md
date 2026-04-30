# Patch incremental para `frontend/app.py`

Este patch NO cambia el diseño general de la app. Mantiene las pestañas y gráficas originales, y solo agrega mejoras.

## 1. Imports nuevos

En la parte superior de `frontend/app.py`, junto a tus imports de frontend, agrega:

```python
from frontend.ui_enhancements import (
    attach_catalog_labels,
    friendly_table,
    render_dual_shop_item_selector,
    render_forecast_distribution,
    render_model_scope_help,
    render_prediction_vs_actual_by_demand,
    render_problem_product_suggestions,
    render_top_shop_forecast_chart,
)
```

## 2. Usar alpha=0.90 en el modelo original

Tu `src/training/train.py` evalúa el modelo original con:

```python
raw_pred = (prob_valid ** alpha) * mu_valid
alpha = 0.90
```

Por consistencia, cambia la función `predict_with_model` de la app a:

```python
def predict_with_model(model_payload: dict, features: pd.DataFrame) -> np.ndarray:
    bundle = model_payload["bundle"]
    feature_cols = bundle["feature_cols"]
    x_test = features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)
    alpha = model_payload.get("meta", {}).get("alpha", 0.90)
    return np.clip((prob ** alpha) * mu, 0, 20)
```

## 3. Enriquecer tablas con nombres reales

Después de cargar o construir `batch_df`, agrega:

```python
batch_df = attach_catalog_labels(batch_df)
```

Después de construir `eval_df`, agrega:

```python
eval_df = attach_catalog_labels(eval_df)
```

Si tienes `test_pairs`, crea una versión etiquetada:

```python
test_pairs_labeled = attach_catalog_labels(test_pairs)
```

## 4. Resumen ejecutivo: mantener tabla, pero con nombres reales

Cambia tablas como:

```python
st.dataframe(batch_df.head(100), width="stretch")
```

por:

```python
friendly_table(batch_df.head(100), max_rows=100)
```

## 5. Dashboard rápido: conservar gráfica de tiendas y mejorar distribución

En vez de agrupar por `shop_id` o mostrar `Tienda 2`, usa:

```python
c1, c2 = st.columns(2)
with c1:
    render_top_shop_forecast_chart(batch_df)
with c2:
    render_forecast_distribution(batch_df)
```

Esto conserva la idea original de las dos gráficas, pero agrega nombres reales y permite ver la distribución con escala `log1p`.

## 6. Inferencia individual: selector + escritura directa de IDs

En la pestaña de inferencia individual reemplaza el bloque de selectores por:

```python
selected_shop, selected_item = render_dual_shop_item_selector(
    test_pairs_labeled,
    key_prefix="single_inference",
)

if selected_shop is None or selected_item is None:
    st.warning("Selecciona o escribe una tienda y producto válidos.")
else:
    selected_rows = test_pairs.query(
        "shop_id == @selected_shop and item_id == @selected_item"
    )

    if selected_rows.empty:
        st.warning(
            "No se encontró ese par tienda-producto en el conjunto futuro. "
            "Puedes usar el selector o escribir otro shop_id/item_id."
        )
    else:
        selected_index = selected_rows.index[0]
        selected_features = test_features.iloc[[selected_index]]
        pred = predict_with_model(model_payload, selected_features)[0]

        metadata = attach_catalog_labels(
            pd.DataFrame([{"shop_id": selected_shop, "item_id": selected_item}])
        ).iloc[0]

        st.metric("Pronóstico próximo mes", f"{pred:.2f} unidades")
        st.write("**Tienda:**", metadata.get("shop_label"))
        st.write("**Producto:**", metadata.get("item_label"))

        with st.expander("Ver features usadas"):
            st.dataframe(selected_features, width="stretch")
```

## 7. Batch CFO: explicar `model_scope`

Justo debajo del título de Batch CFO agrega:

```python
render_model_scope_help()
```

Y para la tabla:

```python
friendly_table(filtered_batch, max_rows=1000)
```

## 8. Evaluación: conservar y enriquecer la gráfica clave

Después de tus métricas de RMSE/MAE, agrega o conserva:

```python
render_prediction_vs_actual_by_demand(eval_df)
```

Esto mantiene la gráfica de “Predicción promedio vs valor real por nivel de demanda” y agrega un diagnóstico explícito cuando el modelo subestima demanda alta.

## 9. KPIs: usar nombres reales en gráficas y tablas

Para top tiendas por error:

```python
by_shop = (
    eval_df.groupby(["shop_id", "shop_name", "shop_label"], as_index=False)
    .agg(
        n=("y", "size"),
        y_mean=("y", "mean"),
        pred_mean=("prediction", "mean"),
        mae=("abs_error", "mean"),
    )
    .sort_values("mae", ascending=False)
)
```

Para productos:

```python
by_item = (
    eval_df.groupby(
        ["item_id", "item_name", "item_label", "item_category_name"],
        as_index=False,
    )
    .agg(
        n=("y", "size"),
        y_mean=("y", "mean"),
        pred_mean=("prediction", "mean"),
        mae=("abs_error", "mean"),
    )
    .sort_values("mae", ascending=False)
)
```

## 10. Feedback: recuperar productos sugeridos para revisión

En la sección “Productos sugeridos para revisión”, usa:

```python
problem_products = safe_read_problem_products(limit=100)
render_problem_product_suggestions(eval_df, rds_df=problem_products)
```

Si RDS trae datos, los muestra. Si RDS está vacío, genera sugerencias desde los productos con mayor error.

## 11. JSON técnico

No uses `st.json(metrics)` directamente para bloques grandes. El archivo `frontend/modelops_sections.py` incluido en este overlay ya transforma esos JSON en tarjetas, tablas y expanders técnicos.
