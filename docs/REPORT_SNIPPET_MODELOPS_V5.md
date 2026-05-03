## Mejoras aplicadas en la UI y ModelOps v5

Esta iteración no reemplaza el diseño original de la aplicación. Se mantuvieron las pestañas y gráficas clave del MVP, y se aplicaron mejoras puntuales para hacer la interfaz más útil para negocio y más consistente con la selección de modelos.

### Cambios principales

- Se eliminaron columnas redundantes como `shop_label` e `item_label` de las tablas visibles y se dejaron `shop_id + shop_name` y `item_id + item_name`.
- El resumen ejecutivo ahora explica claramente cómo usar cada pestaña.
- Se añadieron dos gráficas de resumen: forecast total por categoría y forecast total por `model_scope`.
- La gráfica de distribución de pronósticos ahora puede verse en escala original y en `log(1 + pronóstico)` para mejorar la lectura de una distribución concentrada cerca de cero.
- La evaluación global ahora puede comparar champion, naive y segundo lugar en la gráfica de media predicha vs media real por rango de demanda.
- Los KPIs por categoría usan `item_category_id` en la gráfica de RMSE para evitar problemas de nombres faltantes o muy largos.
- El ranking de modelos usa RMSE como criterio principal y MAE como desempate.
- La pestaña de feedback agrega una tabla de 30 productos más sobreestimados.
- El model registry explica brevemente qué hace cada modelo.

### Política de selección de modelos

La comparación de modelos mantiene el diseño original del modelo incumbent de dos etapas y lo enfrenta a challengers adicionales. La política final usa:

1. **RMSE** como criterio principal.
2. **MAE** como desempate.
3. El incumbent solo es reemplazado si el challenger mejora bajo la política definida.
