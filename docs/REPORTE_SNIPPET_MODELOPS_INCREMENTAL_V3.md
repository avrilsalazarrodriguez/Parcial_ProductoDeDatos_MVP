# Snippet para el reporte: mejora incremental de modelos y UI

La app original se mantuvo como base visual y funcional. En lugar de reemplazarla, se agregó una capa incremental para mejorar trazabilidad y lectura de negocio:

- Los pronósticos y métricas se enriquecen con nombres reales de tienda, producto y categoría.
- Los valores nulos, `unknown/noce` y `recency=99` se presentan como información de negocio comprensible.
- La gráfica de predicción promedio vs valor real por rango de demanda se mantiene y se acompaña de diagnóstico explícito.
- La distribución de pronósticos permite alternar entre escala original y `log(1 + pronóstico)`, útil porque muchas predicciones están cerca de cero.
- El modelo original LightGBM de dos etapas se conserva como candidato incumbent dentro del model registry y se compara contra modelos challenger. El champion se selecciona por RMSE, no por WAPE.
- Los JSON técnicos se muestran en expanders y las métricas se presentan como tarjetas/tablas para evitar ruido visual.

Esta decisión mantiene el valor del diseño original y agrega elementos de MLOps: comparación de modelos, trazabilidad, evaluación por segmento/producto y sugerencias para revisión del equipo de ML.
