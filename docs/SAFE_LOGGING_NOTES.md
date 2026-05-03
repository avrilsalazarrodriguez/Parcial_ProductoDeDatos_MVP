# Notas de logging seguro

La app ya tiene logs operacionales. Al conectar ModelOps, mantén estas reglas:

- Loggear acciones: `load_modelops_forecast`, `upload_batch_export`, `single_inference`, `feedback_insert`.
- Loggear resultados: `success`, `failure`, conteo de registros, duración si se mide.
- No loggear passwords, connection strings, tokens, secrets ni credenciales de AWS.
- No loggear archivos completos ni filas completas con información sensible.
- Usar `exc_info=True` en errores para que CloudWatch muestre stack trace.
- En producción, evitar DEBUG excesivo; usar INFO/WARNING/ERROR.

Ejemplo:

```python
logger.info(
    "action=load_modelops_forecast status=success rows=%s bucket=%s prefix=%s",
    len(df),
    bucket,
    prefix,
)
```
