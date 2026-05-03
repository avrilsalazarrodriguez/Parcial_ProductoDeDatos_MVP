# Producto de Datos — Pronóstico de Ventas 1C Company en AWS

> MVP de producto de datos para consultar pronósticos mensuales de ventas, generar archivos para negocio/CFO, evaluar modelos, registrar feedback operativo y mantener trazabilidad de ModelOps en AWS.

<p align="center">
  <img src="graficasProyecto/apppublica.png" alt="Aplicación Streamlit pública" width="850">
</p>

---

## Tabla de contenidos

1. [Contexto de negocio](#contexto-de-negocio)
2. [Qué hace el producto](#qué-hace-el-producto)
3. [Arquitectura general](#arquitectura-general)
4. [Modelo de datos y entidad-relación](#modelo-de-datos-y-entidad-relación)
5. [Tecnologías utilizadas](#tecnologías-utilizadas)
6. [Vistas de la aplicación](#vistas-de-la-aplicación)
7. [ModelOps y política híbrida](#modelops-y-política-híbrida)
8. [Inputs y outputs](#inputs-y-outputs)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Instalación local](#instalación-local)
11. [Ejecución local](#ejecución-local)
12. [Despliegue en AWS](#despliegue-en-aws)
13. [Operación, estabilidad y seguridad](#operación-estabilidad-y-seguridad)
14. [Documentación del reporte](#documentación-del-reporte)
15. [Limitaciones y siguientes pasos](#limitaciones-y-siguientes-pasos)
16. [Autores](#autores)

---

## Contexto de negocio

El proyecto parte del problema de pronosticar ventas mensuales de 1C Company a nivel **tienda-producto**. El objetivo no es ganar la competencia original de Kaggle, sino convertir un flujo de machine learning en un **producto de datos operable** por usuarios de negocio.

El MVP responde a necesidades de varios stakeholders:

| Stakeholder | Necesidad | Respuesta del MVP |
|---|---|---|
| Planeación de demanda | Consultar pronósticos por tienda, producto o categoría. | App Streamlit pública con filtros, KPIs y vistas de evaluación. |
| Finanzas / CFO | Descargar archivos batch de forecast. | Vista **Batch CFO** con descarga CSV, guardado en S3 e historial. |
| Applied Scientist | Comparar modelos y revisar errores. | Model Registry, curvas de evaluación, RMSE/MAE y comparación contra naive. |
| Operaciones / Inventarios | Detectar productos con baja actividad o riesgo de sub/sobreestimación. | KPIs, Feedback y tablas de productos problemáticos. |
| Plataforma / CTO | Ejecutar una app estable y reproducible en nube. | Docker, ECR, ECS Fargate, ALB, CloudFormation, RDS, S3 y CloudWatch. |
| Seguridad | No guardar credenciales en código. | Secrets Manager, variables de entorno y roles IAM con permisos por prefijo. |

---

## Qué hace el producto

La aplicación permite:

- Consultar forecast mensual por tienda y producto.
- Generar archivos CFO para:
  - todos los productos de una tienda;
  - una categoría/segmento específico;
  - el catálogo completo.
- Guardar archivos CFO en S3 con estructura clara por alcance.
- Subir un CSV estilo test o con features completas para generar inferencia batch ad hoc.
- Guardar predicciones cargadas en S3 bajo una carpeta separada de los archivos CFO.
- Evaluar modelos contra ground truth histórico.
- Comparar modelos contra naive, promedio móvil, LightGBM original, Hurdle HGB, Poisson y router híbrido.
- Registrar feedback de negocio para revisión posterior.
- Consultar un Model Registry ligero con métricas, descripciones y modelo champion.

---

## Arquitectura general

La arquitectura separa tres capas:

1. **Capa de consumo**: app Streamlit desplegada en ECS Fargate y expuesta mediante Application Load Balancer.
2. **Capa de persistencia**: S3 para archivos y artefactos; RDS PostgreSQL para historial, eventos y feedback.
3. **Capa de ModelOps**: artefactos, predicciones, métricas, curvas de evaluación, model registry y modelo champion.

<p align="center">
  <img src="graficasProyecto/infraestructura_a1.png" alt="Arquitectura AWS del MVP" width="900">
</p>

**Lectura del diagrama.** El usuario entra por el ALB hacia la app en ECS Fargate. La app consume artefactos vigentes desde S3, registra información operacional en RDS, obtiene secretos desde Secrets Manager y deja trazabilidad en CloudWatch Logs. Glue Data Catalog funciona como capa de metadatos sobre los archivos analíticos en S3.

---

## Modelo de datos y entidad-relación

El modelo de datos principal vive en S3 y Glue Data Catalog. RDS se usa como base operacional, no como data lake completo.

<p align="center">
  <img src="graficasProyecto/ER1transpa.png" alt="Diagrama entidad-relación del producto de datos" width="900">
</p>

**Lectura del diagrama.** La tabla `features` alimenta el scoring. De ahí se construye `forecast_detail`, que guarda la predicción final y su trazabilidad. Cuando existe valor real, `evaluation_detail` permite calcular errores. A partir de esa evaluación se generan vistas agregadas por producto, categoría y modelo.

Tablas/artefactos principales:

| Artefacto | Propósito | Uso en la app |
|---|---|---|
| `features` | Variables explicativas: lags, medias móviles, recencia, frecuencia, precio y señales acumuladas. | Inferencia individual, batch cargado y scoring. |
| `forecast_detail` | Predicción final por tienda-producto con `model_scope`, `routing_reason` y candidatos. | Resumen, Batch CFO, KPIs. |
| `evaluation_detail` | Comparación fila a fila contra `y` real. | Evaluación, Feedback, análisis de errores. |
| `evaluation_by_item` | Métricas por producto. | Evaluación y KPIs. |
| `evaluation_by_segment` | Métricas por categoría/segmento. | KPIs y lectura por grupo. |
| `model_runs.csv` | Corridas evaluadas y métricas. | Model Registry. |
| `champion.json` | Modelo vigente y política de selección. | Resumen y Model Registry. |
| `review_suggestions.parquet` | Productos sugeridos para revisión. | Feedback. |

---

## Tecnologías utilizadas

| Capa | Tecnología | Uso |
|---|---|---|
| UI | Streamlit | Aplicación web de consulta, forecast, evaluación, KPIs y feedback. |
| Contenedores | Docker | Empaquetado de la aplicación. |
| Registry | Amazon ECR | Almacenamiento de la imagen Docker. |
| Cómputo app | Amazon ECS Fargate | Ejecución serverless del contenedor. |
| Exposición pública | Application Load Balancer | URL pública y balanceo hacia ECS. |
| Persistencia analítica | Amazon S3 | ModelOps, predicciones, CFO exports, batch uploads, métricas. |
| Base operacional | Amazon RDS PostgreSQL | Feedback, historial, eventos, metadata operacional. |
| Credenciales | AWS Secrets Manager | Secreto de conexión a RDS. |
| Logs | Amazon CloudWatch Logs | Depuración, reinicios, permisos y errores. |
| Catálogo | AWS Glue Data Catalog | Metadatos de tablas analíticas sobre S3. |
| Infraestructura | AWS CloudFormation | Despliegue reproducible de recursos. |
| Python env | uv | Administración rápida de dependencias y ejecución. |
| ML | scikit-learn / HGB / LightGBM | Modelos candidatos, baselines y política híbrida. |
| Data | pandas / pyarrow | Manipulación y lectura/escritura de CSV/Parquet. |
| AWS SDK | boto3 | Escritura/lectura en S3 y operación AWS desde Python. |

### Evidencias visuales de tecnología

<p align="center">
  <img src="graficasProyecto/ecr.png" alt="Repositorio ECR" width="800">
</p>

<p align="center">
  <img src="graficasProyecto/ecs.png" alt="Servicio ECS Fargate" width="800">
</p>

<p align="center">
  <img src="graficasProyecto/s3.png" alt="Bucket S3 con outputs" width="800">
</p>

<p align="center">
  <img src="graficasProyecto/rds.png" alt="RDS PostgreSQL" width="800">
</p>

<p align="center">
  <img src="graficasProyecto/glue.png" alt="Glue Data Catalog" width="800">
</p>

<p align="center">
  <img src="graficasProyecto/cloudwatch.png" alt="CloudWatch Logs" width="800">
</p>

> Si los nombres reales de tus capturas son distintos, conserva las imágenes en `graficasProyecto/` y ajusta únicamente el nombre del archivo en el README.

---

## Vistas de la aplicación

### 1. Resumen

Muestra volumen de predicciones, modelo vigente, métricas principales, forecast disponible, distribución de predicciones, categorías con mayor pronóstico y cobertura por `model_scope`.

<p align="center">
  <img src="graficasProyecto/resumen.png" alt="Vista Resumen" width="850">
</p>

### 2. Inferencia individual

Permite seleccionar o escribir `shop_id` e `item_id`, consultar la predicción puntual y revisar el contexto del par tienda-producto.

<p align="center">
  <img src="graficasProyecto/inferenciaindividual.png" alt="Inferencia individual" width="850">
</p>

### 3. Batch CFO

Genera archivos para negocio por tienda, categoría o catálogo completo. Los archivos CFO se guardan bajo `app/batch_exports/` y se particionan por tienda o categoría.

<p align="center">
  <img src="graficasProyecto/batchcfo.png" alt="Batch CFO" width="850">
</p>

### 4. Batch por archivo cargado

Permite subir un CSV con features completas o estilo test (`ID`, `shop_id`, `item_id`). Si el par existe en las features preparadas, la app completa columnas; si no existe, lo trata como cold-start. Estas predicciones ad hoc se guardan en `app/batch_uploads/predictions/`.

<p align="center">
  <img src="graficasProyecto/batchcfo_archivo_cargado.png" alt="Batch por archivo cargado" width="850">
</p>

### 5. Evaluación

Compara modelos contra ground truth usando RMSE, MAE, curvas por rango de demanda real y tablas de performance por segmento/producto.

<p align="center">
  <img src="graficasProyecto/evaluacionseccion.png" alt="Evaluación" width="850">
</p>

### 6. KPIs

Muestra categorías, productos y tiendas con mayor RMSE. También incluye productos con baja actividad reciente, top tiendas con más productos de baja actividad y top categorías afectadas.

<p align="center">
  <img src="graficasProyecto/kpis2.png" alt="KPIs" width="850">
</p>

### 7. Feedback

Permite registrar observaciones de negocio y revisar productos sobreestimados, subestimados o sugeridos para revisión.

<p align="center">
  <img src="graficasProyecto/feedback1.png" alt="Feedback" width="850">
</p>

### 8. Model Registry

Muestra el modelo champion, métricas y corridas evaluadas. Conserva el LightGBM original como incumbent, Hurdle HGB, HGB Poisson, especialista recurrente, rolling mean, promedio histórico y naive.

<p align="center">
  <img src="graficasProyecto/modelregistry.png" alt="Model Registry" width="850">
</p>

---

## ModelOps y política híbrida

La demanda tiene muchos ceros y distintos regímenes. Por eso el MVP no usa un único modelo para todos los casos. Se conserva un Model Registry con modelos y baselines, y se usa una política híbrida que registra su decisión por fila.

| Ruta | Interpretación |
|---|---|
| `inactive:no_recent_sales` | Producto-tienda con baja o nula actividad reciente. Predicción conservadora. |
| `baseline:naive_recent_demand` | Existe venta reciente; la señal del último periodo puede ser informativa. |
| `specialist:recurrent_demand` | Señales históricas sugieren demanda recurrente. |
| `challenger:hurdle_hgb` | Caso de demanda baja/intermitente donde se usa Hurdle HGB. |

Columnas de trazabilidad:

| Columna | Significado |
|---|---|
| `model_id` | Modelo o familia de modelo registrada. |
| `model_scope` | Ruta o política que generó la predicción final. |
| `routing_reason` | Regla histórica que activó esa ruta. |
| `decision_recommendation` | Explicación amigable para negocio. |

---

## Inputs y outputs

### Inputs principales

- Datos históricos de ventas de 1C Company.
- Features preparadas en `data/prep/` o en artefactos publicados en S3.
- Modelo serializado en `artifacts/model.joblib` cuando se ejecuta inferencia local dentro del contenedor.
- Artefactos ModelOps publicados en S3.
- CSV cargado por usuario en Batch CFO.

### Outputs principales en S3

```text
app/batch_exports/todos_los_productos_de_una_tienda/shop_<id>/
app/batch_exports/segmento_categoria/category_<id>/
app/batch_exports/catalogo_completo/all/
app/batch_uploads/predictions/
app/batch_uploads/history/uploaded_predictions_history.csv
modelops/latest/
modelops/registry/
```

### Outputs operacionales en RDS

- Historial de archivos generados.
- Feedback del negocio.
- Eventos de uso.
- Metadata operacional del producto.

---

## Estructura del repositorio

```text
.
├── README.md
├── Dockerfile
├── pyproject.toml / uv.lock
├── frontend/
│   ├── app.py
│   ├── batch_upload_inference.py
│   ├── cfo_s3_exports.py
│   └── low_activity_kpis.py
├── backend/
│   ├── modelops_s3.py
│   └── storage.py
├── src/
│   ├── modelops_v2/
│   ├── modelops_v3/
│   └── modelops_hybrid/
├── scripts/
│   ├── 06_build_push_app.sh
│   ├── 07_deploy_ecs.sh
│   ├── 30_train_compare_models_local.sh
│   ├── 31_upload_modelops_outputs_s3.py
│   ├── 40_train_hybrid_router.py
│   └── ...
├── infra/
│   └── cloudformation templates
├── sql/
├── tests/
├── docs/
│   └── Examen_ProductodeDatos_Avril_Hector.pdf
├── graficasProyecto/
│   ├── infraestructura_a1.png
│   ├── diagrama_entidad_relacion.png
│   └── capturas de la app/AWS
└── data/examples/
    └── batch_upload/
```

---

## Instalación local

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPO>
cd Parcial_ProductoDeDatos_MVP
```

### 2. Instalar dependencias

```bash
uv sync
```

Si no tienes `uv`, instálalo primero siguiendo la documentación oficial de Astral.

### 3. Configurar variables de entorno

Usa archivos `.example.env` como base. No subas archivos `.env` reales al repositorio.

```bash
cp config/modelops_v4_3.example.env config/local.env
```

Variables típicas:

```text
AWS_REGION=us-east-1
MODEL_BUCKET=pfs-modelops-...
USE_MODELOPS_S3=false
MODELOPS_LOCAL_ROOT=modelops_outputs_hybrid
DATA_DIR=data
DISABLE_RDS_WRITES=true
```

---

## Ejecución local

```bash
export USE_MODELOPS_S3=false
export MODELOPS_LOCAL_ROOT=modelops_outputs_hybrid
export DATA_DIR=data
export DISABLE_RDS_WRITES=true

PYTHONPATH=. uv run streamlit run frontend/app.py
```

Validar sintaxis:

```bash
PYTHONPATH=. uv run python scripts/validate_app_syntax.py
```

---

## Despliegue en AWS

### 1. Construir y subir imagen a ECR

```bash
source config/generated.env
./scripts/06_build_push_app.sh
```

### 2. Desplegar ECS/Fargate + ALB

```bash
source config/generated.env
./scripts/07_deploy_ecs.sh
```

### 3. Obtener URL pública

```bash
./scripts/09_print_outputs.sh
```

### 4. Validar app

```bash
source config/generated.env
./scripts/08_smoke_test_cloud.sh
```

### 5. Actualizar permisos S3 para predicciones cargadas

```bash
PYTHONPATH=. uv run python scripts/97_grant_task_role_s3_batch_upload_permissions.py \
  --role-name pfs-mvp-task-role \
  --bucket "$MODEL_BUCKET" \
  --region "$AWS_REGION"
```

---

## Operación, estabilidad y seguridad

Durante las pruebas se ajustó la app para renderizar una vista a la vez, reduciendo carga de navegación. También se aumentaron recursos de ECS Fargate y se ajustaron permisos S3 por prefijo.

Buenas prácticas aplicadas:

- No guardar credenciales en código.
- Secrets Manager para RDS.
- Task role con permisos específicos sobre S3.
- CloudWatch Logs para depuración.
- Separación entre `app/batch_exports/` y `app/batch_uploads/`.
- `.gitignore` para excluir `.env`, outputs locales, backups y artefactos pesados.

---

## Documentación del reporte

El reporte metodológico completo debe guardarse en:

```text
docs/Examen_ProductodeDatos_Avril_Hector.pdf
```

Este PDF documenta contexto de negocio, arquitectura, modelo de datos, evaluación, tour de la aplicación, evidencias AWS, operación, costos y cobertura de la rúbrica.

---

## Limitaciones y siguientes pasos

- Agregar autenticación y autorización por rol.
- Automatizar ModelOps con jobs programados o pipelines externos.
- Monitorear drift y desempeño cuando llegue el valor real futuro.
- Evaluar SageMaker Batch Transform si crece el volumen de scoring.
- Enriquecer cold-start con catálogo de negocio, lanzamientos y atributos externos.
- Agregar notificaciones cuando un archivo CFO o batch cargado quede listo.

---

## Autores

- Avril Salazar Rodríguez
- Héctor Vilchis Peralta

Curso: **Arquitectura de Productos de Datos y Métodos de Gran Escala**  
Institución: **ITAM**  
Fecha: **Mayo 2026**
