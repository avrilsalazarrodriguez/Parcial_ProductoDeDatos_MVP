# Producto de datos para pronóstico de ventas en AWS

> MVP de producto de datos para consultar pronósticos mensuales de ventas de **1C Company**, generar archivos para CFO, evaluar modelos, revisar KPIs, capturar feedback operativo y consumir artefactos ModelOps publicados en AWS.

![Vista pública de la aplicación](graficasProyecto/apppublica.png)

---

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Problema de negocio](#problema-de-negocio)
3. [Arquitectura](#arquitectura)
4. [Funcionalidades principales](#funcionalidades-principales)
5. [Estrategia de modelado y ModelOps](#estrategia-de-modelado-y-modelops)
6. [Inputs y outputs](#inputs-y-outputs)
7. [Estructura del repositorio](#estructura-del-repositorio)
8. [Instalación local](#instalación-local)
9. [Ejecución local](#ejecución-local)
10. [Despliegue en AWS](#despliegue-en-aws)
11. [Operación y validación](#operación-y-validación)
12. [Seguridad y buenas prácticas](#seguridad-y-buenas-prácticas)
13. [Troubleshooting](#troubleshooting)
14. [Licencia](#licencia)

---

## Descripción general

Este proyecto transforma un flujo de machine learning originalmente trabajado en notebooks en una aplicación web de datos lista para ser usada por perfiles de negocio y equipos técnicos. La aplicación permite consultar pronósticos mensuales por tienda-producto, generar archivos batch para finanzas, evaluar modelos, revisar productos problemáticos y registrar observaciones operativas.

La solución final se implementa con **Streamlit**, se despliega como contenedor en **Amazon ECS Fargate**, guarda artefactos y salidas en **Amazon S3**, registra información operacional en **Amazon RDS PostgreSQL**, y usa un flujo **ModelOps** para separar el entrenamiento y evaluación del consumo interactivo.

---

## Problema de negocio

1C Company necesita consultar y operar pronósticos de ventas mensuales de forma accionable. El reto no es solo producir una predicción, sino convertirla en un producto de datos que pueda ser usado por diferentes stakeholders:

| Stakeholder | Necesidad | Solución en la app |
|---|---|---|
| CFO / Finanzas | Generar archivos descargables para tienda, categoría o catálogo completo | Vista **Batch CFO** con guardado en S3 |
| Planeación de inventarios | Identificar productos sobre/subestimados o de baja actividad | Vistas **KPIs** y **Feedback** |
| Data Science | Evaluar modelos y comparar contra naive | Vista **Evaluación** y **Model Registry** |
| Plataforma / DevOps | Ejecutar la app sin depender de una laptop | Despliegue en ECS Fargate + ALB |
| BI / Analítica | Consultar artefactos persistentes | S3 + Glue Catalog + salidas ModelOps |

---

## Arquitectura

La arquitectura separa tres responsabilidades:

1. **Capa de consumo:** Streamlit como interfaz para negocio y operación.
2. **Capa de persistencia:** S3 para artefactos/salidas y RDS para historial/feedback.
3. **Capa ModelOps:** entrenamiento, evaluación, registry y publicación de predicciones.

![Diagrama de arquitectura](graficasProyecto/infraestructura_a1.png)

### Componentes principales

| Componente | Uso |
|---|---|
| Application Load Balancer | Expone la aplicación desde una URL pública |
| ECS Fargate | Ejecuta el contenedor de Streamlit |
| ECR | Almacena la imagen Docker |
| S3 | Guarda ModelOps, predicciones, archivos CFO y batch uploads |
| RDS PostgreSQL | Guarda historial, feedback y eventos operativos |
| Secrets Manager | Administra credenciales de RDS |
| CloudWatch Logs | Permite depurar tareas ECS y errores de la app |
| Glue Data Catalog | Capa de metadatos para consumo analítico sobre S3 |

---

## Funcionalidades principales

### 1. Resumen ejecutivo

Muestra volumen de predicciones, modelo en uso, distribución de pronósticos, principales tiendas/categorías y cobertura por `model_scope`.

![Resumen ejecutivo](graficasProyecto/apppublica.png)

### 2. Inferencia individual

Permite consultar un par tienda-producto usando ID o nombre. La app muestra la predicción y contexto relevante del producto.

![Inferencia individual](graficasProyecto/inferenciaindividual.png)

### 3. Batch CFO

Permite generar archivos para:

- todos los productos de una tienda;
- un segmento/categoría;
- catálogo completo.

Los archivos CFO se guardan en S3 con particiones por alcance:

```text
app/batch_exports/todos_los_productos_de_una_tienda/shop_<id>/
app/batch_exports/segmento_categoria/category_<id>/
app/batch_exports/catalogo_completo/all/
```

![Batch CFO](graficasProyecto/batchcfo.png)

![Batch CFO con historial](graficasProyecto/batchcfo2.png)

### 4. Batch por archivo cargado

Permite subir un CSV estilo test/validación con columnas como:

```text
ID, shop_id, item_id
```

o un archivo con features completas. Si los pares existen en las features preparadas, la app completa variables desde `data/prep/test_features.parquet`. Si no existen, se tratan como casos cold-start.

Las predicciones por archivo cargado se guardan sin particionar en:

```text
app/batch_uploads/predictions/
app/batch_uploads/history/uploaded_predictions_history.csv
```

![Batch por archivo cargado](graficasProyecto/batchcfo3.png)

![Historial de batch uploads](graficasProyecto/batch4.png)

### 5. Evaluación

Compara modelos contra ground truth. Incluye curva real vs modelos seleccionados, distribución por rangos de demanda real y tablas de performance.

![Evaluación](graficasProyecto/evaluacionseccion.png)

![Curvas por modelo](graficasProyecto/evaluacion1_1.png)

![Distribución de demanda real](graficasProyecto/distribuciondemanda.png)

### 6. KPIs

Incluye RMSE por categoría, producto y tienda; además muestra productos con baja actividad reciente.

![KPIs por RMSE](graficasProyecto/kpis1.png)

![KPIs y baja actividad](graficasProyecto/kpis2.png)

### 7. Feedback

Permite capturar observaciones de negocio y revisar productos sobreestimados, subestimados o sugeridos para revisión.

![Feedback](graficasProyecto/feedback1.png)

![Productos sugeridos para revisión](graficasProyecto/feedback3.png)

### 8. Model Registry

Muestra el modelo champion, métricas globales, historial de modelos, baselines y descripciones.

![Model Registry](graficasProyecto/modelregistry.png)

---

## Estrategia de modelado y ModelOps

El proyecto conserva modelos y baselines porque el problema tiene demanda intermitente y muchos ceros. La evaluación mostró que un modelo global puede funcionar bien para demanda baja, mientras que un naive o especialista puede ser competitivo en rangos con señal reciente.

Modelos/candidatos principales:

| Modelo | Descripción |
|---|---|
| Hurdle HGB | Modelo de dos etapas para probabilidad de venta y unidades |
| LightGBM original dos etapas | Modelo incumbent con transformaciones originales |
| Naive lag 1 | Baseline que usa la venta del último periodo |
| Rolling mean | Baseline de media móvil reciente |
| HGB Poisson | Modelo para conteos no negativos |
| Especialista recurrente | Modelo para productos con señales de demanda recurrente |
| Hybrid Router | Política que enruta cada producto-tienda a regla/modelo según señales históricas |

La app consume artefactos publicados en S3 bajo `modelops/latest/`, evitando recalcular entrenamiento o scoring pesado dentro de Streamlit.

---

## Inputs y outputs

### Inputs principales

| Input | Descripción |
|---|---|
| Datos históricos de ventas | Base para features, entrenamiento y validación |
| `data/prep/test_features.parquet` | Features preparadas para inferencia batch |
| `data/prep/test_pairs.parquet` | Pares tienda-producto de referencia |
| CSV cargado por usuario | Archivo para inferencia ad hoc |
| Artefactos ModelOps en S3 | Predicciones, métricas, curvas, registry y sugerencias |

### Outputs principales

| Output | Ruta / uso |
|---|---|
| Forecast vigente | `modelops/latest/predictions/forecast_detail.parquet` |
| Métricas de modelos | `modelops/latest/evaluation/model_metrics.json` |
| Curvas de evaluación | `modelops/latest/evaluation/evaluation_curves_by_model.parquet` |
| Model Registry | `modelops/registry/model_runs.csv` |
| Archivos CFO | `app/batch_exports/...` |
| Predicciones por archivo cargado | `app/batch_uploads/predictions/` |
| Historial batch upload | `app/batch_uploads/history/uploaded_predictions_history.csv` |
| Feedback | RDS PostgreSQL |

---

## Estructura del repositorio

```text
.
├── backend/                 # Conexiones a S3, RDS, almacenamiento y helpers backend
├── config/                  # Archivos .example.env y plantillas de configuración
├── data/examples/           # CSVs pequeños de ejemplo para inferencia batch
├── docs/                    # Documentación auxiliar y snippets de reporte
├── frontend/                # Aplicación Streamlit y componentes UI
├── graficasProyecto/        # Imágenes usadas en README y reporte
├── infra/                   # Plantillas/recursos de infraestructura
├── scripts/                 # Entrenamiento, validación, despliegue, permisos y utilidades
├── src/                     # Lógica ModelOps, métricas y entrenamiento
├── sql/                     # SQL para RDS/tablas operativas
├── tests/                   # Pruebas de import, métricas y outputs ModelOps
├── Dockerfile               # Imagen de la app
├── pyproject.toml           # Dependencias/configuración Python si aplica
├── uv.lock                  # Lockfile de uv si aplica
└── README.md                # Documentación principal del proyecto
```

No se deben versionar outputs generados localmente como:

```text
modelops_outputs/
modelops_outputs_hybrid/
config/*.env
frontend/*.before_*
backend/*.before_*
.venv/
```

---

## Instalación local

### Requisitos

- Python 3.12
- uv
- Docker
- AWS CLI configurado
- Cuenta AWS con permisos para S3, ECS, ECR, RDS, CloudFormation e IAM

### Crear entorno

```bash
uv sync
```

Si el proyecto usa extras o grupos específicos:

```bash
uv sync --all-extras
```

### Variables de entorno

Usa archivos `.example.env` como referencia. No subas `.env` ni `config/generated.env` a GitHub.

```bash
cp config/modelops_v5.example.env config/modelops_v5.env
```

Para correr local:

```bash
export USE_MODELOPS_S3=false
export MODELOPS_LOCAL_ROOT=modelops_outputs_hybrid
export DATA_DIR=data
export DISABLE_RDS_WRITES=true
```

---

## Ejecución local

```bash
PYTHONPATH=. uv run streamlit run frontend/app.py
```

Validar sintaxis de la app:

```bash
PYTHONPATH=. uv run python scripts/validate_app_syntax.py
```

Validar outputs ModelOps:

```bash
PYTHONPATH=. uv run python scripts/32_validate_modelops_outputs.py
```

---

## Despliegue en AWS

### Construir y subir imagen

```bash
source config/generated.env
./scripts/06_build_push_app.sh
```

### Actualizar ECS

```bash
aws ecs update-service \
  --cluster pfs-mvp-cluster \
  --service pfs-mvp \
  --force-new-deployment \
  --region "$AWS_REGION"
```

### Verificar despliegue

```bash
aws ecs describe-services \
  --cluster pfs-mvp-cluster \
  --services pfs-mvp \
  --region "$AWS_REGION" \
  --query "services[0].{running:runningCount,desired:desiredCount,rollout:deployments[0].rolloutState}" \
  --output table
```

### Logs

```bash
aws logs tail /ecs/pfs-mvp \
  --since 20m \
  --region "$AWS_REGION"
```

---

## Operación y validación

### Entrenar / comparar modelos

```bash
PYTHONPATH=. uv run python scripts/40_train_hybrid_router.py
```

### Subir outputs ModelOps a S3

```bash
PYTHONPATH=. uv run python scripts/31_upload_modelops_outputs_s3.py \
  --bucket "$MODEL_BUCKET" \
  --local-root modelops_outputs_hybrid \
  --latest-prefix modelops/latest \
  --registry-prefix modelops/registry
```

### Probar batch upload

```bash
PYTHONPATH=. uv run python scripts/85_generate_batch_upload_examples.py \
  --model-path artifacts/model.joblib \
  --test-features-path data/prep/test_features.parquet \
  --test-pairs-path data/prep/test_pairs.parquet \
  --output-dir data/examples/batch_upload
```

### Permisos S3 para batch uploads

```bash
PYTHONPATH=. uv run python scripts/97_grant_task_role_s3_batch_upload_permissions.py \
  --role-name pfs-mvp-task-role \
  --bucket "$MODEL_BUCKET" \
  --region "$AWS_REGION"
```

---

## Seguridad y buenas prácticas

- No subir credenciales ni `.env`.
- Usar Secrets Manager para credenciales de RDS.
- Usar IAM task role para permisos de ECS.
- Mantener S3 como capa persistente para artefactos.
- No guardar outputs grandes en Git.
- Mantener README y documentación actualizados.
- Separar entrenamiento/scoring pesado de la app Streamlit.

---

## Troubleshooting

### `502 Connection failed`

1. Revisar logs:

```bash
aws logs tail /ecs/pfs-mvp --since 20m --region "$AWS_REGION"
```

2. Aumentar recursos Fargate si hay reinicios/OOM:

```bash
PYTHONPATH=. uv run python scripts/83_update_ecs_fargate_resources.py \
  --cluster pfs-mvp-cluster \
  --service pfs-mvp \
  --cpu 2048 \
  --memory 4096 \
  --region "$AWS_REGION"
```

3. Aumentar timeout del ALB:

```bash
PYTHONPATH=. uv run python scripts/84_update_alb_timeout.py \
  --name-contains pfs-mvp \
  --idle-timeout 120 \
  --region "$AWS_REGION"
```

### `AccessDenied` al guardar batch uploads

Dar permisos al task role:

```bash
PYTHONPATH=. uv run python scripts/97_grant_task_role_s3_batch_upload_permissions.py \
  --role-name pfs-mvp-task-role \
  --bucket "$MODEL_BUCKET" \
  --region "$AWS_REGION"
```

### No aparecen curvas de algún modelo

Verifica que existan predicciones por modelo en:

```text
modelops/latest/evaluation/evaluation_curves_by_model.parquet
```

Si el modelo existe en registry pero no tiene curva, no se puede graficar honestamente hasta publicar sus predicciones por bucket.

---

## Licencia

Proyecto académico desarrollado para el curso **Arquitectura de Productos de Datos y Métodos de Gran Escala**. Uso educativo.

---

## Autores

- Avril Salazar Rodríguez
- Héctor Vilchis Peralta

Instituto Tecnológico Autónomo de México (ITAM), mayo 2026.
