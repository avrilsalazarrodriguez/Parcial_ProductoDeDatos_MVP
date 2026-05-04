# Producto de Datos — Pronóstico de Ventas 1C Company en AWS

> MVP de producto de datos para consultar pronósticos mensuales de ventas, generar archivos para negocio/CFO, evaluar modelos, registrar feedback operativo y mantener trazabilidad de ModelOps en AWS.


## Aplicación pública

La aplicación desplegada en AWS se puede consultar en:

[http://pfs-mvp-alb-2038135488.us-east-1.elb.amazonaws.com](http://pfs-mvp-alb-2038135488.us-east-1.elb.amazonaws.com)


<p align="center">
  <img src="graficasProyecto/apppublica.png" alt="Aplicación pública Streamlit en AWS" width="900">
</p>

---

## Tabla de contenidos

1. [Contexto de negocio](#contexto-de-negocio)
2. [Qué hace el producto](#qué-hace-el-producto)
3. [Arquitectura general](#arquitectura-general)
4. [Modelo de datos y entidad-relación](#modelo-de-datos-y-entidad-relación)
5. [Tecnologías utilizadas y evidencias AWS](#tecnologías-utilizadas-y-evidencias-aws)
6. [Vistas de la aplicación](#vistas-de-la-aplicación)
7. [ModelOps y política híbrida](#modelops-y-política-híbrida)
8. [Inputs y outputs](#inputs-y-outputs)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Instalación local](#instalación-local)
11. [Ejecución local](#ejecución-local)
12. [Despliegue en AWS](#despliegue-en-aws)
13. [Operación, estabilidad y seguridad](#operación-estabilidad-y-seguridad)
14. [Documentación metodológica](#documentación-metodológica)
15. [Limitaciones y siguientes pasos](#limitaciones-y-siguientes-pasos)
16. [Bibliografía y recursos consultados](#bibliografía-y-recursos-consultados)
17. [Autores](#autores)

---

## Contexto de negocio

El proyecto parte del problema de pronosticar ventas mensuales de **1C Company** a nivel **tienda-producto**. La meta no fue ganar la competencia original de Kaggle, sino convertir un flujo de machine learning en un **producto de datos operable** por usuarios de negocio.

El MVP permite que perfiles de negocio consulten predicciones, descarguen archivos para planeación financiera, revisen desempeño del modelo y registren observaciones sin abrir notebooks ni ejecutar código localmente.

| Stakeholder | Necesidad | Respuesta del MVP |
|---|---|---|
| Planeación de demanda | Consultar pronósticos por tienda, producto o categoría. | App Streamlit pública con filtros, KPIs y vistas de evaluación. |
| Finanzas / CFO | Descargar archivos batch de forecast. | Vista **Batch CFO** con descarga CSV, guardado en S3 e historial. |
| Applied Scientist | Comparar modelos y revisar errores. | Model Registry, curvas de evaluación, RMSE/MAE y comparación contra naive. |
| Operaciones / Inventarios | Detectar productos con baja actividad o riesgo de sub/sobreestimación. | KPIs, Feedback y tablas de productos problemáticos. |
| Plataforma / CTO | Ejecutar una app estable y reproducible en nube. | Docker, Amazon ECR, Amazon ECS Fargate, ALB, CloudFormation, Amazon RDS, Amazon S3 y CloudWatch. |
| Seguridad | Evitar credenciales en código. | AWS Secrets Manager, variables de entorno y roles IAM con permisos por prefijo. |

---

## Qué hace el producto

La aplicación permite:

- Consultar forecast mensual por tienda y producto.
- Generar archivos CFO para:
  - todos los productos de una tienda;
  - una categoría/segmento específico;
  - el catálogo completo.
- Guardar archivos CFO en Amazon S3 con estructura por alcance.
- Subir un CSV estilo test (`ID`, `shop_id`, `item_id`) o con features completas para generar inferencia batch ad hoc.
- Guardar predicciones cargadas en Amazon S3 bajo una carpeta separada de los archivos CFO.
- Evaluar modelos contra ground truth histórico.
- Comparar modelos contra naive, promedio móvil, LightGBM original, Hurdle HGB, Poisson y router híbrido.
- Registrar feedback de negocio para revisión posterior.
- Consultar un Model Registry ligero con métricas, descripciones y modelo champion.

---

## Arquitectura general

La arquitectura separa tres responsabilidades:

1. **Capa de consumo:** aplicación Streamlit desplegada en Amazon ECS Fargate y expuesta mediante Application Load Balancer.
2. **Capa de persistencia:** Amazon S3 para archivos/artefactos y Amazon RDS PostgreSQL para historial, eventos y feedback.
3. **Capa de ModelOps:** artefactos, predicciones, métricas, curvas de evaluación, model registry y modelo champion.

<p align="center">
  <img src="graficasProyecto/infraestructura_a1.png" alt="Diagrama de arquitectura general del MVP en AWS" width="950">
</p>

**Lectura del diagrama de infraestructura.** El usuario entra por el **Application Load Balancer**, que enruta tráfico hacia la app en **Amazon ECS Fargate**. La imagen Docker vive en **Amazon ECR**. La app consume artefactos vigentes desde **Amazon S3**, registra información operacional en **Amazon RDS PostgreSQL**, obtiene credenciales desde **AWS Secrets Manager** y deja trazabilidad en **Amazon CloudWatch Logs**. **AWS Glue Data Catalog** funciona como capa de metadatos sobre los archivos analíticos en S3.

La infraestructura fue desplegada con **AWS CloudFormation**, de forma que los recursos principales pueden recrearse con plantillas y scripts, evitando configuraciones manuales dispersas.

---

## Modelo de datos y entidad-relación

El modelo de datos se divide en dos capas:

- **Capa analítica en Amazon S3 + AWS Glue Data Catalog:** contiene features, forecast, evaluación, curvas, métricas y Model Registry.
- **Capa operacional en Amazon RDS PostgreSQL:** contiene feedback, historial de archivos, eventos de uso y metadatos operativos.

<p align="center">
  <img src="graficasProyecto/ER1transpa.png" alt="Diagrama entidad-relación principal del Glue Catalog" width="950">
</p>

**Lectura del diagrama entidad-relación.** La tabla `features` alimenta el scoring. De ahí se construye `forecast_detail`, que contiene la predicción final y su trazabilidad. Cuando existe valor real observado, `evaluation_detail` permite calcular errores. A partir de esa evaluación se generan agregados por producto, categoría, tienda y modelo. El Model Registry queda representado mediante `model_runs.csv` y `champion.json`.

### Artefactos analíticos principales

| Artefacto | Propósito | Uso en la app |
|---|---|---|
| `features` | Variables explicativas: lags, medias móviles, recencia, frecuencia, precio y señales acumuladas. | Inferencia individual, batch cargado y scoring. |
| `forecast_detail` | Predicción final por tienda-producto con `model_scope`, `routing_reason` y candidatos. | Resumen, Batch CFO, KPIs. |
| `evaluation_detail` | Comparación fila a fila contra `y` real. | Evaluación, Feedback, análisis de errores. |
| `evaluation_by_item` | Métricas por producto. | Evaluación y KPIs. |
| `evaluation_by_segment` | Métricas por categoría/segmento. | KPIs y lectura por grupo. |
| `evaluation_curves_by_model` | Curvas de promedio real vs promedio predicho por rango de demanda. | Evaluación y comparación de modelos. |
| `model_runs.csv` | Corridas evaluadas y métricas. | Model Registry. |
| `champion.json` | Modelo vigente y política de selección. | Resumen y Model Registry. |
| `review_suggestions.parquet` | Productos sugeridos para revisión. | Feedback. |

### Tablas operacionales en RDS

| Tabla | Propósito | Uso en la app |
|---|---|---|
| `business_feedback` | Observaciones del negocio sobre productos, tiendas o categorías. | Pestaña Feedback. |
| `problem_products` | Productos o pares tienda-producto sugeridos para revisión. | Feedback y priorización operativa. |
| `batch_exports` | Historial de archivos CFO generados. | Batch CFO. |
| `app_usage_events` | Eventos de uso y acciones relevantes. | Monitoreo operativo. |
| `model_metadata` | Metadatos complementarios de modelos. | Model Registry / operación. |

---

## Tecnologías utilizadas y evidencias AWS

| Capa | Tecnología | Uso |
|---|---|---|
| UI | **Streamlit** | Aplicación web de consulta, forecast, evaluación, KPIs y feedback. |
| Contenedores | **Docker** | Empaquetado reproducible de la aplicación. |
| Registry | **Amazon ECR** | Almacenamiento de la imagen Docker. |
| Cómputo app | **Amazon ECS Fargate** | Ejecución serverless del contenedor. |
| Exposición pública | **Application Load Balancer** | URL pública y balanceo hacia ECS. |
| Persistencia analítica | **Amazon S3** | ModelOps, predicciones, CFO exports, batch uploads, métricas y artefactos. |
| Base operacional | **Amazon RDS PostgreSQL** | Feedback, historial, eventos y metadata operacional. |
| Credenciales | **AWS Secrets Manager** | Secreto de conexión a RDS. |
| Logs | **Amazon CloudWatch Logs** | Depuración, reinicios, permisos y errores. |
| Catálogo | **AWS Glue Data Catalog** | Metadatos de tablas analíticas sobre S3. |
| Infraestructura | **AWS CloudFormation** | Despliegue reproducible de recursos. |
| Python env | **uv** | Administración rápida de dependencias y ejecución. |
| ML | **scikit-learn / HGB / LightGBM** | Modelos candidatos, baselines y política híbrida. |
| Data | **pandas / pyarrow** | Manipulación y lectura/escritura de CSV/Parquet. |
| AWS SDK | **boto3** | Lectura/escritura en S3 y operación AWS desde Python. |

### Capturas reales de recursos AWS

| Recurso | Captura | Archivo esperado |
|---|---|---|
| Aplicación pública Streamlit | Endpoint público del MVP | `graficasProyecto/apppublica.png` |
| Amazon ECS Fargate | Servicio ECS con task corriendo | `graficasProyecto/RecursosECSFargate.png` |
| Amazon ECR | Repositorio con imagen Docker | `graficasProyecto/RecursosECR.png` |
| AWS CloudFormation | Stacks de infraestructura | `graficasProyecto/RecursosCloudFormation.png` |
| Amazon RDS PostgreSQL | Instancia RDS disponible | `graficasProyecto/RecursosRDSBueno.png` |
| Amazon S3 | Bucket con exports, uploads y ModelOps | `graficasProyecto/RecursosS3.png` |
| AWS Glue Data Catalog | Tablas registradas | `graficasProyecto/RecursosGlue.png` |
| Amazon CloudWatch Logs | Logs del contenedor | `graficasProyecto/RecursosCloudwatch.png` |
| AWS Secrets Manager | Secreto de RDS | `graficasProyecto/RecursosSecretmanager.jpeg` |

<p align="center">
  <img src="graficasProyecto/RecursosECSFargate.png" alt="Amazon ECS Fargate con la tarea de Streamlit corriendo" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosECR.png" alt="Amazon ECR con imagen Docker de la app" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosCloudFormation.png" alt="AWS CloudFormation con stacks del MVP" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosRDSBueno.png" alt="Amazon RDS PostgreSQL disponible" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosS3.png" alt="Amazon S3 con archivos CFO, predicciones cargadas y ModelOps" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosGlue.png" alt="AWS Glue Data Catalog con tablas analíticas" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosCloudwatch.png" alt="Amazon CloudWatch Logs del servicio ECS Streamlit" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/RecursosSecretmanager.jpeg" alt="AWS Secrets Manager con secreto de RDS" width="850">
</p>

---

## Vistas de la aplicación

### 1. Resumen

La vista de Resumen muestra volumen de predicciones, modelo en uso, muestra de forecast, distribución de pronósticos, tiendas con mayor pronóstico, categorías con mayor pronóstico y cobertura por `model_scope`.

<p align="center">
  <img src="graficasProyecto/apppublica.png" alt="Vista Resumen de la aplicación pública" width="850">
</p>

### 2. Inferencia individual

Permite consultar un par tienda-producto específico. El usuario puede buscar por ID o nombre y revisar la predicción puntual.

<p align="center">
  <img src="graficasProyecto/inferenciaindividual.png" alt="Vista de inferencia individual" width="850">
</p>

### 3. Batch CFO

Genera archivos para negocio por tienda, categoría o catálogo completo. Los archivos CFO se guardan en Amazon S3 bajo `app/batch_exports/`, particionados por alcance.

<p align="center">
  <img src="graficasProyecto/batchcfo.png" alt="Batch CFO con selección de alcance y vista previa" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/batchcfo2.png" alt="Batch CFO con guardado en S3 e historial" width="850">
</p>

### 4. Batch por archivo cargado

Permite subir un CSV con features completas o estilo test (`ID`, `shop_id`, `item_id`). Si el par existe en las features preparadas, la app completa columnas; si no existe, lo trata como cold-start conservador. Las predicciones cargadas se guardan en `app/batch_uploads/predictions/` sin particionar.

<p align="center">
  <img src="graficasProyecto/batchcfo3.png" alt="Batch por archivo cargado con inferencia y política híbrida" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/batch4.png" alt="Historial exclusivo de predicciones por archivo cargado" width="850">
</p>

### 5. Evaluación

Compara modelos contra ground truth. Incluye curvas por rango de demanda real, distribución de demanda y tablas de performance por segmento/producto.

<p align="center">
  <img src="graficasProyecto/evaluacionseccion.png" alt="Vista de evaluación contra ground truth" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/evaluacion1_1.png" alt="Curvas de evaluación por modelo" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/distribuciondemanda.png" alt="Distribución de registros por rango de demanda real" width="750">
</p>

### 6. KPIs

Muestra categorías, productos y tiendas con mayor RMSE. También incluye productos con baja actividad reciente, top tiendas con más productos de baja actividad y top categorías afectadas.

<p align="center">
  <img src="graficasProyecto/kpis2.png" alt="Vista KPIs con errores y baja actividad reciente" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/kpis1.png" alt="KPIs de RMSE por tienda categoría y producto" width="850">
</p>

### 7. Feedback

Permite registrar observaciones del negocio y revisar productos sobreestimados, subestimados o sugeridos para revisión. Las tablas incluyen trazabilidad con `decision_recommendation`, `model_scope` y `routing_reason`.

<p align="center">
  <img src="graficasProyecto/feedback1.png" alt="Captura de feedback de negocio" width="850">
</p>

<p align="center">
  <img src="graficasProyecto/feedback3.png" alt="Productos sugeridos para revisión" width="850">
</p>

### 8. Model Registry

Muestra el modelo champion, métricas y corridas evaluadas. Conserva LightGBM original como incumbent, Hurdle HGB, HGB Poisson, especialista recurrente, rolling mean, promedio histórico y naive.

<p align="center">
  <img src="graficasProyecto/modelregistry.png" alt="Model Registry con historial de modelos y métricas" width="850">
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
- Features preparadas en `data/prep/` o artefactos publicados en S3.
- Modelo serializado en `artifacts/model.joblib` para inferencia local dentro del contenedor.
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
│   ├── ER1transpa.png
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
- AWS Secrets Manager para RDS.
- Task role con permisos específicos sobre S3.
- CloudWatch Logs para depuración.
- Separación entre `app/batch_exports/` y `app/batch_uploads/`.
- `.gitignore` para excluir `.env`, outputs locales, backups y artefactos pesados.

---

## Documentación metodológica

El reporte metodológico completo debe guardarse en:

```text
docs/Examen_ProductodeDatos_Avril_Hector.pdf
```

Este PDF documenta contexto de negocio, arquitectura, modelo de datos, evaluación, tour de la aplicación, evidencias AWS, costos, operación, seguridad, cobertura de rúbrica, limitaciones y uso de herramientas de IA.

---

## Limitaciones y siguientes pasos

- Agregar autenticación y autorización por rol.
- Automatizar ModelOps con jobs programados o pipelines externos.
- Monitorear drift y desempeño cuando llegue el valor real futuro.
- Evaluar SageMaker Batch Transform si crece el volumen de scoring.
- Enriquecer cold-start con catálogo de negocio, lanzamientos y atributos externos.
- Agregar notificaciones cuando un archivo CFO o batch cargado quede listo.

---

## Bibliografía y recursos consultados

1. Rožanec, J. M., Petelin, G., Costa, J., Bertalanič, B., Cerar, G., Guček, M., Papa, G. y Mladenić, D. (2023). *Dealing with zero-inflated data: achieving SOTA with a two-fold machine learning approach*. arXiv:2310.08088. Disponible en: https://arxiv.org/pdf/2310.08088

2. OpenAI. (2026). *ChatGPT, versión GPT-5.4*. Modelo de lenguaje utilizado como apoyo para estructurar el reporte, revisar redacción, documentar decisiones técnicas y depurar fragmentos de código. Disponible en: https://chat.openai.com/

3. Amazon Web Services. (2026). *AWS Fargate Pricing*. Disponible en: https://aws.amazon.com/fargate/pricing/

4. Amazon Web Services. (2026). *Elastic Load Balancing Pricing*. Disponible en: https://aws.amazon.com/elasticloadbalancing/pricing/

5. Amazon Web Services. (2026). *Amazon RDS for PostgreSQL Pricing*. Disponible en: https://aws.amazon.com/rds/postgresql/pricing/

6. Amazon Web Services. (2026). *Amazon S3 Pricing*. Disponible en: https://aws.amazon.com/s3/pricing/

7. Amazon Web Services. (2026). *Amazon ECR Pricing*. Disponible en: https://aws.amazon.com/ecr/pricing/

8. Amazon Web Services. (2026). *Amazon CloudWatch Pricing*. Disponible en: https://aws.amazon.com/cloudwatch/pricing/

9. Amazon Web Services. (2026). *AWS Secrets Manager Pricing*. Disponible en: https://aws.amazon.com/secrets-manager/pricing/

10. Amazon Web Services. (2026). *AWS Glue Pricing*. Disponible en: https://aws.amazon.com/glue/pricing/

---

## Autores

- Avril Salazar Rodríguez
- Héctor Vilchis Peralta

Curso: **Arquitectura de Productos de Datos y Métodos de Gran Escala**  
Institución: **ITAM**  
Fecha: **Mayo 2026**
