# Costos y apagado

## Recursos que cobran más mientras están activos

- ECS Fargate task.
- Application Load Balancer.
- RDS PostgreSQL.
- SageMaker Processing durante ejecución de jobs.

## Recursos de bajo costo relativo

- S3 storage.
- Glue Data Catalog metadata.
- ECR storage.
- CloudWatch Logs con retención baja.

## Apagar app por la noche

```bash
source config/generated.env
aws ecs update-service \
  --cluster pfs-mvp-cluster \
  --service pfs-mvp \
  --desired-count 0 \
  --region "$AWS_REGION"
```

## Encender app

```bash
source config/generated.env
aws ecs update-service \
  --cluster pfs-mvp-cluster \
  --service pfs-mvp \
  --desired-count 1 \
  --region "$AWS_REGION"
```

## Detener RDS

```bash
aws rds describe-db-instances \
  --query "DBInstances[].{DB:DBInstanceIdentifier,Status:DBInstanceStatus}" \
  --region "$AWS_REGION"

aws rds stop-db-instance \
  --db-instance-identifier <DB_INSTANCE_ID> \
  --region "$AWS_REGION"
```

No borres S3 ni GitHub hasta terminar la evaluación.
