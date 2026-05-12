# Kubernetes manifests

Targets namespace `ege`. Postgres, Redpanda, and Qdrant are deployed
in-namespace as single-replica StatefulSets — fine for staging/dev, swap for
managed services in production.

## Manifests

| File | Workload |
|---|---|
| `api.yaml` | `POST /jobs` API · 2 replicas, HPA → 6 |
| `eval-worker.yaml` | Kafka consumer (Flow 1) · 2 replicas, HPA → #partitions |
| `conspect-worker.yaml` | DB-queue worker (Flow 2) · 1 replica, HPA → 4 |
| `admin.yaml` | Platform simulator + monitoring UI · 1 replica |
| `migrations-job.yaml` | One-shot Alembic upgrade |
| `deps-postgres.yaml` / `deps-redpanda.yaml` / `deps-qdrant.yaml` | StatefulSets + Services |
| `configmap.yaml` / `secret.example.yaml` | Runtime config / secret template |
| `ingress.yaml` | Ingress for the API (adjust host + class) |

## Apply

```bash
# 1. config + secrets (copy template, fill values, then apply)
kubectl apply -f configmap.yaml
cp secret.example.yaml secret.yaml && $EDITOR secret.yaml
kubectl apply -f secret.yaml

# 2. deps + migrations
kubectl apply -f deps-postgres.yaml -f deps-redpanda.yaml -f deps-qdrant.yaml
kubectl apply -f migrations-job.yaml

# 3. workloads
kubectl apply -f api.yaml -f eval-worker.yaml -f conspect-worker.yaml -f admin.yaml
kubectl apply -f ingress.yaml
```

## Image build

```bash
TAG=$(git rev-parse --short HEAD)
podman build -t REGISTRY/ege-rag:$TAG .
podman push REGISTRY/ege-rag:$TAG
# update image: tags in api.yaml, eval-worker.yaml, conspect-worker.yaml, admin.yaml
```

## Scaling notes

- **eval-worker**: cap at the partition count — extra replicas sit idle.
- **conspect-worker**: scales freely (DB queue with `SELECT FOR UPDATE SKIP LOCKED`).
- HPAs use CPU. For Kafka-lag-aware scaling, install KEDA and switch
  `eval-worker` to a `kafka` trigger.

## Secrets

`secret.example.yaml` is a template only. In production, source secrets from
External Secrets Operator + Vault / AWS SM / GCP SM — never commit a populated
`secret.yaml`.

## Admin access

The admin Service is `ClusterIP` only. Reach the dashboard with port-forward:

```bash
kubectl port-forward svc/ege-admin 8100:80 -n ege
# then open http://localhost:8100
```
