# Monitoring Stack (Prometheus + Grafana + Thanos + MinIO)

This project contains a local monitoring stack with long-term metrics storage:

- `prometheus` scrapes metrics
- `react_service` exposes app metrics (for example `page_views`)
- `node_exporter` exposes host metrics
- `thanos sidecar` uploads Prometheus TSDB blocks to MinIO
- `thanos store` reads blocks from MinIO
- `thanos query` serves unified Prometheus-compatible query API
- `grafana` visualizes metrics

Optional load generators are in:

- `str_test_C++`
- `cv-reserch`

## Ports

- Grafana: `http://localhost:3000` (`admin` / `admin`)
- Prometheus: `http://localhost:9090`
- React metrics endpoint: `http://localhost:3001/metrics`
- MinIO API: `http://localhost:9000`

## One-time prerequisites

Because compose uses external resources, create them once before first start:

```bash
docker network create shared_network
docker volume create minio-data
```

Do you need to create them manually?

- `shared_network`: yes, required (`external: true`)
- `minio-data`: yes, required (`external: true`)

## Start the main stack

From `react-monitoring-demo` directory:

```bash
docker compose up -d --force-recreate
```

Notes:

- `minio_init` is a one-shot container. `Exited (0)` is expected.
- `minio_init` creates MinIO bucket `thanos` automatically.

## Start optional load generators

From `react-monitoring-demo` directory:

```bash
docker compose -f str_test_C++/docker-compose.yml up -d --build
docker compose -f cv-reserch/docker-compose.yml up -d --build
```

If they are not running, some Prometheus targets will stay `down` (expected).

## Grafana datasource

Use Thanos Query as datasource URL:

```text
http://query:10902
```

Important:

- Do not use `http://sidecar:10902/metrics` as Grafana datasource.
- `sidecar:10902/metrics` is only sidecar self-metrics endpoint.
- Querying through `query` gives access to current + historical data from MinIO.

Datasource provisioning file:

- `grafana/provisioning/datasources/datasource.yml`

## Persistence behavior (important)

Long-term history is persisted in MinIO through Thanos.

But `/metrics` values from app process are still process-local.

Example with `page_views`:

- before restart: `page_views = 20`
- after `react_service` restart: app counter resets, current value may show `0`
- new requests increment from there again

This is expected for Prometheus `Counter` exposed by a restarted app process.

Your history is not lost if you query via Thanos Query/Grafana.

Use these PromQL expressions for dashboards:

- views over selected range: `sum(increase(page_views[$__range]))`
- per-second rate: `sum(rate(page_views[1m]))`

## Quick checks

```bash
docker compose ps -a
docker compose logs store --tail 100
docker compose logs sidecar --tail 100
docker compose logs minio_init --tail 50
```

Expected healthy signs:

- `store` is `Up` and no `The specified bucket does not exist`
- `sidecar` logs `upload new block`
- `query` sees stores (`sidecar:10901`, `store:10901`)

## Stop

```bash
docker compose down
```

This keeps volumes by default, including external `minio-data`.
