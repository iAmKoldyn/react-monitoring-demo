# Единая DevOps-система: Kubernetes + Registry + Monitoring + CI/CD

## 1. Идея системы

Это единая платформа на Kubernetes-кластере с:

- `master`-нодой (control plane);
- `worker`-нодами (где крутятся приложения и нагрузчики);
- приватным Docker Registry;
- отдельным персистентным контуром мониторинга (Prometheus + Thanos + MinIO + Grafana);
- нагрузчики (имитирующие сервисы), с которых снимаются метрики.

Цель: разработчик публикует Docker-образ в registry, приложение разворачивается в кластере, метрики автоматически собираются и сохраняются персистентно даже при рестартах подов/падении ноды.

## 2. Целевая архитектура

```text
Developer
  -> CI (GitLab CI / Runner)
  -> Build image
  -> Push image to private Registry
  -> Deploy to Kubernetes (namespace workloads)

Kubernetes cluster:
  - master/control-plane
  - worker-1
  - worker-2
  - ...

Runtime + traffic:
  App Deployment (replicas > 1)
  -> Service
  -> Ingress / LoadBalancer

Monitoring (отдельный контур):
  App metrics + node metrics
  -> Prometheus
  -> Thanos Sidecar
  -> MinIO (object storage)
  -> Thanos Store
  -> Thanos Query
  -> Grafana (datasource: Thanos Query)
```

## 3. Что уже есть

- Локальный рабочий стек мониторинга через `docker-compose`:
  - `prometheus`, `grafana`, `node_exporter`, `minio`, `thanos sidecar`, `thanos store`, `thanos query`, `react_service`.
- Настроен datasource Grafana на:
  - `http://query:10902`
- Реализована персистентность метрик через Thanos + MinIO.
- Есть нагрузчики:
  - `str_test_C++`
  - `cv-reserch`

### В wiki (`registry.github.io.wiki`)

- Подробные инструкции по:
  - Kubernetes Docker Registry;
  - `PV/PVC`;
  - `Secret` / `imagePullSecrets`;
  - `Sealed Secrets`;
  - внешнему доступу через `Ingress`/`LoadBalancer`.

## 4. Чекпойнт `k8s/`

`k8s/` содержит 15 манифестов:

- `grafana`: deployment/service/pvc
- `minio`: deployment/service/pvc
- `prometheus`: deployment/configmap/pvc
- `react-service`: deployment/service
- `node-exporter`: daemonset
- `thanos`: sidecar deployment/configmap + store pvc

Примечание:

- Связка мониторинга в `k8s` сейчас неполная.
- Манифесты это просто базовый каркас.
- `kubectl apply --dry-run=client` работать не будет!

Архитектурные разрывы в `k8s/`:

1. Отсутствуют Thanos Query/Store deployment и сервисы.
2. Нет Service для `thanos-sidecar`, но Prometheus скрапит `thanos-sidecar:10902`.
3. Нет Service для `node-exporter`, но Prometheus скрапит `node_exporter:9100`.
4. `thanos-sidecar` не монтирует TSDB volume Prometheus (`/prometheus`), но использует `--tsdb.path=/prometheus`.
5. `react-service` использует локальный placeholder image `localhost:3001/react-service:latest`, не путь к registry внутри кластера.
6. Нет инициализации bucket `thanos` в MinIO (в compose это закрыто `minio_init`, в k8s аналога нет).
7. Учетные данные MinIO лежат в открытом виде в ConfigMap/Deployment.
8. Нет namespace-изоляции (`registry`, `monitoring`, `workloads`, `ci`) нет ingress/load balancer схемы.

## 5. Что нужно доработать

1. Дописать манифесты `k8s/`.
2. Полный Thanos-контур в Kubernetes:
   - `sidecar`, `store`, `query` + сервисы.
3. Разделить систему по namespace-ам:
   - `registry`, `monitoring`, `workloads`, `ci`.
4. Описать и реализовать единый балансировщик:
   - `Ingress` (предпочтительно) или `Service type: LoadBalancer`.
5. Доделать discovery метрик:
   - через `ServiceMonitor/PodMonitor`.
6. Формализовать автодеплой из registry:
   - чтобы push/merge запускал rollout в кластер.

## 6. Что нужно реализовать (план)

### Базовая платформа кластера

- Registry в Kubernetes с персистентным хранилищем (`PVC`).
- Безопасность registry:
  - `Secret` или `Sealed Secret`;
  - `imagePullSecrets` для приложений.
- Базовый ingress-контроллер и внешняя точка входа.

### Мониторинг и хранение метрик

- Prometheus + node exporter (DaemonSet на worker-нодах).
- Thanos Sidecar + Thanos Store + Thanos Query.
- MinIO как object storage для long-term метрик.
- Grafana с datasource на `Thanos Query`.

### Нагрузчики или реальные инстансы сервисов(в оригинале сервисы ходят друг в друга) и отказоустойчивость

- Развернуть нагрузчики по разным worker-нодам.
- Проверить сценарий падения ноды:
  - приложение продолжает работать;
  - новые метрики продолжают писаться;
  - исторические метрики доступны.

### GitLab + Runner (добавление в систему)

- Развернуть GitLab instance ( или GitLab Helm chart).
- Развернуть GitLab Runner (Kubernetes executor).
- CI/CD pipeline:
  1. build image;
  2. push в private registry;
  3. deploy в Kubernetes;
  4. smoke-check + проверка метрик.

## 7. Пайплайн (целевой)

1. Разработчик пушит код в GitLab.
2. GitLab Runner собирает Docker-образ.
3. Образ публикуется в private Registry.
4. Deployment в Kubernetes обновляется на новый тег.
5. Приложение становится доступным через Ingress/LoadBalancer.
6. Prometheus снимает метрики с `/metrics`.
7. Thanos отправляет блоки в MinIO.
8. Grafana показывает текущие и записанные метрики через `query`.

## 8. Критерии готовности кластера, регистри, воркеров на нодах и персистентного хранения метрик:

Система считается собранной, если:

1. Образ из registry автоматически разворачивается в кластере.
2. Трафик идёт через единый балансировщик (`Ingress`/`LB`).
3. Метрики приложения и нод стабильно собираются.
4. После рестарта пода/падения ноды история метрик не теряется.
5. Grafana показывает данные через Thanos Query.
6. CI/CD (GitLab + Runner) закрывает путь от кода до деплоя.

## 10. Текущий локальный режим (для быстрой проверки)

В локальном compose-режиме предварительно нужны:

```bash
docker network create shared_network
docker volume create minio-data
```

Datasource Grafana:

```text
http://query:10902
```
