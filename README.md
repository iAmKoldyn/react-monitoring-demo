# Стек мониторинга (Prometheus + Grafana + Thanos + MinIO)

Описание единой платформы (Registry + Kubernetes + балансировка + персистентный мониторинг) см. в:

- `README.unified.md`

Статус/checkpoint по Kubernetes после апдейта `k8s/` описан в:

- `README.unified.md` (раздел `4. Чекпойнт после восстановления k8s/`)

Проект содержит локальный стек мониторинга с персистентным хранением метрик:

- `prometheus` собирает метрики
- `react_service` отдает метрики приложения (например `page_views`)
- `node_exporter` отдает метрики хоста
- `thanos sidecar` выгружает TSDB-блоки Prometheus в MinIO
- `thanos store` читает блоки из MinIO
- `thanos query` отдает единый Prometheus-совместимый API для запросов
- `grafana` визуализирует метрики

Опциональные генераторы нагрузки находятся в:

- `str_test_C++`
- `cv-reserch`

## Порты

- Grafana: `http://localhost:3000` (`admin` / `admin`)
- Prometheus: `http://localhost:9090`
- Endpoint метрик React: `http://localhost:3001/metrics`
- MinIO API: `http://localhost:9000`

## Одноразовые предварительные шаги

Так как compose использует внешние ресурсы, они создаются один раз перед первым запуском:

```bash
docker network create shared_network
docker volume create minio-data
```

Нужно создавать их вручную

- `shared_network`:  (`external: true`)
- `minio-data`: (`external: true`)

## Запуск основного стека

Из директории `react-monitoring-demo`:

```bash
docker compose up -d --force-recreate
```

Примечания:

- `minio_init` одноразовый контейнер для баскета. Статус `Exited (0)`.
- `minio_init` автоматически создает бакет `thanos` в MinIO.

## Запуск опциональных генераторов нагрузки

Из директории `react-monitoring-demo`:

```bash
docker compose -f str_test_C++/docker-compose.yml up -d --build
docker compose -f cv-reserch/docker-compose.yml up -d --build
```

Если они не запущены, часть таргетов в Prometheus будет в статусе `down`.

## Datasource в Grafana

Используется Thanos Query в качестве URL datasource:

```text
http://query:10902
```

Важно:

- Не использовать `http://sidecar:10902/metrics` как datasource Grafana.
- `sidecar:10902/metrics` это self-metrics самого sidecar.
- Запросы через `query` дают доступ к текущим и записанным данным из MinIO.

Файл provisioning datasource:

- `grafana/provisioning/datasources/datasource.yml`

## Поведение персистентности (важно)

Долгосрочная история метрик сохраняется в MinIO через Thanos.

Но значения на `/metrics` у приложения остаются process-local.

Пример с `page_views`:

- до рестарта: `page_views = 20`
- после рестарта `react_service`: счетчик в приложении сбрасывается, текущее значение может быть `0`
- новые запросы снова увеличивают счетчик

Так и должно быть с Prometheus `Counter`, который отдается перезапущенным процессом приложения.

История не теряется, если смотреть через Thanos Query/Grafana.

PromQL-выражения для дашбордов:

- просмотры за выбранный период: `sum(increase(page_views[$__range]))`
- скорость в секунду: `sum(rate(page_views[1m]))`

## Быстрые проверки

```bash
docker compose ps -a
docker compose logs store --tail 100
docker compose logs sidecar --tail 100
docker compose logs minio_init --tail 50
```

Ожидаемый healfcheck:

- `store` в состоянии `Up` и нет ошибки `The specified bucket does not exist`
- в логах `sidecar` есть `upload new block`
- `query` видит store-источники (`sidecar:10901`, `store:10901`)

## Остановка

```bash
docker compose down
```

По умолчанию тома сохраняются, включая внешний `minio-data`.
