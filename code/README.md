# EGE RAG Service

RAG-сервис генерации персонализированных учебных конспектов для платформы «ЕГЭ Прорыв». Принимает историю ответов ученика, извлекает релевантные атомы знаний из векторной базы (Qdrant) и генерирует конспект с помощью LLM-агентов (LangGraph).

## Архитектура

Два асинхронных потока обработки:

- **Поток 1 — фоновая диагностика** (`eval-worker`): потребляет события `submission.created` из Apache Kafka (Redpanda), запускает LLM-диагностику каждого неправильного ответа и возвращает теги ошибок платформе callback'ом.
- **Поток 2 — генерация конспекта** (`conspect-worker`): принимает задание через `POST /jobs`, помещает в очередь Postgres (`SELECT FOR UPDATE SKIP LOCKED`), запускает RAG-пайплайн и доставляет результат callback'ом или polling'ом.

```
Платформа → Kafka → eval-worker → LLM → callback POST /diagnostic-tags
Платформа → POST /jobs → API → conspect_jobs (PG) → conspect-worker → RAG+LLM → callback
```

## Структура

```
app/
├── api/            — FastAPI routes (/jobs, /healthz, /readyz)
├── core/           — конфигурация (pydantic-settings), логирование
├── domain/         — модели атомов знаний и профиля ученика
├── infrastructure/ — RAG-пайплайн, LLM-клиенты, Qdrant, Postgres, HTTP
├── services/       — бизнес-логика (conspect, diagnosis, profile)
├── workers/        — eval_worker, conspect_worker
├── migrations/     — Alembic
├── data/           — атомы знаний (YAML), промпты, профили
├── rag/eval/       — оценочный контур (DeepEval, бенчмарки)
├── scripts/        — вспомогательные скрипты (импорт, генерация данных)
└── spec/           — OpenAPI и Kafka JSON Schema контракты
admin/              — симулятор платформы + панель наблюдаемости
k8s/                — Kubernetes-манифесты (см. k8s/README.md)
```

## Быстрый старт (docker-compose)

```bash
cp app/.env.example app/.env
# заполнить OPENROUTER_API_KEY и прочие переменные

docker compose up
# API: http://localhost:8000
# Admin/симулятор: http://localhost:8100
```

## Конфигурация

Все параметры читаются из `app/.env` (см. `app/.env.example`) и переменных окружения. Ключевые:

| Переменная | Назначение |
|---|---|
| `LLM_BACKEND` | `groq` / `openrouter` / `ollama` |
| `OPENROUTER_API_KEY` | ключ OpenRouter (если `LLM_BACKEND=openrouter`) |
| `QDRANT_URL` | адрес Qdrant |
| `DATABASE_URL` | asyncpg DSN для Postgres |
| `KAFKA_BOOTSTRAP_SERVERS` | брокер Kafka / Redpanda |
| `PLATFORM_BASE_URL` | адрес платформы для callback'ов |

## Тесты

```bash
uv run pytest tests/
```

## Развёртывание в Kubernetes

См. [`k8s/README.md`](k8s/README.md).

## Контракты интеграции

Спецификации API и схема Kafka-события находятся в `app/spec/`:

- `generator.openapi.yaml` — API сервиса (`POST /jobs`, `GET /jobs/{id}`)
- `platform.openapi.yaml` — callback-эндпоинты платформы
- `kafka_submission_event.schema.yaml` — JSON Schema события `submission.created`
