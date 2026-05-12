# EGE Conspect Generator

RAG + LLM service that generates personalised study notes ("conspects") from a
student's submission history, plus per-submission mistake-tagging. Backend for
the EGE Proryv platform integration; contracts in [`spec/`](spec/).

## Components

```
Platform ─Kafka─► eval-worker ─LLM diagnose─► POST /v1/.../diagnostic-tags
Platform ─POST /jobs─► API ─► conspect_jobs (PG) ─► conspect-worker ─LLM─► POST /v1/personal-conspect
```

- `app.main` — FastAPI app, exposes `POST /jobs`, `GET /jobs/{id}`, `/healthz`, `/readyz`
- `app.workers.eval_worker` — direct Kafka consumer (Flow 1)
- `app.workers.conspect_worker` — DB-queue worker (Flow 2)
- `admin/` — local platform simulator + monitoring dashboard

## Local quickstart

В compose всё это поднимается одним `podman compose up -d`. Adminer (UI для PG)
на :8080, Redpanda Console на :8081.

---

## RAG-движок (исходный PoC)

Простой proof-of-concept для проверки эффекта RAG на персонализированных конспектах по заданию №10 ЕГЭ профиль.

## Быстрый старт

### 1. Установка зависимостей ([uv](https://docs.astral.sh/uv/))

Из **корня репозитория** (где `pyproject.toml`):

```powershell
cd c:\Users\user\Documents\Ege
uv sync
```

Создаётся `.venv` и ставятся зависимости из [`pyproject.toml`](../pyproject.toml). Дальше все команды — через `uv run` (не нужно активировать venv вручную).

Альтернатива без uv: `python -m venv .venv`, затем `pip install -r rag_poc/requirements.txt`.

### 2. LLM: Groq (по умолчанию) или локальный Ollama

**Groq** (облако):

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
# опционально: $env:GROQ_MODEL="qwen/qwen3-32b"
```

**Ollama** (локально, без ключа Groq):

```powershell
$env:LLM_BACKEND="ollama"
$env:OLLAMA_MODEL="qwen3:8b"
# опционально: $env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

Переменная `LLM_BACKEND`: `groq` (по умолчанию), `openrouter` или `ollama`.

**OpenRouter** (любые модели с каталога, один ключ):

```powershell
$env:LLM_BACKEND="openrouter"
$env:OPENROUTER_API_KEY="sk-or-v1-..."
# опционально: $env:OPENROUTER_MODEL="qwen/qwen2.5-72b-instruct"
```

Ключ: [openrouter.ai/keys](https://openrouter.ai/keys). Модель — slug с сайта (например `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`).

**Эмбеддинги через OpenRouter** (без локального `sentence-transformers` / HF): в `rag_poc/.env` задайте `EMBED_BACKEND=openrouter`, тот же `OPENROUTER_API_KEY`, и `EMBED_MODEL` — модель эмбеддингов с каталога OpenRouter (например `openai/text-embedding-3-small`). Иначе по умолчанию `EMBED_BACKEND=local` и `EMBED_MODEL=BAAI/bge-m3`. После смены бэкенда эмбеддингов перезапустите приложение (индекс строится при старте).

Если OpenRouter отвечает **«No successful provider responses»**, задайте **`OPENROUTER_EMBED_PROVIDER_ORDER=openai`** (или другой slug провайдера из документации), либо список запасных моделей **`OPENROUTER_EMBED_MODEL_FALLBACKS`**. В коде есть автоматический перебор запасных моделей при первой успешной попытке.

Шаблон переменных: скопируйте [.env.example](.env.example) в `.env` в этой папке.

**Конспекты (`CONSPECT_VERIFY`)**: второй проход LLM (проверка + ремонт) по умолчанию **включён** для Groq и OpenRouter и **выключен** для Ollama (малые модели не выигрывают от лишних вызовов). Принудительно: `$env:CONSPECT_VERIFY="true"` или `"false"`.

**Структурированный вывод (`CONSPECT_STRUCTURED_OUTPUT`)**: если в [`data/conspect_prompts.yaml`](data/conspect_prompts.yaml) задан блок `conspect_json_output`, модель возвращает JSON по секциям; сервер проверяет математику по разделам «Разбор примера», «Потренируйся», «Найди ошибку» отдельно. Отключить JSON и вернуться к одному markdown-ответу: `$env:CONSPECT_STRUCTURED_OUTPUT="false"`.

**RAG: mastery gating**: переменные `RAG_MASTERY_THRESHOLD` (по умолчанию 0.8) и `RAG_MASTERY_PENALTY` (0.15) ослабляют в retrieval атомы с высокой оценкой освоения.

В UI после генерации конспекта доступна кнопка **«Сохранить в PDF»** (библиотека html2pdf.js); при ошибке экспорта можно использовать `Печать` (Ctrl+P) с `@media print`.

Для qwen3 через Ollama клиент отправляет `think: false`, чтобы ответ не съедался внутренними рассуждениями модели.

**Ollama и облако:** если в ошибке фигурирует `ollama.com` / `unexpected EOF`, локальный Ollama пытается ходить в облако и соединение рвётся. Сделайте `ollama pull <модель>`, чтобы модель была **локально**, проверьте сеть/VPN, либо временно переключитесь на Groq (`LLM_BACKEND=groq`). В `.env` оставьте `OLLAMA_BASE_URL=http://127.0.0.1:11434`. Повторы при 502/503/504: `OLLAMA_MAX_RETRIES` (по умолчанию 3).

Если LLM недоступен (нет ключа при `groq`, Ollama не запущен при `ollama`), диагностика ошибок отключается, генерация конспекта по API вернёт 503.

### 3. Запуск сервера

```powershell
uv run uvicorn rag_poc.app:app --reload
```

Откройте браузер: `http://127.0.0.1:8000`

### Проверка на реальных данных (архив экспорта)

Цепочка та же, что в проде: **профиль** → **RAG retrieval + промпт** → **LLM** (+ опционально `CONSPECT_VERIFY`, `CONSPECT_MATH_VERIFY`) — реализация в [rag/conspect_generation.py](rag/conspect_generation.py), вызывается из [app.py](app.py) как `GET /conspect`.

1. **Импорт CSV** — заполняет `ProfileStore` через `record_attempt` (как при отправке ответа в API). Берутся только строки с `task_number` 6, 10 и 12 из `user_task_submissions.csv` (`;`, UTF-8 BOM).

```powershell
# Default archive: rag_poc/data/submissions_archive/
uv run python -m rag_poc.scripts.import_submissions_archive
uv run python -m rag_poc.scripts.import_submissions_archive --archive path/to/archive --dry-run
```

**FIPI JSON (`fipi_parsed_katex.json`)**: если в задачах нет `problem_katex`, UI показывает плоский `question` без формул. Заполните поле из MathML в `problem`:

```powershell
# from repo root
uv run python -m rag_poc.scripts.enrich_fipi_problem_katex --input fipi_parsed_katex.json
uv run python -m rag_poc.scripts.enrich_fipi_problem_katex --dry-run
```

2. **Пакетные конспекты** — вызывают тот же движок, что UI/API, без дублирования логики.

```powershell
# с корневого каталога репозитория (рекомендуется uv run — подтянет зависимости из pyproject)
uv run python -m rag_poc.scripts.generate_archive_conspects

# только первые N профилей (проверка пайплайна)
uv run python -m rag_poc.scripts.generate_archive_conspects --limit 3

# выбранные user_number из архива
uv run python -m rag_poc.scripts.generate_archive_conspects --users 20,15,11
```

Профили: `rag_poc/data/profiles/archive_u{user_number}.json`. Результаты: `rag_poc/data/archive_conspects/*.md`. В UI **Ученик** = `archive_u3` и т.д. Теги при импорте — эвристика по `node_name` / `task_name` (в архиве нет текстов ответов).

## Векторная база данных (Qdrant)

RAG retrieval использует **Qdrant**: гибридный поиск по плотным эмбеддингам и разреженным векторам (BM25), см. [`rag/vector_db.py`](rag/vector_db.py).

**Запуск локально** — из каталога [`rag/docker-compose.yaml`](rag/docker-compose.yaml):

```powershell
cd rag_poc/rag
docker compose up -d
```

По умолчанию контейнер слушает **6333** (HTTP API); данные монтируются в `./qdrant_data` рядом с compose-файлом.

**Переменные окружения** (в `rag_poc/.env` или в оболочке):

| Переменная                   | Назначение                                                 | По умолчанию            |
| ---------------------------- | ---------------------------------------------------------- | ----------------------- |
| `QDRANT_URL`                 | URL REST API Qdrant                                        | `http://localhost:6333` |
| `QDRANT_COLLECTION`          | Имя коллекции с атомами                                    | `ege_rag_atoms`         |
| `RAG_QDRANT_PREFETCH_FACTOR` | Множитель prefetch для гибридного поиска (см. `RagConfig`) | `2`                     |

При старте приложения коллекция создаётся при необходимости; индекс пересобирается, если сменилась размерность эмбеддингов или хеш контента атомов.

## Использование

1. **Загрузите задачу** — нажмите "Новая задача"
2. **Введите ответ** и нажмите "Отправить"
3. **Посмотрите результат** — система покажет правильность и возможные ошибки
4. **Загрузите конспект**:
   - "Базовый конспект" — без персонализации
   - "RAG конспект" — персонализированный на основе ошибок ученика

## API Endpoints

- `GET /task10/new?student_id=demo` — получить случайную задачу №10
- `POST /task10/submit` — отправить ответ, получить диагностику
- `GET /task10/conspect?student_id=demo&task_id=...` — получить конспект (baseline vs RAG)
- `GET /docs` — Swagger UI

## Структура

- `rag/atoms.py` — атомы знаний (правила, ошибки)
- `rag/subtypes.py` — классификатор подтипов задач
- `rag/rag_engine.py` — RAG-движок (retrieval + промпты)
- `rag/vector_db.py` — Qdrant: коллекция, гибридный dense + sparse поиск
- `rag/profile.py` — профили учеников (JSON)
- `rag/mistakes.py` — диагностика ошибок
- `static/` — фронтенд (HTML/CSS/JS)
- `data/profiles/` — сохранённые профили учеников

## Проверка эффекта RAG

Сравните:

1. **Базовый конспект** — статичный текст без учёта ошибок ученика
2. **RAG конспект** — персонализированный, учитывающий частые ошибки из профиля

После нескольких неправильных ответов профиль наполнится, и RAG-конспект будет подтягивать релевантные атомы по ошибкам ученика.

## Запуск оценки RAG системы
Dependencies: Postgres, Redpanda/Kafka, Qdrant — easiest path is to run them
in a kubernetes cluster via the manifests in [`k8s/`](../k8s/) and port-forward.

```bash
uv sync
cp app/.env.example app/.env       # fill in OPENROUTER_API_KEY etc.
uv run alembic upgrade head        # apply migrations
uv run uvicorn app.main:app        # API (port 8000)
uv run python -m app.workers.eval_worker
uv run python -m app.workers.conspect_worker
uv run uvicorn admin.app:app --port 8100   # monitoring + simulator
```

## Оценка качества генерации конспекта (G-Eval)

Бенчмарк прогоняет LLM-судью (DeepEval G-Eval) по сохранённым конспектам: пары `*.md` + `*.meta.json` в [`rag/eval/conspects/`](rag/eval/conspects/) (контекст ученика для метрики персонализации). Правила метрик: [`data/conspect_eval_rules.yaml`](data/conspect_eval_rules.yaml).

1. Установите зависимости для eval: `uv sync --extra eval` (из корня репозитория).
2. Настройте тот же LLM, что для приложения (например `GROQ_API_KEY` / `LLM_BACKEND` и при необходимости `JUDGE_MODEL` — см. клиент судьи в коде).
3. Запуск из корня репозитория:

```bash
uv run python -m app.rag.eval.run_conspect_benchmarks
```

Результат по умолчанию: [`rag/eval/metrics/conspect_geval_results.json`](rag/eval/metrics/conspect_geval_results.json). Опции: `--conspects-dir` (каталог с конспектами), `--out` (путь к JSON). Параллелизм вызовов судьи: переменная окружения `JUDGE_CONCURRENCY` (по умолчанию 4).
## LLM backends

Set `LLM_BACKEND` to one of: `openrouter` (default), `groq`, `ollama`.
Config keys live in [`app/.env.example`](.env.example).

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
```

## Deployment

See [`k8s/`](../k8s/) for manifests (api, workers, admin, in-namespace deps).

## Retrieval evaluation

Run on demand:

```bash
uv run python -m app.rag.eval.run_retrieval_benchmarks
```

Reports land in `app/rag/eval/benchmark_retrieval_results.json` and
`app/rag/eval/judge_retrieval_results.json`.

## Метрики retrieval

<!-- AUTO_METRICS:START -->

_Источник: `rag/eval/benchmark_retrieval_results.json` и `rag/eval/judge_retrieval_results.json`._

### Task 6: evaluated=46

@1: precision=0.8913 recall=0.1776 ndcg=0.8913 mrr=0.8913 map=0.8913 hitrate=0.8913
@3: precision=0.6087 recall=0.3529 ndcg=0.6915 mrr=0.9022 map=0.6014 hitrate=0.9130
@5: precision=0.3870 recall=0.3683 ndcg=0.5414 mrr=0.9022 map=0.4103 hitrate=0.9130
@10: precision=0.2761 recall=0.5273 ndcg=0.5836 mrr=0.9022 map=0.4279 hitrate=0.9130

### Task 10: evaluated=50

@1: precision=0.7800 recall=0.0366 ndcg=0.7800 mrr=0.7800 map=0.7800 hitrate=0.7800
@3: precision=0.6600 recall=0.0952 ndcg=0.6894 mrr=0.8833 map=0.6056 hitrate=1.0000
@5: precision=0.5960 recall=0.1342 ndcg=0.6383 mrr=0.8833 map=0.5221 hitrate=1.0000
@10: precision=0.5220 recall=0.2343 ndcg=0.5767 mrr=0.8833 map=0.4111 hitrate=1.0000

### Task 12: evaluated=51

@1: precision=0.8235 recall=0.0611 ndcg=0.8235 mrr=0.8235 map=0.8235 hitrate=0.8235
@3: precision=0.6013 recall=0.1320 ndcg=0.6454 mrr=0.8856 map=0.5392 hitrate=0.9608
@5: precision=0.6196 recall=0.2275 ndcg=0.6473 mrr=0.8954 map=0.5024 hitrate=1.0000
@10: precision=0.5255 recall=0.3836 ndcg=0.5750 mrr=0.8954 map=0.3971 hitrate=1.0000

### All tasks: evaluated=147

@1: precision=0.8299 recall=0.0892 ndcg=0.8299 mrr=0.8299 map=0.8299 hitrate=0.8299
@3: precision=0.6236 recall=0.1886 ndcg=0.6748 mrr=0.8900 map=0.5813 hitrate=0.9592
@5: precision=0.5388 recall=0.2398 ndcg=0.6111 mrr=0.8934 map=0.4802 hitrate=0.9728
@10: precision=0.4463 recall=0.3778 ndcg=0.5783 mrr=0.8934 map=0.4115 hitrate=0.9728

## Метрики LLM judge

- Оценено студентов: 27
- Overall avg judge score: 0.4221
- Средняя доля релевантных атомов@12: 55.25% (179/324)

<!-- AUTO_METRICS:END -->
