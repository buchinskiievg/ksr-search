# Перенос на Cloudflare Workers + D1 — план

## Что нужно

| Слой | Сейчас (Python / Flask) | Будет (Cloudflare) |
|---|---|---|
| Backend | Flask на 127.0.0.1:5000 | Worker (TypeScript) |
| Items DB | SQLite `ksr.sqlite`, 172k строк | D1 (managed SQLite, тот же SQL) |
| Поиск | sklearn TF-IDF матрицы 561 MB | D1 FTS5 (full-text search), 0 памяти в Worker |
| Examples (рост) | таблица `examples` в локальной SQLite | таблица `examples` в D1 (то же место, что items) |
| LLM-вызовы | `httpx` → Groq/Gemini | `fetch` → Groq/Gemini (тот же API) |
| Frontend | Flask template | Static `index.html` на Cloudflare Pages, fetch к Worker |
| Хостинг | localhost | https://ksr-search.pages.dev (или свой домен) |

## Что меняется в качестве поиска

**Главное отличие:** Python-версия использует TF-IDF матрицы (word 1-2 ngram + char_wb 3-5 ngram), CF-версия должна использовать SQLite FTS5 (BM25 ранжирование, токенизация по словам).

- ✅ FTS5 хорошо находит «кабель ВВГнг-LS 3х2,5» — даёт BM25 score, можно сортировать.
- ⚠️ FTS5 хуже на «БДЛ-1» (короткая аббревиатура без word-boundary) — нужен trigram tokenizer и/или AI-расшифровка ВСЕГДА включена для коротких запросов.
- ✅ AI re-rank остаётся как был — компенсирует слабые места FTS.
- ✅ Boost от примеров остаётся — D1 хранит всё, JS считает похожесть.

## Лимиты Cloudflare (Free)

| Ресурс | Лимит | Хватает? |
|---|---|---|
| Worker CPU | 10 ms / запрос | Может быть тесно при rerank loop. Решение: Pages Functions (50 ms) или paid Workers ($5/мес → 30 с). |
| Worker memory | 128 MB | ✓ Не загружаем матрицы, только D1 query results. |
| D1 size | 10 GB | ✓ 172k items × ~200 bytes = ~35 MB. Места под рост примеров много. |
| D1 reads | 5M / день | ✓ С запасом. |
| D1 writes | 100k / день | ✓ Клики редкие. |

## План работ

1. **D1 schema** + bulk-import 172k items из текущего SQLite (`wrangler d1 execute --file=...sql` или Python-скрипт через REST API).
2. **FTS5 виртуальная таблица** поверх items (поле `name` + триграммы для аббревиатур).
3. **Worker `/api/search`** — D1 FTS5 MATCH запрос → JSON top-N.
4. **Worker `/api/expand`** + **`/api/rerank`** — fetch к Groq/Gemini, тот же prompt.
5. **Worker `/api/feedback`** + **`/api/ingest`** — INSERT в `examples` в D1.
6. **Pipeline** — TS-версия `_pipeline()`: expand → search (D1) → rerank (LLM) → boost из examples.
7. **Frontend** — текущий `templates/index.html` адаптируется под Cloudflare Pages (поменять `fetch('/api/...')` URL'ы — они и так относительные).
8. **Deploy**: `wrangler deploy` для Worker, `wrangler pages deploy` для UI.

## Что нужно от тебя

- Я могу всё запрограммировать, но нужны:
  - **Cloudflare account_id + API token** (есть у тебя в engineers_apps — заюзаю)
  - **Зелёный свет на занятие D1 base** (создам `ksr-search` D1 — это твой ресурс на бесплатном тарифе, не блокирует другие проекты)

## Оценка времени

- D1 setup + миграция данных: 1-2 ч
- Worker (search + LLM + feedback): 2-3 ч
- Pages frontend (адаптация): 30 мин
- Тестирование: 1 ч

Итого ~5-7 часов чистой работы. Лучше делать одной сессией.
