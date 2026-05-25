"""
KSR semantic search web app with LLM query expansion + batch mode.
Run:  python app.py    →   http://127.0.0.1:5000/
"""
import os
import re
import json
import sqlite3
import pickle
import time
import threading
from datetime import datetime, timezone
import numpy as np
import httpx
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "ksr.sqlite")
INDEX_PATH = os.path.join(HERE, "ksr_index.pkl")
ENV_FILE = "C:/00.CLAUDE CODE/engineers_apps/003-review-pipeline/.env"

if not (os.path.exists(DB_PATH) and os.path.exists(INDEX_PATH)):
    raise SystemExit("Index not built yet. Run: python build_index.py <xlsx>")


# ---------- env / API keys ----------
def _load_env_file(path: str):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file(ENV_FILE)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"


# ---------- load index ----------
print("Loading index...")
with open(INDEX_PATH, "rb") as f:
    idx = pickle.load(f)
WORD_VEC = idx["word_vec"]
CHAR_VEC = idx["char_vec"]
WORD_MAT = idx["word_mat"]
CHAR_MAT = idx["char_mat"]
N = idx["n_items"]

_conn = sqlite3.connect(DB_PATH)
_cur = _conn.cursor()
_cur.execute("SELECT id, sheet FROM items ORDER BY id")
SHEETS = np.empty(N, dtype=object)
for _id, _s in _cur.fetchall():
    SHEETS[_id] = _s

# Ensure the examples / feedback table exists (learning storage).
_cur.execute("""
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    query_norm TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    source TEXT NOT NULL,         -- 'click' | 'file:<name>' | 'manual'
    weight REAL DEFAULT 1.0,
    added_at TEXT NOT NULL,
    FOREIGN KEY(item_id) REFERENCES items(id)
)
""")
_cur.execute("CREATE INDEX IF NOT EXISTS idx_examples_norm ON examples(query_norm)")
_cur.execute("CREATE INDEX IF NOT EXISTS idx_examples_item ON examples(item_id)")
_conn.commit()
_conn.close()
print(f"  {N:,} items · Groq: {'on' if GROQ_API_KEY else 'off'} · Gemini fallback: {'on' if GEMINI_API_KEY else 'off'}")


# ---------- learning: examples store ----------
EXAMPLES: list = []          # [{"id","query","norm","item_id","weight"}]
EXAMPLES_MAT_W = None        # word-ngram sparse matrix (handles full-phrase matches)
EXAMPLES_MAT_C = None        # char-ngram sparse matrix (handles unseen abbreviations like "БДЛ")
EXAMPLES_DIRTY = True
EXAMPLES_LOCK = threading.Lock()


def _reload_examples():
    """Reload examples from SQLite and rebuild both TF-IDF row matrices.
    We need char-ngrams in addition to word-ngrams because user queries often
    contain abbreviations/codes (БДЛ-1, ВВГ, КТП) that aren't in WORD_VEC vocab
    (min_df=2 means a token must appear in ≥2 items to be a word-vocab entry)."""
    global EXAMPLES, EXAMPLES_MAT_W, EXAMPLES_MAT_C, EXAMPLES_DIRTY
    with EXAMPLES_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, query_text, query_norm, item_id, weight FROM examples")
        EXAMPLES = [
            {"id": r[0], "query": r[1], "norm": r[2], "item_id": r[3], "weight": r[4]}
            for r in cur.fetchall()
        ]
        conn.close()
        if EXAMPLES:
            norms = [e["norm"] for e in EXAMPLES]
            EXAMPLES_MAT_W = WORD_VEC.transform(norms)
            EXAMPLES_MAT_C = CHAR_VEC.transform(norms)
        else:
            EXAMPLES_MAT_W = None
            EXAMPLES_MAT_C = None
        EXAMPLES_DIRTY = False


def add_example(query: str, item_id: int, source: str = "click", weight: float = 1.0):
    """Persist one (query → item) pair and mark cache dirty."""
    global EXAMPLES_DIRTY
    qn = normalize(query)
    if not qn or not (0 <= item_id < N):
        return False
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO examples(query_text, query_norm, item_id, source, weight, added_at) "
        "VALUES(?,?,?,?,?,?)",
        (query, qn, item_id, source, weight,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    with EXAMPLES_LOCK:
        EXAMPLES_DIRTY = True
    return True


def get_example_boosts(query: str, min_sim: float = 0.30, boost_per: float = 0.40,
                       max_examples: int = 8) -> dict:
    """Return {item_id: extra_score} computed from past labeled examples.
    Most-similar past queries push their item_id up the rank.

    Matching uses 0.4·word + 0.6·char cosine similarity over example norms;
    exact-norm matches additionally get a hard +boost_per bonus so re-typing
    the same query always surfaces the previously selected item."""
    global EXAMPLES_DIRTY
    if EXAMPLES_DIRTY:
        _reload_examples()
    if not EXAMPLES or EXAMPLES_MAT_C is None:
        return {}
    qn = normalize(query)
    if not qn:
        return {}
    q_w = WORD_VEC.transform([qn])
    q_c = CHAR_VEC.transform([qn])
    sims_w = (EXAMPLES_MAT_W @ q_w.T).toarray().ravel()
    sims_c = (EXAMPLES_MAT_C @ q_c.T).toarray().ravel()
    sims = 0.4 * sims_w + 0.6 * sims_c

    boosts = {}
    # exact-match guarantees: if a past example had identical normalized query,
    # boost its item regardless of cosine score (handles short/abbrev queries
    # like "БДЛ-1" where vocab coverage is weak).
    for i, ex in enumerate(EXAMPLES):
        if ex["norm"] == qn:
            bid = int(ex["item_id"])
            contrib = boost_per * float(ex["weight"])
            if contrib > boosts.get(bid, 0.0):
                boosts[bid] = contrib

    order = np.argsort(-sims)
    used = 0
    for idx in order:
        sim = float(sims[idx])
        if sim < min_sim or used >= max_examples:
            break
        ex = EXAMPLES[idx]
        bid = int(ex["item_id"])
        contrib = boost_per * sim * float(ex["weight"])
        if contrib > boosts.get(bid, 0.0):
            boosts[bid] = contrib
        used += 1
    return boosts


# ---------- text utils ----------
def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9.,х/\-+\s]", " ", s)
    s = re.sub(r"(\d+),(\d+)", r"\1.\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- LLM expansion ----------
EXPAND_SYSTEM_PROMPT = """Ты — инженер-сметчик, эксперт по российскому Классификатору строительных ресурсов (КСР, ФССЦ).

Тебе дают позицию из сметы или из ведомости работ (часто это сокращение/аббревиатура/жаргон, иногда — название РАБОТЫ типа «ретрофит/замена/монтаж X»). Твоя задача — выдать 2–4 варианта поискового запроса, по которым нужный *предмет* (материал, изделие, оборудование) можно найти в классификаторе.

КРИТИЧНЫЕ ПРАВИЛА:
1. КСР — это справочник ПРЕДМЕТОВ, а не работ. Если на входе работа («ретрофит X», «замена Y на Z», «монтаж W»), извлекай и расширяй ПРЕДМЕТ (Y или Z, W), а ГЛАГОЛ-РАБОТУ выбрасывай. Не пиши «ретрофит/замена/монтаж» в выходе.
2. Расшифровывай аббревиатуры на канонические термины из ГОСТ/СНиП и заводской номенклатуры.
3. Сохраняй ВСЕ числа, марки, сечения, габариты, напряжения, токи, классы.
4. Один вариант — одна короткая формулировка. Разделяй " | ". Без пояснений, без нумерации.

СЛОВАРЬ ВЫСОКОВОЛЬТНОГО ОБОРУДОВАНИЯ (используй эти канонические термины):
- ВЭ = выключатель элегазовый
- ВВ = выключатель вакуумный
- ВМ = выключатель масляный
- ВН = выключатель нагрузки
- ВВВ = выключатель воздушный высоковольтный
- БВ / блок выключателя / ячейка выключателя = шкаф КРУ с выключателем / ячейка КРУ
- БВВ / блок выключателя ввода = ячейка ввода КРУ
- БСВ / блок секционного выключателя = секционная ячейка КРУ
- БЛВ / блок линейного выключателя = линейная ячейка КРУ
- КРУ = комплектное распределительное устройство (внутренней установки)
- КРУЭ = комплектное распределительное устройство элегазовое
- КСО = камера сборная одностороннего обслуживания
- КТП / КТПН = комплектная трансформаторная подстанция (наружной установки)
- ТТ = трансформатор тока
- ТН / ТНКИ = трансформатор напряжения
- ОПН = ограничитель перенапряжений нелинейный
- Р / РНДЗ / РЛНД = разъединитель
- ШВЗПС / ШНЭ / ШУ / шкаф РЗА = шкаф релейной защиты и автоматики
- БМРЗ / БЭМП / шкаф микропроцессорной защиты = терминал/блок микропроцессорной релейной защиты
- ЯКНО = ячейка коммутации напряжения объединённая
- ССПИ / СОПТ = система сбора и передачи информации / система оперативного постоянного тока
- ВЛ = воздушная линия электропередачи
- КЛ = кабельная линия
- Iном / In / In ном = номинальный ток
- Uном = номинальное напряжение

ПРИМЕРЫ:

Вход: БДЛ-1
Выход: плита дорожная железобетонная БДЛ-1 | плита железобетонная для дорожного покрытия | плита бетонная дорожная сборная

Вход: ВВГнг(А)-LS 3х2,5
Выход: кабель силовой с медными жилами ВВГнг(А)-LS 3х2,5 | кабель силовой не распространяющий горение с пониженным дымовыделением 3х2,5 | кабель ВВГнг-LS 3х2,5

Вход: Ретрофит ячейки ввода (замена ВЭ на элемент с коммутационным модулем: Iном=3150 А)
Выход: выключатель элегазовый 3150 А для ячейки ввода КРУ | ячейка КРУ ввода с элегазовым выключателем номинальный ток 3150 А | шкаф КРУ выкатной с элегазовым выключателем 3150 А

Вход: Блок секционного выключателя 35 кВ
Выход: ячейка КРУ секционного выключателя 35 кВ | шкаф КРУ секционный с вакуумным выключателем 35 кВ | выключатель вакуумный 35 кВ секционный

Вход: Блок выключателя 35 кВ линии
Выход: ячейка КРУ линейного выключателя 35 кВ | шкаф КРУ с вакуумным выключателем 35 кВ линейный | выключатель вакуумный 35 кВ для отходящей линии

Вход: Ретрофит ячейки: замена релейного шкафа
Выход: шкаф релейной защиты и автоматики микропроцессорный | шкаф РЗА с терминалом БМРЗ для ячейки КРУ | терминал микропроцессорной защиты ячейки КРУ"""


RERANK_SYSTEM_PROMPT = """Ты — инженер-сметчик, эксперт по российскому Классификатору строительных ресурсов (КСР).

Тебе дают исходный запрос и пронумерованный список кандидатов из классификатора. Оцени, насколько каждый кандидат подходит под запрос.

Шкала:
3 — точное или практически точное совпадение (тот же предмет, те же параметры, или очень близкая модель/типоразмер).
2 — тот же класс предметов, но другие параметры (например, выключатель того же типа, но другой ток/напряжение; кабель того же типа, но другое сечение).
1 — связано по области, но другой класс предметов (например, для выключателя — релейная защита; для кабеля — кабельный лоток).
0 — совершенно нерелевантно, другой предмет/материал (асфальт для выключателя, провод для шкафа, и т.п.).

ВАЖНО: если запрос про электрооборудование, а кандидат — строительные материалы (бетон, асфальт, плиты, грунт), это всегда 0.

Верни ТОЛЬКО JSON-массив целых чисел длины ровно N, без пояснений, без markdown, без кода. Пример: [3,2,0,0,1,2,0,3,...]"""


def _call_groq(query: str, timeout: float = 12.0) -> tuple[str | None, str | None]:
    """Returns (text, error). On 429 retries once with Retry-After before bailing,
    so Gemini fallback can take over fast for batch traffic."""
    if not GROQ_API_KEY:
        return None, "no_key"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": EXPAND_SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ],
        "temperature": 0.1,
        "max_tokens": 220,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for attempt in range(2):
        try:
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            if r.status_code == 429:
                if attempt == 0:
                    ra = r.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 1.5
                    if wait <= 3.0:  # short wait only — otherwise fall through to Gemini
                        time.sleep(wait)
                        continue
                return None, "groq_429"
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip(), None
        except httpx.HTTPError as e:
            return None, f"groq_{type(e).__name__}: {e}"
    return None, "groq_429"


def _call_gemini(query: str, timeout: float = 15.0) -> tuple[str | None, str | None]:
    if not GEMINI_API_KEY:
        return None, "no_key"
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
    payload = {
        "systemInstruction": {"parts": [{"text": EXPAND_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": query.strip()}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
    }
    for attempt in range(2):
        try:
            r = httpx.post(url, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                return None, "gemini_429"
            r.raise_for_status()
            data = r.json()
            cands = data.get("candidates") or []
            if not cands:
                return None, "gemini_no_candidates"
            parts = cands[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                return None, "gemini_empty"
            return text, None
        except httpx.HTTPError as e:
            return None, f"gemini_{type(e).__name__}: {e}"
    return None, "gemini_429"


def _llm_chat(user_msg: str, system_prompt: str, timeout: float = 15.0) -> tuple[str | None, str | None, str | None]:
    """Generic chat: try Groq, fallback to Gemini. Returns (text, provider, error)."""
    # Groq
    if GROQ_API_KEY:
        try:
            for attempt in range(2):
                r = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 400,
                    },
                    timeout=timeout,
                )
                if r.status_code == 429:
                    if attempt == 0:
                        ra = r.headers.get("retry-after")
                        wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 1.5
                        if wait <= 3.0:
                            time.sleep(wait); continue
                    break  # fall through to Gemini
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip(), "groq", None
        except httpx.HTTPError as e:
            pass  # try Gemini
    # Gemini
    if GEMINI_API_KEY:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
            r = httpx.post(url, json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 500},
            }, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            cands = data.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return text, "gemini", None
            return None, None, "gemini_empty"
        except httpx.HTTPError as e:
            return None, None, f"gemini_{type(e).__name__}: {e}"
    return None, None, "no_provider"


def rerank_with_llm(query: str, candidates: list, keep_min_score: int = 1) -> dict:
    """Re-rank TF-IDF candidates with LLM relevance scoring (0–3).
    Returns {'reranked': list, 'llm_scores': list, 'no_strong_match': bool,
             'error': str|None, 'provider': str|None}.
    keep_min_score=1: drop only obvious junk (0). 2: stricter.
    If LLM rates ALL candidates as 0 → reranked is empty + no_strong_match flag set
    (so UI shows 'nothing found' rather than misleading 0-rated junk)."""
    out = {"reranked": candidates, "llm_scores": [], "no_strong_match": False,
           "error": None, "provider": None}
    if not candidates or len(candidates) < 2:
        return out

    names = "\n".join(f"{i+1}. {c['name']}" for i, c in enumerate(candidates))
    user_msg = f'Запрос: "{query}"\n\nКандидаты ({len(candidates)} шт):\n{names}\n\nВерни JSON-массив из {len(candidates)} чисел 0–3.'

    text, provider, err = _llm_chat(user_msg, RERANK_SYSTEM_PROMPT, timeout=20.0)
    out["provider"] = provider
    if text is None:
        out["error"] = err
        return out

    # Extract first JSON array of integers from response
    m = re.search(r"\[\s*[\d,\s]+\]", text)
    if not m:
        out["error"] = f"no_json_array (got: {text[:120]})"
        return out
    try:
        scores = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        out["error"] = f"json_parse_err: {e}"
        return out
    if len(scores) != len(candidates):
        # pad or trim to match — best-effort recovery
        if len(scores) < len(candidates):
            scores = scores + [0] * (len(candidates) - len(scores))
        else:
            scores = scores[:len(candidates)]

    out["llm_scores"] = [int(s) for s in scores]
    enriched = []
    for c, s in zip(candidates, out["llm_scores"]):
        c2 = dict(c)
        c2["llm_score"] = int(s)
        enriched.append(c2)

    max_score = max(out["llm_scores"]) if out["llm_scores"] else 0
    # Always return candidates (sorted by AI score desc, TF-IDF desc).
    # `no_strong_match` flag is informational for the UI.
    enriched.sort(key=lambda x: (-x["llm_score"], -x["score"]))
    out["reranked"] = enriched
    out["no_strong_match"] = (max_score < keep_min_score)
    return out


def expand_query_llm(query: str) -> dict:
    """Returns {'variants': [...], 'llm_raw': str|None, 'error': str|None, 'provider': str|None}"""
    out = {"variants": [query], "llm_raw": None, "error": None, "provider": None}
    if not query.strip():
        return out

    text, err = _call_groq(query)
    provider = "groq"
    if text is None and GEMINI_API_KEY:
        text, err2 = _call_gemini(query)
        provider = "gemini"
        if text is None:
            out["error"] = f"{err} → {err2}"
            return out
    if text is None:
        out["error"] = err or "no_provider"
        return out

    out["llm_raw"] = text
    out["provider"] = provider
    first = text.splitlines()[0]
    parts = [p.strip(" -*\"'") for p in first.split("|") if p.strip()]
    seen = {normalize(query)}
    for p in parts:
        if normalize(p) not in seen and len(p) > 2:
            out["variants"].append(p)
            seen.add(normalize(p))
    return out


# ---------- scoring ----------
def _score_one(q: str, sheet: str = None) -> np.ndarray:
    qn = normalize(q)
    if not qn:
        return np.full(N, -1.0)
    qw = WORD_VEC.transform([qn])
    qc = CHAR_VEC.transform([qn])
    sw = (WORD_MAT @ qw.T).toarray().ravel()
    sc = (CHAR_MAT @ qc.T).toarray().ravel()
    score = 0.65 * sw + 0.35 * sc
    if sheet:
        mask = SHEETS != sheet
        if mask.any():
            score = score.copy()
            score[mask] = -1.0
    return score


def search_variants(variants, limit=30, sheet=None, original_query: str = None):
    """Search multiple query variants; per-item score = max across variants.
    `original_query` is used to fetch labelled-example boosts (variants are
    LLM-paraphrased so don't match the user's wording for the example index)."""
    combined = None
    for q in variants:
        s = _score_one(q, sheet)
        combined = s if combined is None else np.maximum(combined, s)
    if combined is None:
        return []

    # Boost items the user (or an imported file) has previously labelled for
    # similar queries. Always seeded from the human-written original query.
    boosts = get_example_boosts(original_query or (variants[0] if variants else ""))
    boosted_ids = set()
    if boosts:
        combined = combined.copy()
        for item_id, b in boosts.items():
            if 0 <= item_id < N and combined[item_id] >= 0:
                combined[item_id] += b
                boosted_ids.add(item_id)

    # digit tokens from any variant (boost when literal numbers match)
    all_text = " ".join(normalize(v) for v in variants)
    tokens = [t for t in re.findall(r"[a-zа-я0-9.+\-/х]+", all_text) if len(t) >= 2]
    digit_tokens = list({t for t in tokens if any(ch.isdigit() for ch in t)})

    top_k = min(limit * 6, N)
    order = np.argpartition(-combined, top_k - 1)[:top_k]
    order = order[np.argsort(-combined[order])]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(order))
    cur.execute(
        f"SELECT id, sheet, code, name, unit, category FROM items WHERE id IN ({placeholders})",
        [int(i) for i in order],
    )
    rows = {r[0]: r for r in cur.fetchall()}
    conn.close()

    results = []
    for i in order:
        r = rows.get(int(i))
        if not r:
            continue
        s = float(combined[i])
        if s < 0:
            continue
        if digit_tokens:
            name_norm = normalize(r[3])
            hits = sum(1 for t in digit_tokens if t in name_norm)
            s += 0.02 * hits
        results.append({
            "id": r[0], "sheet": r[1], "code": r[2], "name": r[3],
            "unit": r[4], "category": r[5], "score": round(s, 4),
            "from_examples": int(r[0]) in boosted_ids,
        })
        if len(results) >= limit:
            break
    results.sort(key=lambda x: -x["score"])
    return results


# ---------- Flask ----------
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT sheet, COUNT(*) FROM items GROUP BY sheet")
    s = dict(cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM examples")
    n_examples = cur.fetchone()[0]
    cur.execute("SELECT source, COUNT(*) FROM examples GROUP BY source")
    by_source = dict(cur.fetchall())
    conn.close()
    providers = []
    if GROQ_API_KEY: providers.append(f"Groq {GROQ_MODEL}")
    if GEMINI_API_KEY: providers.append(f"Gemini {GEMINI_MODEL} (fallback)")
    return jsonify({
        "total": N, "by_sheet": s,
        "llm_enabled": bool(GROQ_API_KEY or GEMINI_API_KEY),
        "llm_providers": providers,
        "examples": n_examples,
        "examples_by_source": by_source,
    })


@app.post("/api/feedback")
def api_feedback():
    """Record a user click: query → chosen item_id. Used for online learning."""
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    try:
        item_id = int(data.get("item_id"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad item_id"}), 400
    if not query:
        return jsonify({"ok": False, "error": "empty query"}), 400
    weight = float(data.get("weight") or 1.0)
    ok = add_example(query, item_id, source="click", weight=weight)
    return jsonify({"ok": ok, "examples_total": len(EXAMPLES) + (1 if ok else 0)})


@app.post("/api/ingest")
def api_ingest():
    """Bulk-load labelled examples from JSON.
    Body: {"examples": [{"query": "...", "code": "..."} | {"query": "...", "item_id": N}, ...],
           "source": "file:budget.xlsx"}
    Looks up item_id by code if not given."""
    data = request.get_json(force=True) or {}
    rows = data.get("examples") or []
    source = data.get("source") or "manual"
    if not isinstance(rows, list):
        return jsonify({"ok": False, "error": "examples must be a list"}), 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    added, skipped = 0, []
    for i, row in enumerate(rows):
        q = (row.get("query") or "").strip()
        if not q:
            skipped.append({"idx": i, "reason": "empty query"}); continue
        item_id = row.get("item_id")
        if item_id is None and row.get("code"):
            cur.execute("SELECT id FROM items WHERE code=? LIMIT 1", (row["code"],))
            r = cur.fetchone()
            if not r:
                skipped.append({"idx": i, "reason": f"code not found: {row['code']}"}); continue
            item_id = r[0]
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            skipped.append({"idx": i, "reason": "bad item_id"}); continue
        if not (0 <= item_id < N):
            skipped.append({"idx": i, "reason": f"item_id out of range: {item_id}"}); continue
        weight = float(row.get("weight") or 1.0)
        cur.execute(
            "INSERT INTO examples(query_text, query_norm, item_id, source, weight, added_at) "
            "VALUES(?,?,?,?,?,?)",
            (q, normalize(q), item_id, source, weight,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        added += 1
    conn.commit()
    conn.close()
    global EXAMPLES_DIRTY
    with EXAMPLES_LOCK:
        EXAMPLES_DIRTY = True
    return jsonify({"ok": True, "added": added, "skipped": skipped[:20], "skipped_total": len(skipped)})


@app.post("/api/expand")
def api_expand():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    return jsonify(expand_query_llm(query))


def _pipeline(query: str, limit: int, sheet, use_llm: bool, use_rerank: bool):
    """Full pipeline for one query: expand → TF-IDF (+ examples boost) → optional LLM rerank.
    Returns dict with variants, results, llm/rerank diagnostics."""
    variants = [query]
    exp_err = None
    if use_llm:
        exp = expand_query_llm(query)
        variants = exp["variants"]
        exp_err = exp["error"]

    pool_size = max(limit * 4, 20) if use_rerank else limit
    pool = search_variants(variants, limit=pool_size, sheet=sheet, original_query=query)

    rerank_err = None
    no_strong = False
    if use_rerank and pool:
        rr = rerank_with_llm(query, pool, keep_min_score=1)
        rerank_err = rr["error"]
        no_strong = rr["no_strong_match"]
        pool = rr["reranked"]

    return {
        "variants": variants,
        "results": pool[:limit],
        "llm_error": exp_err,
        "rerank_error": rerank_err,
        "no_strong_match": no_strong,
    }


@app.post("/api/search")
def api_search():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    limit = int(data.get("limit") or 20)
    sheet = data.get("sheet") or None
    use_llm = bool(data.get("expand"))
    use_rerank = bool(data.get("rerank"))
    if not query:
        return jsonify({"results": [], "variants": []})

    pipe = _pipeline(query, limit, sheet, use_llm, use_rerank)
    return jsonify({
        "query": query,
        "variants": pipe["variants"],
        "llm": {"used": use_llm, "error": pipe["llm_error"]},
        "rerank": {"used": use_rerank, "error": pipe["rerank_error"],
                   "no_strong_match": pipe["no_strong_match"]},
        "results": pipe["results"],
    })


@app.post("/api/batch")
def api_batch():
    data = request.get_json(force=True) or {}
    items = data.get("items") or []
    if isinstance(items, str):
        items = [ln.strip() for ln in items.splitlines() if ln.strip()]
    items = [str(x).strip() for x in items if str(x).strip()][:2000]
    limit = int(data.get("limit") or 5)
    sheet = data.get("sheet") or None
    use_llm = bool(data.get("expand"))
    use_rerank = bool(data.get("rerank"))

    def process(q):
        return _pipeline(q, limit, sheet, use_llm, use_rerank)

    out = []
    if items:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for q, pipe in zip(items, ex.map(process, items)):
                out.append({
                    "query": q,
                    "variants": pipe["variants"],
                    "llm_error": pipe["llm_error"],
                    "rerank_error": pipe["rerank_error"],
                    "no_strong_match": pipe["no_strong_match"],
                    "results": pipe["results"],
                })
    return jsonify({
        "items": out, "llm_used": use_llm, "rerank_used": use_rerank,
        "n": len(items),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
