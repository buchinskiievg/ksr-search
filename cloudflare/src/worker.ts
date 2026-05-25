/**
 * KSR semantic search — Cloudflare Worker port.
 *
 * Pipeline (matches the Python reference in ../../app.py):
 *   1. AI expand    — Groq Llama 3.3 70b primary, Gemini 2.0 Flash fallback on 429
 *   2. D1 FTS5      — BM25 retrieval over items_fts
 *   3. AI re-rank   — LLM scores each candidate 0–3
 *   4. Examples boost — past labelled (query → item) pairs nudge their items up
 *
 * Endpoints:
 *   GET  /api/stats      — counts + LLM provider info
 *   POST /api/expand     — { query } → { variants, error }
 *   POST /api/search     — { query, sheet?, limit?, expand?, rerank? } → results
 *   POST /api/batch      — { items[], sheet?, limit?, expand?, rerank? } → results per item
 *   POST /api/feedback   — { query, item_id } → record a click
 *   POST /api/ingest     — { examples[{ query, code|item_id }], source } → bulk import
 *   GET  /*              — static assets via ASSETS binding
 */

import { EXPAND_SYSTEM_PROMPT, RERANK_SYSTEM_PROMPT } from "./prompts";

export interface Env {
  DB: D1Database;
  ASSETS: Fetcher;
  AI: Ai;
  GROQ_API_KEY?: string;
  GEMINI_API_KEY?: string;
  GROQ_MODEL: string;
  GEMINI_MODEL: string;
}

const CF_AI_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

// Cloudflare Workers AI: runs the model in the Worker runtime, no outbound
// HTTP call. Primary provider because Groq blocks our IP (403) and Gemini
// refuses our datacenter region (400). Workers AI doesn't have those
// limitations and uses ~10–50 neurons per short chat (free tier covers it).
async function callCfAi(env: Env, system: string, user: string): Promise<{ text: string | null; error: string | null }> {
  try {
    const result: any = await env.AI.run(CF_AI_MODEL as any, {
      messages: [
        { role: "system", content: system },
        { role: "user", content: user.trim() },
      ],
      max_tokens: 500,
      temperature: 0.0,
    });
    const text = (result?.response || "").toString().trim();
    if (!text) return { text: null, error: "cfai_empty" };
    return { text, error: null };
  } catch (e: any) {
    return { text: null, error: `cfai_${e?.message || e?.name || "err"}`.slice(0, 80) };
  }
}

type Item = {
  id: number;
  sheet: string;
  code: string;
  name: string;
  unit: string | null;
  category: string | null;
  score: number;
  llm_score?: number | null;
  from_examples?: boolean;
};

// ---------- text utils ----------
function normalize(s: string): string {
  if (!s) return "";
  return s.toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[^a-zа-я0-9.,х/\-+\s]/g, " ")
    .replace(/(\d+),(\d+)/g, "$1.$2")
    .replace(/\s+/g, " ")
    .trim();
}

// Tokenise a normalised query into FTS5-safe terms.
// FTS5 query syntax has special chars (", *, etc.) — we quote each
// token and OR them so any keyword in any variant boosts the score.
function ftsQuery(text: string, maxTokens = 30): string {
  const tokens = (text.toLowerCase().replace(/ё/g, "е")
    .match(/[a-zа-я0-9]{2,}/g) || [])
    .slice(0, maxTokens);
  if (!tokens.length) return "";
  // Quote and OR. Dedup to avoid degenerate queries.
  const uniq = Array.from(new Set(tokens));
  return uniq.map(t => `"${t.replace(/"/g, "")}"`).join(" OR ");
}

// ---------- LLM (Groq primary, Gemini fallback) ----------
async function callGroq(
  env: Env, system: string, user: string, signal?: AbortSignal
): Promise<{ text: string | null; error: string | null }> {
  if (!env.GROQ_API_KEY) return { text: null, error: "no_groq_key" };
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: env.GROQ_MODEL,
          messages: [
            { role: "system", content: system },
            { role: "user", content: user },
          ],
          temperature: 0.0,
          max_tokens: 400,
        }),
        signal,
      });
      if (r.status === 429) {
        if (attempt === 0) {
          const ra = parseFloat(r.headers.get("retry-after") || "1.5");
          if (ra <= 3.0) {
            await new Promise(res => setTimeout(res, ra * 1000));
            continue;
          }
        }
        return { text: null, error: "groq_429" };
      }
      if (!r.ok) {
        const body = await r.text().catch(() => "");
        console.log(`groq ${r.status}: ${body.slice(0, 300)}`);
        return { text: null, error: `groq_${r.status}` };
      }
      const data: any = await r.json();
      return { text: (data.choices?.[0]?.message?.content || "").trim(), error: null };
    } catch (e: any) {
      return { text: null, error: `groq_${e?.name || "err"}` };
    }
  }
  return { text: null, error: "groq_429" };
}

async function callGemini(
  env: Env, system: string, user: string, signal?: AbortSignal
): Promise<{ text: string | null; error: string | null }> {
  if (!env.GEMINI_API_KEY) return { text: null, error: "no_gemini_key" };
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${env.GEMINI_MODEL}:generateContent?key=${env.GEMINI_API_KEY}`;
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig: { temperature: 0.0, maxOutputTokens: 500 },
      }),
      signal,
    });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      console.log(`gemini ${r.status}: ${body.slice(0, 400)}`);
      return { text: null, error: `gemini_${r.status}` };
    }
    const data: any = await r.json();
    const cand = data.candidates?.[0];
    const finish = cand?.finishReason;
    const parts = cand?.content?.parts || [];
    const text = parts.map((p: any) => p.text || "").join("").trim();
    if (!text) {
      console.log(`gemini empty: finish=${finish} safety=${JSON.stringify(cand?.safetyRatings || []).slice(0,200)}`);
      return { text: null, error: `gemini_empty_${finish || "noreason"}` };
    }
    return { text, error: null };
  } catch (e: any) {
    return { text: null, error: `gemini_${e?.name || "err"}` };
  }
}

async function llmChat(env: Env, system: string, user: string): Promise<{ text: string | null; provider: string | null; error: string | null }> {
  // CF AI primary — only provider that works reliably from a Worker.
  const c = await callCfAi(env, system, user);
  if (c.text !== null) return { text: c.text, provider: "cf-ai", error: null };
  // External fallbacks — usually fail from the Worker but kept just in case
  // a future Cloudflare colo gets allowlisted.
  const a = await callGroq(env, system, user);
  if (a.text !== null) return { text: a.text, provider: "groq", error: null };
  const b = await callGemini(env, system, user);
  if (b.text !== null) return { text: b.text, provider: "gemini", error: null };
  return { text: null, provider: null, error: `${c.error} → ${a.error} → ${b.error}` };
}

// ---------- expand ----------
async function expandQuery(env: Env, query: string): Promise<{ variants: string[]; error: string | null; provider: string | null }> {
  const out = { variants: [query], error: null as string | null, provider: null as string | null };
  if (!query.trim()) return out;
  if (!env.GROQ_API_KEY && !env.GEMINI_API_KEY) return out;

  const { text, provider, error } = await llmChat(env, EXPAND_SYSTEM_PROMPT, query.trim());
  if (text === null) { out.error = error; return out; }
  out.provider = provider;

  const first = text.split("\n")[0] || "";
  const parts = first.split("|").map(p => p.trim().replace(/^[\s\-*"']+|[\s\-*"']+$/g, "")).filter(Boolean);
  const seen = new Set([normalize(query)]);
  for (const p of parts) {
    if (p.length > 2 && !seen.has(normalize(p))) {
      out.variants.push(p);
      seen.add(normalize(p));
    }
  }
  return out;
}

// ---------- D1 search ----------
// КСР sheets contain both short codes (NN.N.NN.NN-XXXX, ≤ 4 dots — the
// canonical КСР reference) and ОКПД2-prefixed long codes (NN.NN.NN.NNN.<short>-XXX-000,
// ≥ 5 dots — duplicates of the short ones, just decorated with an external
// classifier prefix). We want only the short rows in user-facing output.
const SHORT_CODE_FILTER = "(LENGTH(items.code) - LENGTH(REPLACE(items.code, '.', ''))) <= 4";

async function searchFts(env: Env, variants: string[], sheet: string | null, poolSize: number): Promise<Item[]> {
  const allText = variants.map(v => normalize(v)).join(" ");
  const fts = ftsQuery(allText);
  if (!fts) return [];

  let sql: string;
  let bindings: any[];
  if (sheet) {
    sql = `SELECT items.id, items.sheet, items.code, items.name, items.unit, items.category,
                  -bm25(items_fts) AS score
           FROM items_fts
           JOIN items ON items.id = items_fts.rowid
           WHERE items_fts MATCH ? AND items.sheet = ? AND ${SHORT_CODE_FILTER}
           ORDER BY bm25(items_fts) ASC
           LIMIT ?`;
    bindings = [fts, sheet, poolSize];
  } else {
    sql = `SELECT items.id, items.sheet, items.code, items.name, items.unit, items.category,
                  -bm25(items_fts) AS score
           FROM items_fts
           JOIN items ON items.id = items_fts.rowid
           WHERE items_fts MATCH ? AND ${SHORT_CODE_FILTER}
           ORDER BY bm25(items_fts) ASC
           LIMIT ?`;
    bindings = [fts, poolSize];
  }
  const { results } = await env.DB.prepare(sql).bind(...bindings).all<any>();
  // Normalise score to roughly 0..1 range based on top score, so it
  // composes well with example-boost numbers (which are 0..0.5-ish).
  const items: Item[] = (results || []).map(r => ({
    id: r.id, sheet: r.sheet, code: r.code, name: r.name, unit: r.unit,
    category: r.category, score: Number(r.score),
  }));
  if (items.length === 0) return items;
  const maxRaw = Math.max(...items.map(i => i.score));
  if (maxRaw > 0) {
    for (const it of items) it.score = it.score / maxRaw;
  }
  return items;
}

// ---------- examples boost ----------
// Returns:
//   itemBoosts:  Map<item_id, boost>     direct hits (exact + FTS-fuzzy match)
//   groupBoosts: Map<code_prefix, boost> fan-out by КСР group (e.g. 59.1.25.03-%)
// The pipeline merge step applies both — direct on item id, fan-out on
// item.code.startsWith(prefix + "-"). This handles the common case where
// bulk-ingest labels by group code (4 segments, no -NNNN suffix), so the
// actual matching specific item in search results would otherwise differ
// from the ingest's "first-item-in-group" placeholder.
type Boosts = { itemBoosts: Map<number, number>; groupBoosts: Map<string, number> };

async function exampleBoosts(env: Env, query: string): Promise<Boosts> {
  const empty: Boosts = { itemBoosts: new Map(), groupBoosts: new Map() };
  const qn = normalize(query);
  if (!qn) return empty;
  const fts = ftsQuery(qn);
  if (!fts) return empty;

  // Exact-norm match: hard boost.
  const exactSql = `SELECT item_id, weight FROM examples WHERE query_norm = ? LIMIT 20`;
  const exact = await env.DB.prepare(exactSql).bind(qn).all<any>();

  // Fuzzy match via FTS5 BM25 on examples.query_norm.
  const fuzzySql = `SELECT examples.item_id, examples.weight, -bm25(examples_fts) AS sim
                    FROM examples_fts
                    JOIN examples ON examples.id = examples_fts.rowid
                    WHERE examples_fts MATCH ?
                    ORDER BY bm25(examples_fts) ASC
                    LIMIT 10`;
  let fuzzy: any = { results: [] };
  try {
    fuzzy = await env.DB.prepare(fuzzySql).bind(fts).all<any>();
  } catch { /* MATCH on empty index → no rows */ }

  const itemBoosts = new Map<number, number>();
  const BOOST_PER = 0.40;
  for (const r of exact.results || []) {
    const b = BOOST_PER * Number(r.weight);
    itemBoosts.set(r.item_id, Math.max(itemBoosts.get(r.item_id) || 0, b));
  }
  const fuzzyResults = (fuzzy.results || []) as any[];
  const maxSim = fuzzyResults.length ? Math.max(...fuzzyResults.map((r: any) => Number(r.sim))) : 0;
  for (const r of fuzzyResults) {
    if (maxSim <= 0) break;
    const normSim = Number(r.sim) / maxSim;
    if (normSim < 0.3) continue;
    const b = BOOST_PER * normSim * Number(r.weight);
    itemBoosts.set(r.item_id, Math.max(itemBoosts.get(r.item_id) || 0, b));
  }

  // Derive group prefixes from the directly-boosted item codes. The pipeline
  // applies these via startsWith() on each pool item's code — far cheaper and
  // more accurate than materialising all 3000+ siblings into the id map.
  const groupBoosts = new Map<string, number>();
  if (itemBoosts.size) {
    const directIds = [...itemBoosts.keys()];
    const placeholders = directIds.map(() => "?").join(",");
    const direct = await env.DB.prepare(
      `SELECT id, code FROM items WHERE id IN (${placeholders})`
    ).bind(...directIds).all<any>();
    const GROUP_BOOST = 0.25;
    for (const r of (direct.results || []) as any[]) {
      const m = String(r.code).match(/^(.+)-\d+$/);
      if (m) groupBoosts.set(m[1], Math.max(groupBoosts.get(m[1]) || 0, GROUP_BOOST));
    }
  }
  return { itemBoosts, groupBoosts };
}

// ---------- rerank ----------
async function rerank(env: Env, query: string, candidates: Item[]): Promise<{ scored: Item[]; noStrong: boolean; error: string | null; provider: string | null }> {
  const out = { scored: candidates, noStrong: false, error: null as string | null, provider: null as string | null };
  if (candidates.length < 2) return out;

  const names = candidates.map((c, i) => `${i + 1}. ${c.name}`).join("\n");
  const userMsg = `Запрос: "${query}"\n\nКандидаты (${candidates.length} шт):\n${names}\n\nВерни JSON-массив из ${candidates.length} чисел 0–3.`;

  const { text, provider, error } = await llmChat(env, RERANK_SYSTEM_PROMPT, userMsg);
  out.provider = provider;
  if (text === null) { out.error = error; return out; }
  const m = text.match(/\[\s*[\d,\s]+\]/);
  if (!m) { out.error = "no_json_array"; return out; }
  let scores: number[];
  try { scores = JSON.parse(m[0]); }
  catch { out.error = "json_parse_err"; return out; }

  if (scores.length < candidates.length) {
    scores = scores.concat(Array(candidates.length - scores.length).fill(0));
  } else if (scores.length > candidates.length) {
    scores = scores.slice(0, candidates.length);
  }
  const enriched = candidates.map((c, i) => ({ ...c, llm_score: Math.max(0, Math.min(3, scores[i] | 0)) }));
  enriched.sort((a, b) => (b.llm_score! - a.llm_score!) || (b.score - a.score));
  out.scored = enriched;
  out.noStrong = Math.max(...enriched.map(e => e.llm_score!)) < 1;
  return out;
}

// ---------- pipeline ----------
async function pipeline(
  env: Env, query: string, limit: number, sheet: string | null,
  useExpand: boolean, useRerank: boolean,
): Promise<{ variants: string[]; results: Item[]; llm_error: string | null; rerank_error: string | null; no_strong_match: boolean }> {
  let variants = [query];
  let llm_error: string | null = null;
  if (useExpand) {
    const exp = await expandQuery(env, query);
    variants = exp.variants;
    llm_error = exp.error;
  }

  const poolSize = useRerank ? Math.max(limit * 4, 20) : limit;
  let pool = await searchFts(env, variants, sheet, poolSize);

  // Apply example boosts based on the ORIGINAL query (human-typed wording).
  const { itemBoosts, groupBoosts } = await exampleBoosts(env, query);
  if ((itemBoosts.size || groupBoosts.size) && pool.length) {
    const idSet = new Set(pool.map(p => p.id));
    for (const it of pool) {
      const ib = itemBoosts.get(it.id);
      if (ib) { it.score += ib; it.from_examples = true; }
      // Group fan-out via code prefix
      for (const [pfx, gb] of groupBoosts) {
        if (it.code && it.code.startsWith(pfx + "-")) {
          it.score += gb;
          it.from_examples = true;
          break;
        }
      }
    }
    // Items boosted by exact id but missing from FTS pool — fetch and inject.
    const missing = [...itemBoosts.keys()].filter(id => !idSet.has(id));
    if (missing.length) {
      const placeholders = missing.map(() => "?").join(",");
      const { results } = await env.DB.prepare(
        `SELECT id, sheet, code, name, unit, category FROM items WHERE id IN (${placeholders}) AND ${SHORT_CODE_FILTER}`
      ).bind(...missing).all<any>();
      for (const r of (results || [])) {
        pool.push({
          id: r.id, sheet: r.sheet, code: r.code, name: r.name, unit: r.unit,
          category: r.category, score: itemBoosts.get(r.id) || 0, from_examples: true,
        });
      }
    }
    pool.sort((a, b) => b.score - a.score);
    pool = pool.slice(0, poolSize);
  }

  let rerank_error: string | null = null;
  let no_strong = false;
  if (useRerank && pool.length) {
    const rr = await rerank(env, query, pool);
    pool = rr.scored;
    rerank_error = rr.error;
    no_strong = rr.noStrong;
  }

  return {
    variants,
    results: pool.slice(0, limit),
    llm_error, rerank_error,
    no_strong_match: no_strong,
  };
}

// ---------- HTTP helpers ----------
function json(data: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: { "Content-Type": "application/json; charset=utf-8", ...(init?.headers || {}) },
  });
}

// ---------- main router ----------
export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname;

    try {
      if (path === "/api/stats" && req.method === "GET") {
        const total = await env.DB.prepare(`SELECT COUNT(*) as n FROM items WHERE ${SHORT_CODE_FILTER}`).first<any>();
        const bySheet = await env.DB.prepare(`SELECT sheet, COUNT(*) as n FROM items WHERE ${SHORT_CODE_FILTER} GROUP BY sheet`).all<any>();
        const examples = await env.DB.prepare("SELECT COUNT(*) as n FROM examples").first<any>();
        const bySource = await env.DB.prepare("SELECT source, COUNT(*) as n FROM examples GROUP BY source").all<any>();
        const providers: string[] = [`Cloudflare ${CF_AI_MODEL}`];
        if (env.GROQ_API_KEY) providers.push(`Groq ${env.GROQ_MODEL} (fallback)`);
        if (env.GEMINI_API_KEY) providers.push(`Gemini ${env.GEMINI_MODEL} (fallback)`);
        return json({
          total: total?.n || 0,
          by_sheet: Object.fromEntries((bySheet.results || []).map((r: any) => [r.sheet, r.n])),
          llm_enabled: true,
          llm_providers: providers,
          examples: examples?.n || 0,
          examples_by_source: Object.fromEntries((bySource.results || []).map((r: any) => [r.source, r.n])),
        });
      }

      if (path === "/api/expand" && req.method === "POST") {
        const body: any = await req.json().catch(() => ({}));
        const q = String(body.query || "").trim();
        return json(await expandQuery(env, q));
      }

      if (path === "/api/search" && req.method === "POST") {
        const body: any = await req.json().catch(() => ({}));
        const q = String(body.query || "").trim();
        if (!q) return json({ results: [], variants: [] });
        const limit = Math.max(1, Math.min(100, parseInt(body.limit ?? 20)));
        const sheet = body.sheet ? String(body.sheet) : null;
        const pipe = await pipeline(env, q, limit, sheet, !!body.expand, !!body.rerank);
        return json({
          query: q,
          variants: pipe.variants,
          llm: { used: !!body.expand, error: pipe.llm_error },
          rerank: { used: !!body.rerank, error: pipe.rerank_error, no_strong_match: pipe.no_strong_match },
          results: pipe.results,
        });
      }

      if (path === "/api/batch" && req.method === "POST") {
        const body: any = await req.json().catch(() => ({}));
        let items: string[] = [];
        if (typeof body.items === "string") items = body.items.split("\n").map((s: string) => s.trim()).filter(Boolean);
        else if (Array.isArray(body.items)) items = body.items.map((x: any) => String(x).trim()).filter(Boolean);
        items = items.slice(0, 2000);
        const limit = Math.max(1, Math.min(20, parseInt(body.limit ?? 5)));
        const sheet = body.sheet ? String(body.sheet) : null;
        const useExpand = !!body.expand, useRerank = !!body.rerank;

        // Concurrency 12 — matches the Python ThreadPoolExecutor config.
        const out: any[] = [];
        const queue = items.map((q, i) => ({ q, i }));
        const inFlight: Promise<void>[] = [];
        const limitN = 12;
        async function work() {
          while (queue.length) {
            const job = queue.shift();
            if (!job) break;
            const p = await pipeline(env, job.q, limit, sheet, useExpand, useRerank);
            out[job.i] = {
              query: job.q,
              variants: p.variants,
              llm_error: p.llm_error,
              rerank_error: p.rerank_error,
              no_strong_match: p.no_strong_match,
              results: p.results,
            };
          }
        }
        for (let k = 0; k < limitN; k++) inFlight.push(work());
        await Promise.all(inFlight);

        return json({ items: out, llm_used: useExpand, rerank_used: useRerank, n: items.length });
      }

      if (path === "/api/feedback" && req.method === "POST") {
        const body: any = await req.json().catch(() => ({}));
        const query = String(body.query || "").trim();
        const itemId = parseInt(body.item_id);
        if (!query || !Number.isFinite(itemId)) return json({ ok: false, error: "bad_input" }, { status: 400 });
        const weight = Number(body.weight) || 1.0;
        const exists = await env.DB.prepare("SELECT id FROM items WHERE id = ?").bind(itemId).first<any>();
        if (!exists) return json({ ok: false, error: "item not found" }, { status: 404 });
        await env.DB.prepare(
          "INSERT INTO examples(query_text, query_norm, item_id, source, weight, added_at) VALUES(?,?,?,?,?,?)"
        ).bind(query, normalize(query), itemId, "click", weight, new Date().toISOString()).run();
        const cnt = await env.DB.prepare("SELECT COUNT(*) AS n FROM examples").first<any>();
        return json({ ok: true, examples_total: cnt?.n || 0 });
      }

      if (path === "/api/ingest" && req.method === "POST") {
        const body: any = await req.json().catch(() => ({}));
        const rows = Array.isArray(body.examples) ? body.examples : [];
        const source = String(body.source || "manual");
        const now = new Date().toISOString();
        let added = 0;
        const skipped: any[] = [];
        const stmts: D1PreparedStatement[] = [];
        for (let i = 0; i < rows.length; i++) {
          const row = rows[i] || {};
          const q = String(row.query || "").trim();
          if (!q) { skipped.push({ idx: i, reason: "empty query" }); continue; }
          let itemId: number | null = null;
          if (row.item_id != null) itemId = parseInt(row.item_id);
          else if (row.code) {
            // Exact match first (NN.N.NN.NN-NNNN). If that fails, treat the
            // input as a group prefix (NN.N.NN.NN) and pick the first item
            // under that group. This is how conjunctural-analysis files label
            // — by КСР group, not by specific item.
            let r = await env.DB.prepare(
              `SELECT id FROM items WHERE code = ? AND ${SHORT_CODE_FILTER} LIMIT 1`
            ).bind(String(row.code)).first<any>();
            if (!r) {
              r = await env.DB.prepare(
                `SELECT id FROM items WHERE code LIKE ? AND ${SHORT_CODE_FILTER} ORDER BY code LIMIT 1`
              ).bind(String(row.code) + "-%").first<any>();
            }
            if (!r) { skipped.push({ idx: i, reason: `code not found: ${row.code}` }); continue; }
            itemId = r.id;
          }
          if (!Number.isFinite(itemId as number)) { skipped.push({ idx: i, reason: "bad item_id" }); continue; }
          const weight = Number(row.weight) || 1.0;
          stmts.push(env.DB.prepare(
            "INSERT INTO examples(query_text, query_norm, item_id, source, weight, added_at) VALUES(?,?,?,?,?,?)"
          ).bind(q, normalize(q), itemId, source, weight, now));
          added++;
        }
        if (stmts.length) await env.DB.batch(stmts);
        return json({ ok: true, added, skipped: skipped.slice(0, 20), skipped_total: skipped.length });
      }

      // Static assets fallback (index.html + anything in public/).
      return env.ASSETS.fetch(req);
    } catch (e: any) {
      return json({ error: e?.message || String(e), stack: e?.stack }, { status: 500 });
    }
  },
};
