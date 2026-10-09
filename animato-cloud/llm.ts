/**
 * Self-hosted LLM — no API keys, no AI services.
 *
 * Every model runs on a GitHub CPU runner with llama.cpp:
 *  - in the video / editor runs, a llama-server started by the workflow on the same
 *    runner (LOCAL_LLM_URL, OpenAI-compatible: http://127.0.0.1:8080/v1), plus an optional
 *    vision server for picture checks (LOCAL_VLM_URL);
 *  - in the app (Express server / Cloudflare Worker), the "brain" — a runner session that
 *    keeps the model loaded and answers queued requests — reached through `relay`.
 *
 * The class keeps the old interface (attempts(), hasKeys, maxWaitMs, lastErrors) so every
 * caller works unchanged. Web research: the runner reads public search results itself
 * (DuckDuckGo's HTML page — no key) and the model answers from those pages, with sources.
 */
export type Provider = 'local' | 'brain';
export interface LlmConfig {
  /** OpenAI-compatible llama-server base URL, e.g. http://127.0.0.1:8080/v1 */
  localUrl?: string;
  /** Optional vision llama-server (Qwen2.5-VL + mmproj). */
  visionUrl?: string;
  /** App side: send the request to the brain runner and wait for its text. */
  relay?: (req: LlmRequest) => Promise<string>;
  model?: string;
  fetchImpl?: typeof fetch;
  log?: (msg: string) => void;
  seed?: number;
  /** Legacy fields (ignored — no external AI providers any more). */
  geminiKeys?: string[]; groqKeys?: string[]; geminiModels?: string[]; groqModels?: string[]; geminiBase?: string; groqBase?: string;
}
export interface LlmRequest {
  system: string;
  user: string;
  saferUser?: string;
  temperature?: number;
  maxTokens?: number;
  json?: boolean;
  timeoutMs?: number;
  /** Research: the answer is grounded on live search results fetched by the runner. */
  webSearch?: boolean;
  /** Pictures to look at (needs the vision server). */
  images?: { mime: string; data: string }[];
  task?: string;
}
export interface WebSource { title: string; uri: string }
export interface LlmAttempt { provider: Provider; model: string; text: string; ms: number; sources?: WebSource[] }

/** Default models (downloaded by the workflows). */
export const LOCAL_TEXT_MODEL = 'Qwen3-4B-Instruct-2507 (Q4_K_M, llama.cpp)';
export const LOCAL_VISION_MODEL = 'Qwen2.5-VL-3B-Instruct (Q4_K_M + mmproj, llama.cpp)';

export class LlmPool {
  cfg: LlmConfig;
  maxWaitMs = 4 * 60 * 1000;
  lastErrors: string[] = [];
  exhausted = new Set<string>();
  constructor(cfg: LlmConfig) { this.cfg = cfg; }

  get hasKeys() { return !!(this.cfg.localUrl || this.cfg.relay); }
  get hasVision() { return !!(this.cfg.visionUrl || this.cfg.relay); }
  private log(m: string) { this.cfg.log?.(m); }

  /** Up to three answers (the caller stops when one is good): a retry nudges the temperature. */
  async *attempts(req: LlmRequest): AsyncGenerator<LlmAttempt> {
    if (!this.hasKeys) return;
    if (req.images?.length && !this.hasVision) return;   // no vision model here → "could not ask"
    let sources: WebSource[] = [];
    let user = req.user;
    if (req.webSearch && this.cfg.localUrl) {
      try {
        const found = await webContext(req.user, this.cfg.fetchImpl || fetch);
        sources = found.sources;
        if (found.text) user = `LIVE SEARCH RESULTS (use ONLY these for facts; cite the outlet):\n${found.text}\n\n${req.user}`;
        else this.log('Web research: no search results could be read — answering without sources.');
      } catch (e: any) { this.log(`Web research failed (${e?.message || e}).`); }
    }
    for (let i = 0; i < 3; i++) {
      const t0 = Date.now();
      const r = { ...req, user: i > 0 && req.saferUser ? req.saferUser : user, temperature: Math.min(1.1, (req.temperature ?? 0.7) + i * 0.15) };
      try {
        const text = this.cfg.relay ? await this.cfg.relay(r) : await this.callLocal(r);
        const clean = stripThink(text);
        if (!clean) throw new Error('empty answer');
        yield { provider: this.cfg.relay ? 'brain' : 'local', model: req.images?.length ? LOCAL_VISION_MODEL : (this.cfg.model || LOCAL_TEXT_MODEL), text: clean, ms: Date.now() - t0, sources };
      } catch (e: any) {
        const note = `local model: ${String(e?.message || e).slice(0, 200)}`;
        this.lastErrors.push(note); this.log(note);
        if (/not running|offline|starting/i.test(String(e?.message))) return;   // no point retrying
      }
    }
  }

  private async callLocal(req: LlmRequest): Promise<string> {
    const f = this.cfg.fetchImpl || fetch;
    const vision = !!req.images?.length;
    const base = String((vision ? this.cfg.visionUrl : this.cfg.localUrl) || '').replace(/\/+$/, '');
    if (!base) throw new Error('local model not running');
    const body: any = {
      model: 'local', temperature: req.temperature ?? 0.7, max_tokens: req.maxTokens || 2048, cache_prompt: true,
      messages: [{ role: 'system', content: req.system }, {
        role: 'user',
        content: vision ? [...req.images!.map((im) => ({ type: 'image_url', image_url: { url: `data:${im.mime};base64,${im.data}` } })), { type: 'text', text: req.user }] : req.user,
      }],
    };
    if (req.json) body.response_format = { type: 'json_object' };
    const res = await f(`${base}/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(Math.max(req.timeoutMs || 0, 600000)) });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} ${text.slice(0, 160)}`);
    const data = JSON.parse(text);
    return String(data?.choices?.[0]?.message?.content || '');
  }
}

/** Remove reasoning blocks some models print. */
export function stripThink(s: string): string { return String(s || '').replace(/<think>[\s\S]*?<\/think>/gi, '').trim(); }

/** Public search results + the first readable paragraphs of the top pages (no API, no key). */
export async function webContext(query: string, f: typeof fetch): Promise<{ text: string; sources: WebSource[] }> {
  const q = String(query).replace(/\s+/g, ' ').slice(0, 300);
  const ua = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36' };
  const res = await f(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, { headers: ua, signal: AbortSignal.timeout(15000) });
  const html = await res.text();
  const strip = (s: string) => s.replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();
  const items: { title: string; uri: string; snippet: string }[] = [];
  for (const m of html.matchAll(/<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)) {
    let uri = m[1];
    const u = uri.match(/[?&]uddg=([^&]+)/); if (u) uri = decodeURIComponent(u[1]);
    if (!/^https?:/.test(uri)) continue;
    items.push({ title: strip(m[2]), uri, snippet: strip(m[3]) });
    if (items.length >= 6) break;
  }
  const pages = await Promise.all(items.slice(0, 3).map(async (it) => {
    try {
      const r = await f(it.uri, { headers: ua, signal: AbortSignal.timeout(12000) });
      const h = (await r.text()).replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, ' ');
      const paras = [...h.matchAll(/<p[^>]*>([\s\S]*?)<\/p>/gi)].map((p) => strip(p[1])).filter((p) => p.length > 60);
      return paras.join(' ').slice(0, 1800);
    } catch { return ''; }
  }));
  const text = items.map((it, i) => `[${i + 1}] ${it.title} — ${it.uri}\n${it.snippet}${pages[i] ? `\n${pages[i]}` : ''}`).join('\n\n');
  return { text, sources: items.map((it) => ({ title: it.title, uri: it.uri })) };
}

/** First JSON object in a model answer (tolerates code fences and <think> blocks). */
export function extractJsonObject(text: string): any {
  const cleaned = String(text || '').replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/```json/gi, '```').replace(/```/g, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end <= start) throw new Error('no JSON object in model output');
  return JSON.parse(cleaned.slice(start, end + 1));
}
