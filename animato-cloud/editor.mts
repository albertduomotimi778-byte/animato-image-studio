/**
 * Animato — AI Auto Video Editor (runs on GitHub Actions).
 *
 * Input: a video (e.g. a screen recording of an app) and, optionally, a
 * separate voice recording talking about it. Output: one finished MP4 with the
 * captions drawn onto the picture.
 *
 *   1. The files are downloaded together; then, at the same time,
 *        - the voice is transcribed word by word (faster-whisper on this runner),
 *        - the loudness of the voice is measured every 10 ms (to place cuts),
 *        - a separately recorded voice: the video is "watched" (frames described
 *          by a vision model, several batches at once).
 *   2. Fillers (um, uh…), stutters, restarted phrases ("I want to — I want to
 *      show you") and long pauses are removed.
 *   3. Retakes are found by comparing sentences (same words / a line started again);
 *      every take gets a quality score (complete, fluent, few fillers, later is
 *      better) and exactly one take per retake group is kept. An AI editor then
 *      keeps what informs and holds attention, following the creator's instructions.
 *   4. Every cut is placed in the quietest moment between two words (measured
 *      from the audio itself, never inside a word) and rounded to a whole video
 *      frame, so picture and sound stay in sync however many cuts there are.
 *   5. The video pieces are rendered in parallel (framing + captions), joined
 *      without re-encoding, and muxed with the cleaned voice (denoise,
 *      compression, −16 LUFS). H.264 + AAC, fast start.
 *
 * TUTORIAL MODE — a screen recording with NO voice. When the video has no voice-over of its own
 * (no sound track, a silent one, or only music / system sounds) and none was uploaded, the editor
 * writes one instead of failing: it watches the recording (vision model + a freeze detector), cuts
 * waiting / loading / mistakes / wandering, speeds up the slow-but-needed steps, writes a narration
 * for every step from what is actually on screen, speaks it with a neural voice, draws word-by-word
 * subtitles and mixes in the original score. A video that already has somebody talking is never
 * narrated over: it goes through the voice-driven pipeline above, untouched.
 *
 * Run with: node --experimental-strip-types animato-cloud/editor.mts
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { LlmPool, extractJsonObject, createRelay, splitKeys } from './llm.ts';
import { composeWav, type MusicMood } from './music.ts';

const ENV = process.env;
const EVENT = (() => { try { return JSON.parse(fs.readFileSync(ENV.GITHUB_EVENT_PATH || '', 'utf8')); } catch { return {}; } })();
const CP = EVENT.client_payload || {};
const JOB = CP.job || {};
const AUTH = CP.auth || {};
const JOB_ID = String(CP.job_id || ENV.EDITOR_JOB_ID || 'local');
const APP = String(CP.app_url || ENV.APP_URL || '').replace(/\/+$/, '');
const RUN_URL = ENV.GITHUB_RUN_ID ? `${ENV.GITHUB_SERVER_URL || 'https://github.com'}/${ENV.GITHUB_REPOSITORY}/actions/runs/${ENV.GITHUB_RUN_ID}` : '';
const keys = (v: any) => String(v || '').split(/[\s,;]+/).map((x) => x.trim()).filter((x) => x.length > 8);
const GEMINI: string[] = [];   // no AI API keys: the models run on this runner
const GROQ: string[] = [];     // speech → text: faster-whisper on this runner (below)
if (ENV.GITHUB_ACTIONS === 'true') for (const s of [AUTH.runner_key, ...GEMINI, ...GROQ]) if (s && String(s).length > 6) console.log(`::add-mask::${s}`);

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FONTS = path.join(HERE, 'assets', 'fonts');
const WORK = path.resolve(ENV.ANIMATO_OUTPUT_DIR || 'output/editor');
fs.mkdirSync(WORK, { recursive: true });
const CPUS = Math.max(2, os.cpus()?.length || 2);
const FPS = 30;
const log = (m: string) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`);
class EditError extends Error {}

// The planner is the self-hosted model started by the workflow on this runner (no API keys).
// Hosted AI through the app server (which rotates the keys: Gemini first for editing, Groq / OpenRouter as fallbacks).
const RELAY = createRelay({ appUrl: APP, scope: `editor:${JOB_ID}`, runnerKey: String(AUTH.runner_key || '') });
const LLM = new LlmPool({
  relay: RELAY.available ? (r) => RELAY.llm(r) : undefined,
  groqKeys: splitKeys(ENV.GROQ_API_KEYS), geminiKeys: splitKeys(ENV.GEMINI_API_KEYS), openrouterKeys: splitKeys(ENV.OPENROUTER_API_KEYS),
  log
});

let lastReport = 0;
async function report(status: string, step: string, progress: number, logLine = '', extra: any = {}) {
  if (logLine) log(logLine);
  lastReport = Date.now();
  if (!APP) return;
  try {
    await fetch(`${APP}/api/editor/jobs/${JOB_ID}/status`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Animato-Runner-Key': String(AUTH.runner_key || '') },
      body: JSON.stringify({ status, step, progress, log: logLine, runUrl: RUN_URL, ...extra }), signal: AbortSignal.timeout(20000)
    });
  } catch (e: any) { log(`(status update failed: ${e?.message})`); }
}

function run(cmd: string, args: string[], timeoutMs = 3600_000): Promise<{ code: number; out: string; err: string }> {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    const t = setTimeout(() => { try { p.kill('SIGKILL'); } catch {} }, timeoutMs);
    p.stdout.on('data', (d) => { out += d; if (out.length > 4e6) out = out.slice(-2e6); });
    p.stderr.on('data', (d) => { err += d; if (err.length > 4e6) err = err.slice(-2e6); });
    p.on('close', (code) => { clearTimeout(t); resolve({ code: code ?? -1, out, err }); });
    p.on('error', (e) => { clearTimeout(t); resolve({ code: -1, out, err: String(e) }); });
  });
}
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const ff = (args: string[], timeoutMs?: number) => run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args], timeoutMs);

async function probe(file: string): Promise<{ duration: number; width: number; height: number; hasAudio: boolean; hasVideo: boolean; fps: number; rotation: number }> {
  const r = await run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate', '-of', 'json', file]);
  const j = JSON.parse(r.out || '{}');
  const v = (j.streams || []).find((s: any) => s.codec_type === 'video');
  const [n, d] = String(v?.r_frame_rate || '30/1').split('/').map(Number);
  const rot = Math.abs(Number(v?.side_data_list?.find((x: any) => x.rotation !== undefined)?.rotation ?? v?.tags?.rotate ?? 0)) % 180;
  // Phone videos store portrait as landscape + a rotation flag: report the size as shown.
  const [w, h] = rot === 90 ? [v?.height || 0, v?.width || 0] : [v?.width || 0, v?.height || 0];
  return { duration: Number(j.format?.duration || 0), width: w, height: h, hasAudio: (j.streams || []).some((s: any) => s.codec_type === 'audio'), hasVideo: !!v, fps: d ? n / d : 30, rotation: rot };
}

/** Runs async jobs with at most `n` at a time, keeping the order of the results. */
async function pool<T, R>(items: T[], n: number, fn: (x: T, i: number) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
async function fetchFile(spec: any, dest: string) {
  if (!spec) return false;
  if (spec.url) {
    // Streamed straight to disk (large screen recordings), with a few retries.
    for (let a = 0; a < 4; a++) {
      const res = await fetch(spec.url, { signal: AbortSignal.timeout(3600_000) }).catch(() => null);
      if (res?.ok && res.body) {
        const { Readable } = await import('node:stream');
        const { pipeline } = await import('node:stream/promises');
        await pipeline(Readable.fromWeb(res.body as any), fs.createWriteStream(dest));
        if (!spec.size || fs.statSync(dest).size === Number(spec.size)) return true;
        log(`Download of ${spec.name || 'the file'} was incomplete — retrying.`);
      } else if (res && res.status < 500 && res.status !== 429) {
        throw new EditError(`Could not download ${spec.name || 'the file'} (HTTP ${res.status}).`);
      }
      await new Promise((z) => setTimeout(z, 3000 * (a + 1)));
    }
    throw new EditError(`Could not download ${spec.name || 'the file'} after several tries.`);
  }
  return false;
}

/** Sends the finished file back in pieces, several at once. Returns the pieces' ETags. */
async function sendFile(spec: any, file: string): Promise<{ parts?: string[]; size: number }> {
  const bytes = fs.readFileSync(file);
  if (!spec?.base) { log(`(local run: kept ${path.basename(file)} in ${WORK})`); return { size: bytes.length }; }
  const size = Number(spec.chunkSize || 6 * 1024 * 1024);
  const pieces = Array.from({ length: Math.max(1, Math.ceil(bytes.length / size)) }, (_, i) => i);
  const parts = await pool(pieces, 4, async (n) => {
    for (let a = 0; a < 5; a++) {
      const r = await fetch(`${spec.base}/${n}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/octet-stream', 'X-Animato-Runner-Key': String(AUTH.runner_key || ''), 'X-Animato-Parts': 'client' },
        body: bytes.subarray(n * size, (n + 1) * size), signal: AbortSignal.timeout(180000)
      }).catch(() => null);
      if (r?.ok) { const d: any = await r.json().catch(() => ({})); if (d?.etag) return String(d.etag); }
      await new Promise((z) => setTimeout(z, 2000 * (a + 1)));
    }
    throw new EditError(`Could not upload part ${n + 1} of the result.`);
  });
  return { parts, size: bytes.length };
}

// ---------------------------------------------------------------------------
// 1a. Transcription (word timings)
// ---------------------------------------------------------------------------
interface W { w: string; s: number; e: number }
const bare = (x: string) => x.toLowerCase().replace(/[^a-z0-9'\u00C0-\u024F\u0400-\u04FF]/g, '');

async function transcribe(audio: string, dur: number, hint: string): Promise<W[]> {
  // Test hook (never set in production): a ready word list.
  if (ENV.EDITOR_TRANSCRIPT && fs.existsSync(ENV.EDITOR_TRANSCRIPT)) return JSON.parse(fs.readFileSync(ENV.EDITOR_TRANSCRIPT, 'utf8'));
  // Hosted Groq Whisper (through the app server, keys rotated). Long voices are sent in pieces
  // under the upload limit and the word times are shifted back into place.
  const wav = path.join(WORK, 'asr_in.wav');
  if ((await ff(['-i', audio, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav])).code !== 0) throw new EditError('Transcription failed: the voice track could not be read.');
  const CHUNK = 600;   // seconds per request (16 kHz mono WAV ≈ 19 MB per 10 min, under the 24 MB limit)
  const pieces = Math.max(1, Math.ceil(dur / CHUNK));
  const all: W[] = [];
  const t0 = Date.now();
  for (let i = 0; i < pieces; i++) {
    const offset = i * CHUNK;
    const file = pieces === 1 ? wav : path.join(WORK, `asr_${i}.wav`);
    if (pieces > 1 && (await ff(['-ss', String(offset), '-t', String(CHUNK), '-i', wav, '-c:a', 'pcm_s16le', file])).code !== 0) throw new EditError('Transcription failed: could not split the voice track.');
    const bytes = new Uint8Array(fs.readFileSync(file));
    let got: { text: string; words: { word: string; start: number; end: number }[] } | null = null, lastErr = '';
    for (let attempt = 0; attempt < 3 && !got; attempt++) {
      try {
        if (RELAY.available) got = await RELAY.stt(bytes, 'audio/wav', { prompt: hint.slice(0, 220) });
        else throw new Error('no app address / runner key to reach the speech service');
      } catch (e: any) { lastErr = String(e?.message || e).slice(0, 200); log(`Speech-to-text attempt ${attempt + 1} failed (${lastErr}).`); await sleep(3000 * (attempt + 1)); }
    }
    if (!got) throw new EditError(`Transcription failed: ${lastErr || 'unknown error'}`);
    for (const w of got.words) all.push({ w: w.word, s: +(w.start + offset).toFixed(3), e: +(w.end + offset).toFixed(3) });
  }
  log(`Transcribed ${all.length} words in ${Math.round((Date.now() - t0) / 1000)}s (Groq Whisper).`);
  return all;
}
function punctuate(words: W[], segs: { text: string; s: number; e: number }[]) {
  for (const sg of segs) {
    const toks = sg.text.trim().split(/\s+/);
    const inSeg = words.filter((w) => w.s >= sg.s - 0.05 && w.e <= sg.e + 0.3 && !/[.!?,]$/.test(w.w));
    let j = 0;
    for (const w of inSeg) {
      while (j < toks.length && bare(toks[j]) !== bare(w.w)) j++;
      if (j < toks.length) { if (/[.!?,;:]$/.test(toks[j])) w.w = w.w.replace(/[.!?,;:]*$/, '') + toks[j].slice(-1); j++; }
    }
  }
}
function dedupeOverlap(words: W[]): W[] {
  words.sort((a, b) => a.s - b.s);
  const out = words.filter((w, i) => i === 0 || !(Math.abs(w.s - words[i - 1].s) < 0.02 && w.w === words[i - 1].w));
  // Whisper sometimes gives a word zero length or overlapping its neighbour.
  for (let i = 0; i < out.length; i++) {
    if (out[i].e < out[i].s + 0.04) out[i].e = out[i].s + 0.04;
    if (i + 1 < out.length && out[i].e > out[i + 1].s) out[i].e = Math.max(out[i].s + 0.02, out[i + 1].s);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 1b. Loudness every 10 ms — cuts go where the voice is quietest
// ---------------------------------------------------------------------------
class Energy {
  db: Float32Array;
  constructor(db: Float32Array) { this.db = db; }
  static async of(pcmFile: string): Promise<Energy> {
    const buf = fs.readFileSync(pcmFile);
    const samples = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 2));
    const hop = 480; // 10 ms at 48 kHz
    const n = Math.floor(samples.length / hop);
    const db = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = i * hop, end = k + hop; k < end; k++) sum += samples[k] * samples[k];
      db[i] = 10 * Math.log10(sum / hop / (32768 * 32768) + 1e-10);
    }
    return new Energy(db);
  }
  at(t: number) { const i = Math.max(0, Math.min(this.db.length - 1, Math.round(t * 100))); return this.db[i] ?? -100; }
  /** The quietest moment in [lo, hi] (a short 30 ms window, so a single dip in a word is not chosen). */
  quietest(lo: number, hi: number): number {
    if (!(hi > lo + 0.005)) return (lo + hi) / 2;
    let best = lo, bestV = Infinity;
    for (let t = lo; t <= hi + 1e-6; t += 0.01) {
      const v = Math.max(this.at(t - 0.01), this.at(t), this.at(t + 0.01));
      if (v < bestV - 0.5 || (Math.abs(v - bestV) <= 0.5 && Math.abs(t - (lo + hi) / 2) < Math.abs(best - (lo + hi) / 2))) { bestV = v; best = t; }
    }
    return best;
  }
}

// ---------------------------------------------------------------------------
// 2. Clean-up: fillers, stutters, restarted phrases, sentences
// ---------------------------------------------------------------------------
const FILLER = /^(um+|uh+|uhm+|erm+|er+|ah+|eh+|hmm+|mm+|mhm|uh-huh|umm+)[.,!?]?$/i;
const STOP = new Set('a an the and or but so to of in on at for with is are was were be it this that i you we they he she my your our its as by from'.split(' '));
interface Sentence { id: number; text: string; s: number; e: number; words: W[]; score: number; issues: number }

function clean(words: W[]): { kept: W[]; removed: W[]; fillers: number; stutters: number; restarts: number; restartTexts: string[] } {
  let fillers = 0, stutters = 0, restarts = 0;
  const restartTexts: string[] = [];
  const drop = new Set<number>();
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const b = bare(w.w);
    if (FILLER.test(w.w.trim())) { drop.add(i); fillers++; continue; }
    const nb = bare(words[i + 1]?.w || '');
    // Stutter / repeated word ("the the", "I I").
    if (b && b === nb && words[i + 1].s - w.e < 0.6 && b.length < 8) { drop.add(i); stutters++; continue; }
    // "you know" / "I mean" as verbal fillers between pauses.
    if ((`${b} ${nb}` === 'you know' || `${b} ${nb}` === 'i mean') && (i === 0 || w.s - words[i - 1].e > 0.25)) { drop.add(i); drop.add(i + 1); fillers++; i++; continue; }
  }
  // A phrase started again: "I want to — I want to show you". The abandoned start goes.
  const live = words.map((_, i) => i).filter((i) => !drop.has(i));
  for (let a = 0; a < live.length; a++) {
    for (let gap = 2; gap <= 6 && a + gap < live.length; gap++) {
      const b0 = a + gap;
      let m = 0;
      while (m < 5 && b0 + m < live.length && a + m < b0 && bare(words[live[a + m]].w) && bare(words[live[a + m]].w) === bare(words[live[b0 + m]].w)) m++;
      if (m < 2) continue;
      const pause = words[live[b0]].s - words[live[b0 - 1]].e;
      const allStop = Array.from({ length: m }, (_, k) => bare(words[live[a + k]].w)).every((x) => STOP.has(x));
      // Only inside one sentence (a whole sentence said again is a retake, handled below).
      const sentenceBreak = live.slice(a, b0).some((k) => /[.!?]$/.test(words[k].w));
      if (sentenceBreak || allStop) continue;
      if (m >= 3 || pause >= 0.3) {
        for (let k = a; k < b0; k++) drop.add(live[k]);
        restarts++;
        restartTexts.push(live.slice(a, b0).map((k) => words[k].w).join(' '));
        a = b0 - 1;
        break;
      }
    }
  }
  return { kept: words.filter((_, i) => !drop.has(i)), removed: words.filter((_, i) => drop.has(i)), fillers, stutters, restarts, restartTexts };
}

function sentences(all: W[], kept: W[], removed: W[]): Sentence[] {
  const list: Sentence[] = [];
  let cur: W[] = [];
  const flush = () => {
    if (!cur.length) return;
    const s = cur[0].s, e = cur[cur.length - 1].e;
    const issues = removed.filter((r) => r.s >= s - 0.05 && r.e <= e + 0.05).length;
    const pauses = cur.slice(1).filter((w, k) => w.s - cur[k].e > 0.8).length;
    const complete = /[.!?]$/.test(cur[cur.length - 1].w) ? 1 : 0;
    list.push({ id: list.length, text: cur.map((x) => x.w).join(' ').replace(/\s+([,.!?])/g, '$1'), s, e, words: cur, issues: issues + pauses, score: complete * 2 - issues - pauses * 0.5 - (cur.length < 3 ? 1 : 0) });
    cur = [];
  };
  kept.forEach((w, i) => {
    if (cur.length && w.s - cur[cur.length - 1].e > 1.1) flush(); // a long pause ends a thought
    cur.push(w);
    if (/[.!?]$/.test(w.w) || cur.length > 40 || i === kept.length - 1) flush();
  });
  flush();
  return list;
}

// ---------------------------------------------------------------------------
// 3. Retakes, then the AI editor
// ---------------------------------------------------------------------------
const toks = (t: string) => t.toLowerCase().replace(/[^a-z0-9' \u00C0-\u024F\u0400-\u04FF]/g, ' ').split(/\s+/).filter(Boolean);
function similarity(a: string, b: string): number {
  const A = toks(a), B = toks(b);
  if (!A.length || !B.length) return 0;
  const bag = new Map<string, number>();
  for (const x of A) bag.set(x, (bag.get(x) || 0) + 1);
  let common = 0;
  for (const x of B) { const n = bag.get(x) || 0; if (n > 0) { common++; bag.set(x, n - 1); } }
  return common / Math.max(A.length, B.length);
}
/** a is an abandoned start of b: b begins with (most of) a's opening words. */
function startsAgain(a: string, b: string): boolean {
  const A = toks(a), B = toks(b);
  const n = Math.min(4, A.length);
  if (n < 2 || A.length > B.length) return false;
  let same = 0;
  for (let i = 0; i < n; i++) if (A[i] === B[i]) same++;
  return same >= Math.max(2, n - 1) && A.slice(0, n).some((x) => !STOP.has(x));
}
/** Groups of sentences that are takes of the same line (said again within ~6 sentences / 90 s). */
function retakeGroups(list: Sentence[]): number[][] {
  const parent = list.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length && j <= i + 6; j++) {
      if (list[j].s - list[i].e > 90) break;
      const sim = similarity(list[i].text, list[j].text);
      if ((sim >= 0.6 && Math.min(list[i].words.length, list[j].words.length) >= 3) || startsAgain(list[i].text, list[j].text)) parent[find(i)] = find(j);
    }
  }
  const groups = new Map<number, number[]>();
  list.forEach((_, i) => { const r = find(i); groups.set(r, [...(groups.get(r) || []), i]); });
  return [...groups.values()].filter((g) => g.length > 1);
}
/** The best take: complete and fluent wins; the later take wins a tie (people re-record to fix a line). */
function bestTake(list: Sentence[], g: number[]): number {
  return g.map((i, rank) => ({ i, v: list[i].score + rank * 0.35 + Math.min(1, list[i].words.length / 12) * 0.5 })).sort((a, b) => b.v - a.v)[0].i;
}

async function planEdit(list: Sentence[], groups: number[][], instructions: string): Promise<{ keep: number[]; removed: { id: number; reason: string }[]; title: string; model: string }> {
  const best = new Map<number, number>();
  for (const g of groups) { const b = bestTake(list, g); for (const i of g) best.set(i, b); }
  const lines = list.map((x) => `${x.id} [${x.s.toFixed(1)}-${x.e.toFixed(1)}s]${x.issues ? ` (${x.issues} slip${x.issues > 1 ? 's' : ''})` : ''}${/[.!?]$/.test(x.text) ? '' : ' (unfinished)'} ${x.text}`).join('\n');
  const groupText = groups.length
    ? groups.map((g) => `- takes ${g.join(', ')} → best take: ${best.get(g[0])}`).join('\n')
    : '(none found)';
  const user = `You are a top short-form video editor. Below is the word-accurate transcript of a creator's voice-over, split into numbered sentences with timings. Filler sounds, stutters and restarted phrases are already removed. "(n slips)" counts removed fillers/pauses inside a sentence; "(unfinished)" means it trails off.

RETAKES DETECTED (the same line said more than once). Keep EXACTLY ONE take per group — normally the best take given (complete, fluent, usually the last). Pick another take only if it is clearly better worded:
${groupText}

Decide what stays in the final video:
- KEEP every sentence that explains, shows, persuades or entertains — the substance of what they are saying.
- REMOVE: the other takes of every retake group, false starts and abandoned sentences, rambling that repeats a point already made, off-topic asides, "wait / let me try again / sorry / cut that" moments, test phrases ("testing, one two"), and dead openings ("so, um, hi, so today…") when a stronger opening exists.
- Keep the original order, EXCEPT you may move ONE strong hook sentence (a benefit or surprising fact) to the very start if the recording opens weakly.
- Never cut the middle of an explanation so that it stops making sense. When unsure, keep.
${instructions ? `\nTHE CREATOR'S INSTRUCTIONS (follow them): ${instructions}\n` : ''}
TRANSCRIPT
${lines}

Return ONLY JSON: {"keep": [sentence ids in final order], "removed": [{"id": 3, "reason": "retake of 4"}], "title": "a short title for the video"}`;
  const ids = new Set(list.map((x) => x.id));
  /** Never two takes of the same line: keep the best one the plan chose. */
  const enforce = (keep: number[]) => {
    const out: number[] = [];
    for (const id of keep) {
      const g = groups.find((x) => x.includes(id));
      const twin = g ? out.find((o) => g.includes(o) && (similarity(list[o].text, list[id].text) >= 0.75 || startsAgain(list[Math.min(o, id)].text, list[Math.max(o, id)].text))) : undefined;
      if (twin === undefined) { out.push(id); continue; }
      const b = bestTake(list, g!.filter((x) => x === twin || x === id));
      if (b === id) out[out.indexOf(twin)] = id;
    }
    return out;
  };
  for await (const a of LLM.attempts({ system: 'You are an expert video editor. You answer with one JSON object.', user, json: true, temperature: 0.2, maxTokens: 6000, timeoutMs: 120000, task: 'editor_plan' })) {
    try {
      const j = extractJsonObject(a.text);
      const keep = enforce((Array.isArray(j.keep) ? j.keep : []).map(Number).filter((x: number, i: number, arr: number[]) => ids.has(x) && arr.indexOf(x) === i));
      if (keep.length < Math.max(1, Math.round((list.length - groups.reduce((n, g) => n + g.length - 1, 0)) * 0.3))) throw new Error(`kept only ${keep.length}/${list.length}`);
      return { keep, removed: (Array.isArray(j.removed) ? j.removed : []).map((r: any) => ({ id: Number(r.id), reason: String(r.reason || '') })), title: String(j.title || ''), model: `${a.provider}/${a.model}` };
    } catch (e: any) { log(`Edit plan from ${a.provider}/${a.model} unusable (${e?.message}).`); }
  }
  // No model: one take per retake group (the best), unfinished one-word bits dropped, the rest in order.
  const keep = list.filter((x) => (!best.has(x.id) || best.get(x.id) === x.id) && x.words.length >= 2).map((x) => x.id);
  return { keep, removed: [], title: '', model: 'rules' };
}

// ---------------------------------------------------------------------------
// 4. Watch the video and match each sentence to what is on screen
// ---------------------------------------------------------------------------
async function describeFrames(video: string, duration: number, rich = false): Promise<{ t: number; d: string }[]> {
  const n = rich ? Math.max(8, Math.min(100, Math.round(duration / 2.5))) : Math.max(6, Math.min(90, Math.round(duration / 2)));
  const step = duration / n;
  const times = Array.from({ length: n }, (_, i) => +(i * step + step / 2).toFixed(2));
  const grabbed = await pool(times, CPUS, async (t, i) => {
    const f = path.join(WORK, `f_${i}.jpg`);
    await ff(['-ss', String(t), '-i', video, '-frames:v', '1', '-vf', `scale=${rich ? 704 : 512}:-2`, '-q:v', '6', f]);
    return fs.existsSync(f) ? { t, file: f } : null;
  });
  const frames = grabbed.filter(Boolean) as { t: number; file: string }[];
  const batches: { t: number; file: string }[][] = [];
  for (let i = 0; i < frames.length; i += 8) batches.push(frames.slice(i, i + 8));
  const described = await pool(batches, 3, async (batch) => {
    for await (const a of LLM.attempts({
      system: 'You describe video frames for an editor. Answer with one JSON object.',
      user: `These ${batch.length} images are frames from a screen recording / video, in order. ${rich ? 'For EACH frame say in 12-28 words: which app / screen / tab it is, the exact label of the button, menu, tool or field the person is using or that just changed, any readable heading or dialog text, and what they are doing (clicking, typing, dragging, waiting for something to load). If nothing is happening on screen (idle, a loading spinner, the same as before) start the sentence with "IDLE:".' : 'For EACH frame say in 8-18 words exactly what is on screen (which app/page/screen, the feature or button in focus, any headline text, what the person is doing).'} Return ONLY JSON: {"frames": ["...", "..."]} with exactly ${batch.length} items.`,
      images: batch.map((b) => ({ mime: 'image/jpeg', data: fs.readFileSync(b.file).toString('base64') })),
      json: true, temperature: 0, maxTokens: rich ? 2400 : 1200, timeoutMs: 90000, task: 'editor_vision'
    })) {
      try {
        const j = extractJsonObject(a.text);
        return batch.map((b, k) => ({ t: b.t, d: String(j.frames?.[k] || '').slice(0, rich ? 260 : 160) }));
      } catch {}
    }
    return batch.map((b) => ({ t: b.t, d: '' }));
  });
  return described.flat();
}

async function matchVideo(kept: Sentence[], frames: { t: number; d: string }[], videoDur: number, audioDur: number): Promise<number[]> {
  // Default: time-proportional (works when the voice was recorded along with the video).
  const linear = kept.map((x) => Math.max(0, Math.min(videoDur - (x.e - x.s), x.s * (videoDur / Math.max(1, audioDur)))));
  if (!frames.some((f) => f.d)) return linear;
  const user = `You are editing a video. The voice-over sentences below must each be shown over the part of the video that matches what is said.

VIDEO (${videoDur.toFixed(1)} s) — what is on screen at each time:
${frames.map((f) => `${f.t.toFixed(1)}s: ${f.d || '(unknown)'}`).join('\n')}

VOICE-OVER SENTENCES (in final order) with how long each lasts:
${kept.map((x, i) => `${i}. (${(x.e - x.s).toFixed(1)} s) ${x.text}`).join('\n')}

For each sentence choose the video START time (seconds) of the clip to show while it is spoken (the clip lasts as long as the sentence). Rules: match the feature/screen being talked about; generally move forward through the video; avoid showing the same seconds twice unless nothing else fits; start + duration must be ≤ ${videoDur.toFixed(1)}.
Return ONLY JSON: {"starts": [number, ...]} with exactly ${kept.length} numbers.`;
  for await (const a of LLM.attempts({ system: 'You are an expert video editor. You answer with one JSON object.', user, json: true, temperature: 0.1, maxTokens: 3000, timeoutMs: 120000, task: 'editor_match' })) {
    try {
      const j = extractJsonObject(a.text);
      const st = (j.starts || []).map(Number);
      if (st.length !== kept.length || st.some((x: number) => !Number.isFinite(x))) throw new Error('wrong length');
      return st.map((x: number, i: number) => Math.max(0, Math.min(videoDur - Math.min(videoDur, kept[i].e - kept[i].s), x)));
    } catch (e: any) { log(`Match from ${a.provider}/${a.model} unusable (${e?.message}).`); }
  }
  return linear;
}

// ---------------------------------------------------------------------------
// 4b. The timeline: cuts between words, rounded to whole frames
// ---------------------------------------------------------------------------
interface Seg { a0: number; a1: number; si: number; frames: number; v0?: number }

function buildTimeline(kept: Sentence[], all: W[], energy: Energy, audioDur: number, cutPauses: boolean): Seg[] {
  const index = new Map<W, number>();
  all.forEach((w, i) => index.set(w, i));
  const prevEnd = (w: W) => { const i = index.get(w)!; return i > 0 ? all[i - 1].e : 0; };
  const nextStart = (w: W) => { const i = index.get(w)!; return i < all.length - 1 ? all[i + 1].s : audioDur; };
  // Start just before a word / end just after one, at the quietest point, never inside a neighbouring word.
  const startAt = (w: W, reach: number) => {
    const lo = Math.max(prevEnd(w) + 0.01, w.s - reach), hi = Math.max(lo, w.s - 0.02);
    return Math.max(0, Math.min(w.s, energy.quietest(lo, hi)));
  };
  const endAt = (w: W, reach: number) => {
    const lo = w.e + 0.04, hi = Math.min(nextStart(w) - 0.01, w.e + reach);
    return Math.min(audioDur, hi <= lo ? Math.max(w.e, Math.min(lo, nextStart(w))) : energy.quietest(lo, hi));
  };
  const segs: Seg[] = [];
  kept.forEach((x, si) => {
    const ws = x.words;
    let a0 = startAt(ws[0], 0.25);
    for (let k = 1; k < ws.length; k++) {
      const gap = ws[k].s - ws[k - 1].e;
      const skipped = index.get(ws[k])! - index.get(ws[k - 1])! > 1; // a filler / restart was removed here
      if ((cutPauses && gap > 0.55) || skipped) {
        const a1 = endAt(ws[k - 1], skipped ? 0.12 : 0.16);
        segs.push({ a0, a1, si, frames: 0 });
        a0 = startAt(ws[k], skipped ? 0.12 : 0.12);
      }
    }
    segs.push({ a0, a1: endAt(ws[ws.length - 1], 0.3), si, frames: 0 });
  });
  // Round every piece to whole frames (audio follows), drop slivers, join pieces that touch.
  const out: Seg[] = [];
  for (const g of segs) {
    const n = Math.round((g.a1 - g.a0) * FPS);
    if (n < 3) continue;
    const prev = out[out.length - 1];
    if (prev && prev.si === g.si && Math.abs(prev.a0 + prev.frames / FPS - g.a0) < 0.03) { prev.frames = Math.round((g.a1 - prev.a0) * FPS); prev.a1 = prev.a0 + prev.frames / FPS; continue; }
    out.push({ ...g, frames: n, a1: g.a0 + n / FPS });
  }
  return out;
}

// ---------------------------------------------------------------------------
// 5. Captions drawn onto the video (ASS, word by word)
// ---------------------------------------------------------------------------
function assTime(t: number) {
  const cs = Math.max(0, Math.round(t * 100));
  const h = Math.floor(cs / 360000), m = Math.floor(cs / 6000) % 60, s = Math.floor(cs / 100) % 60;
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(cs % 100).padStart(2, '0')}`;
}
function captionsAss(segs: Seg[], kept: Sentence[], W: number, H: number): string {
  // Every kept word placed on the final timeline.
  const placed: { w: string; s: number; e: number; end: boolean }[] = [];
  let t = 0;
  for (const g of segs) {
    for (const w of kept[g.si].words) {
      if (w.s >= g.a0 - 0.02 && w.e <= g.a1 + 0.05) {
        const txt = w.w.replace(/[{}\\]/g, '').trim();
        if (txt) placed.push({ w: txt, s: t + Math.max(0, w.s - g.a0), e: t + Math.min(g.a1 - g.a0, w.e - g.a0), end: /[.!?]$/.test(txt) || w === kept[g.si].words[kept[g.si].words.length - 1] });
      }
    }
    t += g.frames / FPS;
  }
  return captionsFromPlaced(placed, W, H);
}
type Placed = { w: string; s: number; e: number; end: boolean };
/** Word-by-word captions for words already placed on the final timeline (extra: ready-made ASS event lines). */
function captionsFromPlaced(placed: Placed[], W: number, H: number, extra: string[] = []): string {
  // Short chunks (≤ 4 words / ~22 letters), broken at sentence ends and pauses.
  const chunks: typeof placed[] = [];
  let cur: typeof placed = [];
  placed.forEach((p, i) => {
    cur.push(p);
    const next = placed[i + 1];
    const chars = cur.reduce((n, x) => n + x.w.length + 1, 0);
    if (!next || p.end || cur.length >= 4 || chars >= 22 || next.s - p.e > 0.45) { chunks.push(cur); cur = []; }
  });
  const vertical = H > W;
  const size = Math.round(Math.min(W, H) * (vertical ? 0.068 : 0.066));
  const lines: string[] = [];
  let shown = 0; // captions never overlap: each one starts after the previous one ends
  chunks.forEach((c, ci) => {
    const nextStart = Math.max(0, (chunks[ci + 1]?.[0].s ?? c[c.length - 1].e + 0.6) - 0.05);
    const chunkEnd = Math.min(nextStart, c[c.length - 1].e + 0.4);
    c.forEach((wd, k) => {
      const s = Math.max(shown, k === 0 ? wd.s - 0.05 : wd.s);
      const e = k === c.length - 1 ? chunkEnd : c[k + 1].s;
      if (e <= s + 0.005) return;
      shown = e;
      const text = c.map((x, m) => (m === k ? `{\\c&H0000D7FF&}${x.w}{\\r}` : x.w)).join(' ');
      lines.push(`Dialogue: 0,${assTime(s)},${assTime(e)},Cap,,0,0,0,,${text}`);
    });
  });
  const marginV = Math.round(H * (vertical ? 0.2 : 0.08));
  const marginLR = Math.round(W * 0.07);
  return `[Script Info]
ScriptType: v4.00+
PlayResX: ${W}
PlayResY: ${H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Poppins,${size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,1,${Math.max(2, Math.round(size * 0.09))},${Math.max(1, Math.round(size * 0.04))},2,${marginLR},${marginLR},${marginV},1
Style: Badge,Poppins,${Math.round(size * 0.5)},&H00FFFFFF,&H00FFFFFF,&H64000000,&H64000000,-1,0,0,0,100,100,0,0,3,${Math.round(size * 0.14)},0,9,${marginLR},${marginLR},${Math.round(H * 0.035)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
${[...lines, ...extra].join('\n')}
`;
}

// Content-aware original-score choice. Each edit gets a seeded, newly synthesised score;
// no stock recording is bundled or reused. Dialog is ducked underneath the voice.
function musicMoodForText(text: string): MusicMood {
  const t = text.toLowerCase();
  if (/\b(horror|haunted|scary|ghost|fear|terrifying|nightmare|murder|creepy)\b/.test(t)) return 'horror';
  if (/\b(mystery|detective|crime|investigat|secret|clue|suspense|thriller)\b/.test(t)) return 'mystery';
  if (/\b(news|headline|breaking|politics|report|update|latest)\b/.test(t)) return 'news';
  if (/\b(technology|tech|software|app|ai|phone|computer|digital|review|tutorial)\b/.test(t)) return 'tech';
  if (/\b(cooking|recipe|food|kitchen|baking|meal|cook)\b/.test(t)) return 'upbeat';
  if (/\b(comedy|funny|joke|prank|laugh|humour|humor)\b/.test(t)) return 'comedy';
  if (/\b(action|fight|battle|race|adventure|challenge|competition|sports)\b/.test(t)) return 'action';
  if (/\b(love|heartbreak|sad|emotional|memories|grief|hope|family|inspiring|inspiration)\b/.test(t)) return 'emotional';
  return 'story';
}
function makeOriginalMusic(seconds: number, mood: MusicMood, seed: string, dest: string) {
  // Bound synthesis memory for long uploads; FFmpeg loops the generated bed after ten minutes.
  const duration = Math.max(3, Math.min(600, seconds));
  fs.writeFileSync(dest, Buffer.from(composeWav(mood, duration, seed)));
}

// ---------------------------------------------------------------------------
// 6. Render: pieces in parallel → joined → voice muxed
// ---------------------------------------------------------------------------
function frameSize(format: string, w: number, h: number): [number, number] {
  const even = (x: number) => Math.max(2, Math.round(x / 2) * 2);
  switch (format) {
    case '9:16': return [1080, 1920];
    case '16:9': return [1920, 1080];
    case '1:1': return [1080, 1080];
    case '4:5': return [1080, 1350];
    default: { const s = Math.min(1, 1920 / Math.max(w, h, 1)); return [even(w * s) || 1920, even(h * s) || 1080]; }
  }
}
const escF = (p: string) => p.replace(/\\/g, '/').replace(/:/g, '\\:').replace(/'/g, "\\'");

/** The cleaned voice for the final timeline, spliced sample-exactly in Node (48 kHz mono). */
function spliceVoice(pcm: Buffer, segs: Seg[], out: string) {
  const src = new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.length / 2));
  const perFrame = 48000 / FPS;
  const total = segs.reduce((n, g) => n + g.frames * perFrame, 0);
  const dst = new Int16Array(total);
  const fade = 480; // 10 ms — no clicks at the cuts
  let o = 0;
  for (const g of segs) {
    const n = g.frames * perFrame;
    const start = Math.round(g.a0 * 48000);
    for (let k = 0; k < n; k++) {
      let v = src[start + k] ?? 0;
      if (k < fade) v = v * (k / fade);
      else if (k > n - fade) v = v * ((n - k) / fade);
      dst[o + k] = v;
    }
    o += n;
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0); header.writeUInt32LE(36 + dst.length * 2, 4); header.write('WAVE', 8);
  header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(48000, 24); header.writeUInt32LE(96000, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write('data', 36); header.writeUInt32LE(dst.length * 2, 40);
  fs.writeFileSync(out, Buffer.concat([header, Buffer.from(dst.buffer)]));
}

/** Fits the picture into W×H; when the shape differs, a blurred copy of it fills the rest. */
function framingFilter(vInfo: { width: number; height: number }, W: number, H: number): string {
  const srcAR = vInfo.width / Math.max(1, vInfo.height), dstAR = W / H;
  const sameShape = Math.abs(srcAR - dstAR) / dstAR < 0.012;
  const bw = Math.max(2, Math.round(W / 16) * 2), bh = Math.max(2, Math.round(H / 16) * 2);
  // The picture fitted inside, a blurred copy (made small, so it is quick) filling the rest.
  return sameShape
    ? `scale=${W}:${H}:flags=bicubic,setsar=1`
    : `split[b][f];[b]scale=${bw}:${bh}:force_original_aspect_ratio=increase,crop=${bw}:${bh},boxblur=6:2,scale=${W}:${H},eq=brightness=-0.06[bg];[f]scale=${W}:${H}:force_original_aspect_ratio=decrease:flags=bicubic[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1`;
}

/** The cleaned voice track + the joined picture pieces + the original score → the final MP4. */
async function finishVideo(o: { count: number; finalDuration: number; rawVoice: string; out: string; musicText?: string; voiceFilter?: string; silent?: boolean }) {
  const voice = path.join(WORK, 'voice_final.wav');
  const a = o.silent ? (fs.copyFileSync(o.rawVoice, voice), { code: 0, err: '' }) : await ff(['-i', o.rawVoice, '-af', (o.voiceFilter || 'highpass=f=80,afftdn=nr=10:nf=-40,acompressor=threshold=-20dB:ratio=3:attack=5:release=120:makeup=2,loudnorm=I=-16:TP=-1.5:LRA=9'), '-ar', '48000', '-ac', '1', voice]);
  if (a.code !== 0) throw new EditError(`FFmpeg could not clean the voice: ${a.err.slice(-400)}`);
  const list = path.join(WORK, 'segments.txt');
  fs.writeFileSync(list, Array.from({ length: o.count }, (_, i) => `file '${path.join(WORK, `seg_${String(i).padStart(4, '0')}.mp4`).replace(/'/g, "'\\''")}'`).join('\n'));
  const finalDuration = o.finalDuration;
  const addMusic = String(JOB.background_music ?? 'true') !== 'false';
  let m;
  if (addMusic) {
    const text = String(o.musicText || `${JOB.title || ''} ${JOB.instructions || ''}`).slice(0, 12000);
    const mood = musicMoodForText(text);
    const music = path.join(WORK, 'original_score.wav');
    const seed = `${JOB_ID}:${String(JOB.title || '').slice(0, 120)}:${text.slice(0, 240)}`;
    log(`Composing an original ${mood} score for this edit…`);
    makeOriginalMusic(finalDuration, mood, seed, music);
    // Music ducks under speech and rises slightly in pauses; the vocal itself is left intact.
    m = await ff(['-f', 'concat', '-safe', '0', '-i', list, '-i', voice, '-stream_loop', '-1', '-i', music,
      '-filter_complex', '[2:a]volume=0.20[bed];[bed][1:a]sidechaincompress=threshold=0.025:ratio=8:attack=18:release=280[ducked];[1:a][ducked]amix=inputs=2:duration=first:weights=1 0.90:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=9[mix]',
      '-map', '0:v:0', '-map', '[mix]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', finalDuration.toFixed(3), '-shortest', '-movflags', '+faststart', o.out]);
  } else {
    m = await ff(['-f', 'concat', '-safe', '0', '-i', list, '-i', voice, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-shortest', '-movflags', '+faststart', o.out]);
  }
  if (m.code !== 0 || !fs.existsSync(o.out)) throw new EditError(`FFmpeg could not join the video: ${m.err.slice(-500)}`);
}

async function render(opts: { video: string; vInfo: Awaited<ReturnType<typeof probe>>; segs: Seg[]; W: number; H: number; ass: string | null; pcm: Buffer; out: string; musicText?: string; onProgress: (done: number, total: number) => void }) {
  const { segs, W, H, vInfo } = opts;
  const framing = framingFilter(vInfo, W, H);
  const subs = opts.ass ? `,subtitles=filename='${escF(opts.ass)}':fontsdir='${escF(FONTS)}'` : '';
  let t = 0, done = 0;
  const jobs = segs.map((g, i) => { const j = { g, i, t }; t += g.frames / FPS; return j; });
  const threads = String(Math.max(1, Math.round((CPUS * 1.5) / Math.min(CPUS, segs.length))));
  await pool(jobs, CPUS, async ({ g, i, t: at }) => {
    const d = g.frames / FPS;
    const v0 = Math.max(0, Math.min(g.v0 ?? g.a0, Math.max(0, vInfo.duration - 0.05)));
    const file = path.join(WORK, `seg_${String(i).padStart(4, '0')}.mp4`);
    // Exactly g.frames frames: a clip running past the end of the video holds its last frame.
    const graph = `[0:v]setpts=PTS-STARTPTS,fps=${FPS},tpad=stop_mode=clone:stop_duration=${(d + 0.5).toFixed(3)},trim=end_frame=${g.frames},setpts=PTS-STARTPTS,${framing}${subs ? `,setpts=PTS-STARTPTS+${at.toFixed(4)}/TB${subs},setpts=PTS-STARTPTS` : ''},format=yuv420p[v]`;
    const r = await ff(['-ss', v0.toFixed(3), '-t', (d + 0.3).toFixed(3), '-i', opts.video, '-filter_complex', graph, '-map', '[v]', '-an',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-r', String(FPS), '-g', String(FPS * 4), '-threads', threads, '-pix_fmt', 'yuv420p', file], 1800_000);
    if (r.code !== 0 || !fs.existsSync(file)) throw new EditError(`FFmpeg could not render part ${i + 1}: ${r.err.slice(-500)}`);
    opts.onProgress(++done, segs.length);
  });
  // The voice: spliced exactly, then rumble cut, light denoise, gentle compression, broadcast loudness.
  const raw = path.join(WORK, 'voice_cut.wav');
  spliceVoice(opts.pcm, segs, raw);
  await finishVideo({ count: segs.length, finalDuration: segs.reduce((n, g) => n + g.frames / FPS, 0), rawVoice: raw, out: opts.out, musicText: opts.musicText });
}

// ---------------------------------------------------------------------------
// 7. Tutorial mode — a screen recording with NO voice gets one
//    Only used when the video has no voice-over of its own AND none was uploaded.
//    A recording that already has somebody talking is never narrated over.
// ---------------------------------------------------------------------------
type ProbeInfo = Awaited<ReturnType<typeof probe>>;
const MIN_VOICE_WORDS = 6;      // fewer spoken words than this → not a voice-over (speech-to-text also "hears" words in music)
const MIN_VOICE_WPM = 12;       // ...and fewer words per minute than this → not a voice-over
const SILENT_PEAK_DB = -48;     // a sound track that never gets louder than this is silence
const SPEECH_WPS = 2.6;         // narrator pace: sizes the scripts and times the subtitles when no voice engine is reachable
const LEAD = 0.2, TAIL = 0.3;   // breathing room around each narrated step (seconds)
const NARRATORS: Record<string, string[]> = {
  female: ['en-US-AvaMultilingualNeural', 'en-US-JennyNeural', 'en-US-AriaNeural'],
  male: ['en-US-AndrewMultilingualNeural', 'en-US-GuyNeural', 'en-US-ChristopherNeural']
};

/** Is somebody talking in this video? Silent, music-only and system-sound recordings are not voice-overs. */
async function detectVoiceover(video: string, vInfo: ProbeInfo, hint: string): Promise<{ voiced: boolean; words: W[] | null; why: string }> {
  if (!vInfo.hasAudio) return { voiced: false, words: null, why: 'the video has no sound' };
  const vd = await run('ffmpeg', ['-hide_banner', '-nostats', '-i', video, '-vn', '-af', 'volumedetect', '-f', 'null', '-'], 900_000);
  const peak = Number(/max_volume:\s*(-?[\d.]+)\s*dB/.exec(vd.err)?.[1]);
  if (Number.isFinite(peak) && peak < SILENT_PEAK_DB) return { voiced: false, words: null, why: `the sound track is silent (peak ${peak} dB)` };
  const words = await transcribe(video, vInfo.duration, hint);
  const perMinute = words.length / Math.max(1 / 60, vInfo.duration / 60);
  const voiced = words.length >= MIN_VOICE_WORDS && perMinute >= MIN_VOICE_WPM;
  log(`Voice check: ${words.length} spoken word(s) in ${vInfo.duration.toFixed(0)}s (${perMinute.toFixed(0)}/min) → ${voiced ? 'the video already has a voice-over, it is kept as it is' : 'no voice-over'}.`);
  return { voiced, words, why: voiced ? '' : `only ${words.length} spoken word(s) in ${vInfo.duration.toFixed(0)}s — music or system sounds, not a voice-over` };
}

/** Stretches of 1.2 s+ where the picture does not change at all (waiting, loading, idle). */
async function findFrozen(video: string, duration: number): Promise<{ s: number; e: number }[]> {
  const r = await run('ffmpeg', ['-hide_banner', '-nostats', '-i', video, '-an', '-vf', 'scale=640:-2,freezedetect=n=-58dB:d=1.2', '-f', 'null', '-'], 1800_000);
  const starts = [...r.err.matchAll(/freeze_start:\s*([\d.]+)/g)].map((m) => Number(m[1]));
  const ends = [...r.err.matchAll(/freeze_end:\s*([\d.]+)/g)].map((m) => Number(m[1]));
  return starts.map((s, i) => ({ s, e: ends[i] ?? duration })).filter((x) => x.e - x.s >= 1.2);
}

interface Scene { start: number; end: number; speed: number; narration: string }

/** Whatever the AI returned becomes forward-running, non-overlapping scenes inside the recording. */
function cleanScenes(raw: any, duration: number): Scene[] {
  const list = (Array.isArray(raw) ? raw : [])
    .map((x: any) => ({ start: Number(x?.start), end: Number(x?.end), speed: Number(x?.speed), narration: String(x?.narration || '').replace(/\s+/g, ' ').trim().slice(0, 700) }))
    .filter((x: Scene) => Number.isFinite(x.start) && Number.isFinite(x.end))
    .sort((a: Scene, b: Scene) => a.start - b.start);
  const out: Scene[] = [];
  let cursor = 0;
  for (const x of list) {
    const start = Math.max(cursor, x.start, 0);
    const end = Math.min(duration, x.end);
    if (end - start < 0.8) continue;
    out.push({ start, end, speed: Number.isFinite(x.speed) ? Math.max(1, Math.min(4, x.speed)) : 1, narration: x.narration });
    cursor = end;
  }
  return out;
}

async function planTutorial(frames: { t: number; d: string }[], frozen: { s: number; e: number }[], duration: number): Promise<{ title: string; description: string; scenes: Scene[]; model: string } | null> {
  // Test hook (never set in production): a ready plan.
  if (ENV.EDITOR_TUTORIAL_PLAN && fs.existsSync(ENV.EDITOR_TUTORIAL_PLAN)) {
    const j = JSON.parse(fs.readFileSync(ENV.EDITOR_TUTORIAL_PLAN, 'utf8'));
    return { title: String(j.title || ''), description: String(j.description || ''), scenes: cleanScenes(j.scenes, duration), model: 'test' };
  }
  const notes = String(JOB.instructions || '').trim();
  const user = `You are a top how-to video editor. A creator screen-recorded themselves doing something and said NOTHING. Turn the raw recording into a finished tutorial for social media: cut the waste and write the voice-over.

THE CREATOR'S NOTES: ${notes || '(none — work out what the recording teaches from the frames)'}
VIDEO TITLE: ${JOB.title || '(untitled)'}
RECORDING LENGTH: ${duration.toFixed(1)} s

WHAT IS ON SCREEN (a frame every few seconds; "IDLE:" = nothing is happening):
${frames.map((f) => `${f.t.toFixed(1)}s: ${f.d || '(unknown)'}`).join('\n')}

STRETCHES WHERE THE PICTURE DOES NOT CHANGE AT ALL (waiting or loading — but slow typing can show up here too, so check the frames): ${frozen.length ? frozen.slice(0, 40).map((x) => `${x.s.toFixed(1)}–${x.e.toFixed(1)}s`).join(', ') : 'none'}

Decide:
1. KEEP only what teaches. CUT: idle waiting, loading screens, mistakes and undoing them, repeated attempts (keep the one that worked), aimless cursor wandering, switching to unrelated apps, notifications, fumbling at the start or the end. SPEED UP (speed 2–4) steps that are needed but slow: long typing, progress bars, scrolling. Never reorder: scenes run forward through the recording and must not overlap.
2. Cover every step a viewer needs to repeat the process; never drop a needed step just to be short. A good result is usually 35–70% of the original length (never longer) unless the notes say otherwise.
3. Write the VOICE-OVER of each scene: friendly, second person, present tense, simple spoken language ("Now tap Export…"). Say exactly what to click or tap, using the button / menu names visible in the frames. NEVER invent screens, buttons or features that the frames do not show. Open the first scene with a one-sentence hook that says what the viewer will learn; end the last scene with a short closing line. Write in the language the creator's notes ask for; otherwise English.
4. Each narration must be speakable inside its scene: at most ${SPEECH_WPS} words per second of ((end − start) ÷ speed), plus 6 words. (If a step really needs more words, make the scene longer.)
5. Scene times are SECONDS of the original recording: 0 ≤ start < end ≤ ${duration.toFixed(1)}.

Return ONLY JSON: {"title": "catchy title, max 70 characters", "description": "1–2 sentence social caption followed by 4–6 hashtags", "scenes": [{"start": 0.0, "end": 8.5, "speed": 1, "narration": "..."}]}`;
  for await (const a of LLM.attempts({ system: 'You are an expert tutorial video editor. You answer with one JSON object.', user, json: true, temperature: 0.3, maxTokens: 7000, timeoutMs: 150000, task: 'editor_plan' })) {
    try {
      const j = extractJsonObject(a.text);
      const scenes = cleanScenes(j.scenes, duration);
      if (!scenes.length || !scenes.some((s) => s.narration.split(' ').length >= 3)) throw new Error('no usable scenes');
      return { title: String(j.title || '').slice(0, 100), description: String(j.description || '').slice(0, 900), scenes, model: `${a.provider}/${a.model}` };
    } catch (e: any) { log(`Tutorial plan from ${a.provider}/${a.model} unusable (${e?.message}).`); }
  }
  return null;
}

// --- the voice -------------------------------------------------------------
interface Spoken { file: string; dur: number; words: { w: string; s: number; e: number }[] }
let ttsDead = false; // set when the voice engine cannot be used on this runner: the rest is subtitles only

function estimateTimes(tokens: string[], dur: number): { w: string; s: number; e: number }[] {
  const weights = tokens.map((t) => t.length + 2);
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  let t = 0;
  return tokens.map((w, i) => { const d = (weights[i] / sum) * dur; const x = { w, s: t, e: t + d }; t += d; return x; });
}

/** One scene's narration through the neural voice (edge-tts), with word timings. Null when the engine is unusable. */
async function speak(text: string, voices: string[], idx: number): Promise<Spoken | null> {
  if (ttsDead) return null;
  const python = ENV.PYTHON || 'python3';
  const txt = path.join(WORK, `nar_${idx}.txt`), mp3 = path.join(WORK, `nar_${idx}.mp3`), wj = path.join(WORK, `nar_${idx}.json`);
  fs.writeFileSync(txt, text);
  const tokens = text.split(/\s+/).filter(Boolean);
  const attempts = [{ v: voices[0], rate: '+4%' }, { v: voices[1] || voices[0], rate: '+4%' }, { v: voices[0], rate: '+0%' }];
  let blocked = 0;
  for (let a = 0; a < attempts.length; a++) {
    if (ttsDead) return null;
    if (a > 0) await sleep(1500 * a);
    for (const f of [mp3, wj]) { try { fs.unlinkSync(f); } catch {} }
    // "--rate=+4%" (with "="): a value starting with "-" would otherwise be read as a new option.
    const r = await run(python, [path.join(HERE, 'tts.py'), `--text-file=${txt}`, `--voice=${attempts[a].v}`, `--rate=${attempts[a].rate}`, `--out-audio=${mp3}`, `--out-words=${wj}`], 180_000);
    const dur = fs.existsSync(mp3) ? (await probe(mp3)).duration : 0;
    if (r.code === 0 && dur > 0.3) {
      let timed: Spoken['words'] = [];
      try {
        const ws = JSON.parse(fs.readFileSync(wj, 'utf8')).words || [];
        // Use the engine's timings only when they line up with the script word for word.
        if (ws.length === tokens.length) timed = tokens.map((w, i) => ({ w, s: Number(ws[i].start), e: Number(ws[i].end) }));
      } catch {}
      return { file: mp3, dur, words: timed.length ? timed : estimateTimes(tokens, dur) };
    }
    const why = r.err.trim().split('\n').filter(Boolean).slice(-1)[0] || `exit ${r.code}`;
    log(`Narration ${idx + 1}: ${attempts[a].v} failed (${why.slice(0, 160)}).`);
    if (/No module named|ENOENT|command not found/i.test(why)) { ttsDead = true; log('The voice engine is not installed on this runner — the video gets subtitles only.'); return null; }
    if (/\b403\b/.test(why) && ++blocked >= 2) { ttsDead = true; log('The speech service refuses this runner — the video gets subtitles only.'); return null; }
  }
  return null;
}

// --- the timeline ------------------------------------------------------------
interface Slot { scene: Scene; spoken: Spoken | null; clip: number; rate: number; outDur: number; frames: number; at: number }

/**
 * Each scene lasts as long as its narration needs (or its sped-up picture, if longer). When the voice needs more
 * time than the picture has, the picture slows down (never below 0.75×) and then holds its last frame.
 */
function buildSlots(scenes: Scene[], spokens: (Spoken | null)[]): Slot[] {
  const slots: Slot[] = [];
  let at = 0;
  scenes.forEach((scene, i) => {
    const spoken = spokens[i];
    const clip = scene.end - scene.start;
    const base = clip / scene.speed;
    const need = spoken ? spoken.dur + LEAD + TAIL : 0;
    const frames = Math.max(Math.round(FPS * 0.8), Math.round(Math.max(base, need) * FPS));
    const outDur = frames / FPS;
    const rate = Math.max(0.75, Math.min(scene.speed, clip / outDur));
    slots.push({ scene, spoken, clip, rate, outDur, frames, at });
    at += outDur;
  });
  return slots;
}

function placeCaptions(slots: Slot[]): { placed: Placed[]; extra: string[] } {
  const placed: Placed[] = [];
  const extra: string[] = [];
  for (const sl of slots) {
    if (sl.spoken) {
      sl.spoken.words.forEach((w, k, all) => {
        const txt = w.w.replace(/[{}\\]/g, '').trim();
        if (!txt) return;
        placed.push({ w: txt, s: sl.at + LEAD + w.s, e: Math.min(sl.at + sl.outDur, sl.at + LEAD + w.e), end: /[.!?]$/.test(txt) || k === all.length - 1 });
      });
    }
    if (sl.rate >= 1.5) extra.push(`Dialogue: 1,${assTime(sl.at + 0.1)},${assTime(sl.at + sl.outDur - 0.1)},Badge,,0,0,0,,${Math.round(sl.rate)}× speed`);
  }
  return { placed, extra };
}

function writeWav48(dst: Int16Array, out: string) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0); header.writeUInt32LE(36 + dst.length * 2, 4); header.write('WAVE', 8);
  header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(48000, 24); header.writeUInt32LE(96000, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write('data', 36); header.writeUInt32LE(dst.length * 2, 40);
  fs.writeFileSync(out, Buffer.concat([header, Buffer.from(dst.buffer, dst.byteOffset, dst.byteLength)]));
}

/** Every narration laid onto one 48 kHz mono track at its place on the timeline. False when there is no voice at all. */
async function writeNarrationTrack(slots: Slot[], out: string): Promise<boolean> {
  const total = Math.round(slots.reduce((n, s) => n + s.frames, 0) * (48000 / FPS));
  const dst = new Int16Array(total);
  let any = false;
  for (let i = 0; i < slots.length; i++) {
    const sp = slots[i].spoken;
    if (!sp?.file) continue;
    const pcmFile = path.join(WORK, `nar_${i}.pcm`);
    if ((await ff(['-i', sp.file, '-vn', '-ac', '1', '-ar', '48000', '-f', 's16le', pcmFile])).code !== 0) continue;
    const buf = fs.readFileSync(pcmFile);
    const src = new Int16Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length - (buf.length % 2)));
    const off = Math.round((slots[i].at + LEAD) * 48000);
    for (let k = 0; k < src.length && off + k < total; k++) dst[off + k] = Math.max(-32768, Math.min(32767, dst[off + k] + src[k]));
    any = true;
  }
  writeWav48(dst, out);
  return any;
}

async function renderSlots(video: string, vInfo: ProbeInfo, slots: Slot[], W: number, H: number, ass: string | null, onProgress: (done: number, total: number) => void) {
  const framing = framingFilter(vInfo, W, H);
  const subs = ass ? `,subtitles=filename='${escF(ass)}':fontsdir='${escF(FONTS)}'` : '';
  const threads = String(Math.max(1, Math.round((CPUS * 1.5) / Math.min(CPUS, slots.length))));
  let done = 0;
  await pool(slots, CPUS, async (sl, i) => {
    const v0 = Math.max(0, Math.min(sl.scene.start, Math.max(0, vInfo.duration - 0.05)));
    const file = path.join(WORK, `seg_${String(i).padStart(4, '0')}.mp4`);
    // Exactly sl.frames frames: the clip is sped up / slowed down to fit, and holds its last frame if it runs out.
    const pad = Math.max(0, sl.outDur - sl.clip / sl.rate) + 0.6;
    const graph = `[0:v]setpts=(PTS-STARTPTS)/${sl.rate.toFixed(4)},fps=${FPS},tpad=stop_mode=clone:stop_duration=${pad.toFixed(3)},trim=end_frame=${sl.frames},setpts=PTS-STARTPTS,${framing}${subs ? `,setpts=PTS-STARTPTS+${sl.at.toFixed(4)}/TB${subs},setpts=PTS-STARTPTS` : ''},format=yuv420p[v]`;
    const r = await ff(['-ss', v0.toFixed(3), '-t', (sl.clip + 0.3).toFixed(3), '-i', video, '-filter_complex', graph, '-map', '[v]', '-an',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-r', String(FPS), '-g', String(FPS * 4), '-threads', threads, '-pix_fmt', 'yuv420p', file], 1800_000);
    if (r.code !== 0 || !fs.existsSync(file)) throw new EditError(`FFmpeg could not render step ${i + 1}: ${r.err.slice(-500)}`);
    onProgress(++done, slots.length);
  });
}

/** A silent screen recording in, a finished narrated tutorial out. */
async function tutorialEdit(o: { video: string; vInfo: ProbeInfo; reason: string; t0: number }) {
  const { video, vInfo, t0 } = o;
  const dur = vInfo.duration;
  if (dur < 3) throw new EditError('The recording is too short to edit (under 3 seconds).');
  const lap = (() => { let l = Date.now(); return (what: string) => { const n = Date.now(); log(`⏱ ${what}: ${((n - l) / 1000).toFixed(1)}s`); l = n; }; })();
  await report('running', '2/5 Watching your recording', 18, `No voice-over found (${o.reason}) — the AI will watch the recording, cut the dead parts and narrate it.`);
  const hook = !!(ENV.EDITOR_TUTORIAL_PLAN && fs.existsSync(ENV.EDITOR_TUTORIAL_PLAN));
  const [frames, frozen] = await Promise.all([
    hook ? Promise.resolve([] as { t: number; d: string }[]) : describeFrames(video, dur, true).catch((e) => { log(`Watching the video failed (${e?.message}).`); return [] as { t: number; d: string }[]; }),
    findFrozen(video, dur).catch(() => [] as { s: number; e: number }[])
  ]);
  lap('watching');
  if (!hook && frames.filter((f) => f.d).length < Math.max(3, Math.floor(frames.length / 3))) throw new EditError('The AI could not look at the recording right now (the vision service did not answer). Please try again in a few minutes.');
  log(`Watched ${frames.length} frame(s); ${frozen.length} frozen stretch(es) (${frozen.reduce((n, x) => n + x.e - x.s, 0).toFixed(0)}s of waiting).`);

  await report('running', '3/5 Planning the tutorial', 36, 'Deciding what to keep and writing the voice-over…');
  const plan = await planTutorial(frames, frozen, dur);
  if (!plan) throw new EditError('The AI could not plan this tutorial. Please try again, or describe what the recording shows in the notes box.');
  const keptSeconds = plan.scenes.reduce((n, s) => n + (s.end - s.start), 0);
  log(`Plan (${plan.model}): ${plan.scenes.length} step(s), keeping ${keptSeconds.toFixed(0)}s of ${dur.toFixed(0)}s. Title: "${plan.title}".`);
  lap('plan');

  await report('running', '4/5 Recording the voice-over', 50, `Narrating ${plan.scenes.length} step(s)…`);
  const voices = NARRATORS[String(JOB.narration_voice) === 'male' ? 'male' : 'female'];
  const spoken = await pool(plan.scenes, 3, (sc, i) => (sc.narration ? speak(sc.narration, voices, i) : Promise.resolve(null)));
  const narrated = spoken.some((s) => s?.file);
  // No voice engine: the script still goes on screen as subtitles, timed at a natural reading pace.
  const spokens = plan.scenes.map((sc, i) => {
    if (spoken[i] || !sc.narration) return spoken[i];
    const tokens = sc.narration.split(/\s+/).filter(Boolean);
    const d = tokens.length / SPEECH_WPS;
    return { file: '', dur: d, words: estimateTimes(tokens, d) } as Spoken;
  });
  if (!narrated) log('⚠ No voice could be produced — the tutorial is finished with subtitles only.');
  lap('voice-over');

  const slots = buildSlots(plan.scenes, spokens);
  const finalDur = slots.reduce((n, s) => n + s.frames / FPS, 0);
  log(`Timeline: ${slots.length} step(s), ${finalDur.toFixed(1)}s (from ${dur.toFixed(1)}s). ${slots.filter((s) => s.rate >= 1.5).length} sped up, ${slots.filter((s) => s.rate < 0.99).length} slowed to fit the voice.`);

  await report('running', '5/5 Compiling the final video', 60, `Rendering ${slots.length} step(s) on ${CPUS} cores…`);
  const [W, H] = frameSize(String(JOB.format || 'auto'), vInfo.width, vInfo.height);
  let ass: string | null = null;
  if (JOB.captions !== 'false') {
    ass = path.join(WORK, 'captions.ass');
    const { placed, extra } = placeCaptions(slots);
    fs.writeFileSync(ass, captionsFromPlaced(placed, W, H, extra));
  }
  await renderSlots(video, vInfo, slots, W, H, ass, (d, n) => {
    if (Date.now() - lastReport > 8000) report('running', `5/5 Compiling the final video (${d}/${n})`, Math.round(60 + (28 * d) / n));
  });
  const rawVoice = path.join(WORK, 'voice_cut.wav');
  const hasVoice = await writeNarrationTrack(slots, rawVoice);
  const out = path.join(WORK, 'edited.mp4');
  const script = plan.scenes.map((s) => s.narration).filter(Boolean).join(' ');
  await finishVideo({
    count: slots.length, finalDuration: finalDur, rawVoice, out, silent: !hasVoice,
    voiceFilter: 'highpass=f=70,acompressor=threshold=-22dB:ratio=2.5:attack=5:release=120:makeup=2,loudnorm=I=-16:TP=-1.5:LRA=9',
    musicText: `tutorial ${plan.title} ${JOB.title || ''} ${JOB.instructions || ''} ${script}`.slice(0, 12000)
  });
  lap('render');
  const oInfo = await probe(out);
  log(`Rendered ${W}x${H}, ${oInfo.duration.toFixed(1)}s, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB${ass ? ', subtitles on the picture' : ''}${hasVoice ? ', AI voice-over' : ''}.`);

  await report('running', 'Uploading the finished video', 92, 'Uploading…');
  const up = await sendFile(CP.output?.result, out);
  const result = {
    size: up.size, captions: !!ass, burnedCaptions: !!ass, duration: +oInfo.duration.toFixed(2), width: W, height: H,
    originalDuration: +dur.toFixed(2), removedSeconds: Math.max(0, +(dur - finalDur).toFixed(1)), fillers: 0, stutters: 0, retakes: 0,
    keptSentences: slots.length, totalSentences: slots.length, title: plan.title, description: plan.description, planModel: plan.model,
    mode: 'tutorial', narrated: hasVoice, steps: slots.length, narration: script.slice(0, 6000), seconds: Math.round((Date.now() - t0) / 1000)
  };
  await report('done', 'Your video is ready', 100, `✅ Done in ${((Date.now() - t0) / 1000).toFixed(0)}s — ${result.duration}s tutorial from a ${dur.toFixed(0)}s silent recording${hasVoice ? ', narrated' : ' (subtitles only — the voice engine was unavailable)'}.`,
    { result, ...(up.parts ? { outputParts: { result: up.parts } } : {}) });
}

// ---------------------------------------------------------------------------
async function main() {
  const t0 = Date.now();
  const lap = (() => { let l = Date.now(); return (what: string) => { const n = Date.now(); log(`⏱ ${what}: ${((n - l) / 1000).toFixed(1)}s`); l = n; }; })();
  log(`AI AUTO VIDEO EDITOR — job ${JOB_ID} (${CPUS} CPUs)`);
  await report('running', '1/5 Downloading your files', 8, 'GitHub runner started the edit.');
  const video = path.join(WORK, `source_video${path.extname(CP.files?.video?.name || '.mp4') || '.mp4'}`);
  let voice = CP.files?.audio ? path.join(WORK, `source_audio${path.extname(CP.files.audio.name || '.m4a') || '.m4a'}`) : '';
  const [gotVideo] = await Promise.all([fetchFile(CP.files?.video, video), voice ? fetchFile(CP.files.audio, voice) : Promise.resolve(false)]);
  if (!gotVideo) {
    if (!fs.existsSync(ENV.EDITOR_VIDEO || '')) throw new EditError('No video was provided.');
    fs.copyFileSync(ENV.EDITOR_VIDEO!, video);
  }
  if (!voice && ENV.EDITOR_AUDIO) voice = ENV.EDITOR_AUDIO;
  lap('download');
  const vInfo = await probe(video);
  if (!vInfo.hasVideo) throw new EditError('The video file has no video stream.');
  const separate = !!voice;
  // Is there a voice-over? A separate recording always counts. Otherwise the video's own sound is checked:
  // no voice at all → tutorial mode (the AI narrates it); somebody talking → the normal pipeline, untouched.
  const autoNarrate = String(JOB.auto_narrate ?? 'true') !== 'false';
  let preWords: W[] | null = null;
  if (!separate) {
    const det = await detectVoiceover(video, vInfo, [JOB.title, JOB.instructions].filter(Boolean).join('. '));
    if (!det.voiced) {
      if (!autoNarrate) throw new EditError(vInfo.hasAudio ? 'No speech was found in the video, and the AI voice-over is switched off.' : 'The video has no sound and no voice recording was uploaded (the AI voice-over is switched off).');
      await tutorialEdit({ video, vInfo, reason: det.why, t0 });
      return;
    }
    preWords = det.words;
  }
  // One decode of the voice: 48 kHz mono PCM for the cuts, the loudness map and the final mix.
  const pcmFile = path.join(WORK, 'voice.pcm');
  const dec = await ff(['-i', separate ? voice : video, '-vn', '-ac', '1', '-ar', '48000', '-f', 's16le', pcmFile]);
  if (dec.code !== 0 || !fs.existsSync(pcmFile)) throw new EditError(`Could not read the sound: ${dec.err.slice(-300)}`);
  const pcm = fs.readFileSync(pcmFile);
  const audioDur = pcm.length / 2 / 48000;
  const voiceWav = path.join(WORK, 'voice.wav');
  await ff(['-f', 's16le', '-ar', '48000', '-ac', '1', '-i', pcmFile, voiceWav]);
  log(`Video ${vInfo.width}x${vInfo.height}, ${vInfo.duration.toFixed(1)}s; voice ${audioDur.toFixed(1)}s (${separate ? 'separate recording' : 'from the video'}).`);

  await report('running', '2/5 Listening to your voice', 20, separate ? 'Transcribing the voice and watching the video at the same time…' : 'Transcribing the voice word by word…');
  const hint = [JOB.title, JOB.instructions].filter(Boolean).join('. ');
  // In parallel: the words, the loudness map and (separate voice) what the video shows.
  const framesP = separate ? describeFrames(video, vInfo.duration).catch((e) => { log(`Watching the video failed (${e?.message}) — placing clips by time.`); return []; }) : Promise.resolve([]);
  const [words, energy] = await Promise.all([preWords ? Promise.resolve(preWords) : transcribe(voiceWav, audioDur, hint), Energy.of(pcmFile)]);
  lap('transcription');
  if (words.length < 3) throw new EditError('No speech was found in the recording.');
  const { kept: cleanWords, removed, fillers, stutters, restarts, restartTexts } = clean(words);
  const list = sentences(words, cleanWords, removed);
  const groups = retakeGroups(list);
  log(`${list.length} sentences; removed ${fillers} filler word(s), ${stutters} stutter(s), ${restarts} restarted phrase(s); ${groups.length} retake group(s).`);
  if (restartTexts.length) log(`Restarted phrases cut: ${restartTexts.slice(0, 6).map((t) => `"${t}…"`).join(', ')}`);
  for (const g of groups.slice(0, 8)) log(`Retake group: ${g.map((i) => `#${i} "${list[i].text.slice(0, 50)}"`).join(' | ')} → best #${bestTake(list, g)}`);

  await report('running', '3/5 Editing: keeping the best parts', 38, `Cleaned ${fillers + stutters + restarts} slips. Found ${groups.length} retake${groups.length === 1 ? '' : 's'}. Deciding what to keep…`);
  const plan = await planEdit(list, groups, String(JOB.instructions || ''));
  const kept = plan.keep.map((id) => list.find((x) => x.id === id)!).filter(Boolean);
  log(`Edit plan (${plan.model}): keeping ${kept.length}/${list.length} sentences. ${plan.removed.slice(0, 6).map((r) => `#${r.id} ${r.reason}`).join('; ')}`);
  lap('edit plan');

  const segs = buildTimeline(kept, words, energy, audioDur, JOB.remove_silences !== 'false');
  if (!segs.length) throw new EditError('Nothing was left to keep after editing.');
  const finalDur = segs.reduce((n, g) => n + g.frames / FPS, 0);
  log(`Timeline: ${segs.length} pieces, ${finalDur.toFixed(1)}s (from ${audioDur.toFixed(1)}s — ${(audioDur - finalDur).toFixed(1)}s removed). Every cut sits in a pause between words.`);

  await report('running', '4/5 Matching the video to what you say', 55, separate ? 'Matching each sentence to the video…' : 'Voice and video are in sync — cutting both together.');
  if (separate) {
    const frames = await framesP;
    const sentStarts = await matchVideo(kept, frames, vInfo.duration, audioDur);
    const sentA0 = new Map<number, number>();
    segs.forEach((g) => { if (!sentA0.has(g.si)) sentA0.set(g.si, g.a0); });
    // Each piece inside a sentence continues that sentence's clip.
    segs.forEach((g) => { g.v0 = sentStarts[g.si] + (g.a0 - sentA0.get(g.si)!); });
  } else {
    segs.forEach((g) => { g.v0 = g.a0; }); // same recording: cut picture and sound together
  }
  lap('matching');

  await report('running', '5/5 Compiling the final video', 60, `Rendering ${segs.length} pieces on ${CPUS} cores…`);
  const [W, H] = frameSize(String(JOB.format || 'auto'), vInfo.width, vInfo.height);
  let ass: string | null = null;
  if (JOB.captions !== 'false') {
    ass = path.join(WORK, 'captions.ass');
    fs.writeFileSync(ass, captionsAss(segs, kept, W, H));
  }
  const out = path.join(WORK, 'edited.mp4');
  await render({ video, vInfo, segs, W, H, ass, pcm, out, musicText: `${JOB.title || ''} ${JOB.instructions || ''} ${kept.map((x) => x.text).join(' ')}`.slice(0, 12000), onProgress: (d, n) => {
    if (Date.now() - lastReport > 8000) report('running', `5/5 Compiling the final video (${d}/${n})`, Math.round(60 + (28 * d) / n));
  } });
  lap('render');
  const oInfo = await probe(out);
  log(`Rendered ${W}x${H}, ${oInfo.duration.toFixed(1)}s, ${(fs.statSync(out).size / 1e6).toFixed(1)} MB${ass ? ', captions on the picture' : ''}.`);

  await report('running', 'Uploading the finished video', 92, 'Uploading…');
  const up = await sendFile(CP.output?.result, out);
  lap('upload');
  const result = { size: up.size, captions: !!ass, burnedCaptions: !!ass, duration: +oInfo.duration.toFixed(2), width: W, height: H,
    originalDuration: +audioDur.toFixed(2), removedSeconds: +(audioDur - finalDur).toFixed(1), fillers, stutters: stutters + restarts, retakes: groups.length,
    keptSentences: kept.length, totalSentences: list.length, title: plan.title, planModel: plan.model, seconds: Math.round((Date.now() - t0) / 1000) };
  await report('done', 'Your video is ready', 100, `✅ Done in ${((Date.now() - t0) / 1000).toFixed(0)}s — ${result.duration}s video, ${result.removedSeconds}s of filler/pauses/retakes removed.`,
    { result, ...(up.parts ? { outputParts: { result: up.parts } } : {}) });
}

main().then(() => process.exit(0)).catch(async (err: any) => {
  const msg = err?.message || String(err);
  console.error(`❌ ${msg}`);
  if (!(err instanceof EditError)) console.error(err?.stack || err);
  await report('failed', 'Failed', 0, `❌ ${msg}`, { error: msg });
  process.exit(1);
});
