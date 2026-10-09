/**
 * The 2D animation and podcast channels — the cloud pipeline (called by
 * renderer.mts for categories "animation" and "podcast").
 *
 *   1. Script (Gemini → Groq): a fight choreography, a short film with a cast
 *      and dialogue, or a podcast conversation between the designed hosts.
 *   2. Voices: every line is recorded with its own voice (Edge neural TTS,
 *      several lines at once), with word timings for captions and lip-sync.
 *   3. Sound: the dialogue track, sound effects timed to every blow (made
 *      here, no samples), and an original music bed ducked under the voices.
 *   4. Picture: the anim stage (stage/animStage.ts) in headless Chrome.
 *
 * Kept free of renderer internals: everything it needs comes in the kit.
 */
import fs from 'node:fs';
import path from 'node:path';
import { choreograph, vsFighters, planRound, roundHealth, equipSides, ARCHETYPES, type VsSide, type VsRound } from './stickman.ts';
import { normalizeStudio } from './studio.ts';
import { arenaOrder, ARENA_IDS } from './arenas.ts';

export interface Word { text: string; start: number; end: number; speaker?: string }
export interface Kit {
  CFG: any;
  W: number; H: number; FPS: number; IS_SHORTS: boolean;
  WORK_DIR: string; HERE: string;
  LLM: any; extractJsonObject: (t: string) => any;
  log: (m: string) => void;
  reportStatus: (status: any, step: string, progress: number, line: string) => Promise<void>;
  run: (cmd: string, args: string[], opts?: { timeoutMs?: number }) => Promise<{ code: number; stdout: Buffer; stderr: string }>;
  probeDuration: (f: string) => Promise<number>;
  /** Renders the stage bundle `anim.js` with this job; the final audio is muxed in. */
  runStage: (o: { stageFile: string; job: any; audioFinal: string; files: Record<string, string>; duration: number }) => Promise<{ ok: boolean; reason?: string; character?: string }>;
  composeMusic: (mood: string, seconds: number, seed: string) => { L: Float32Array; R: Float32Array; sampleRate: number };
  automateLevel: (L: Float32Array, R: Float32Array, sr: number, o: { speech: { start: number; end: number }[]; speechGain: number; pauseGain: number }) => void;
  eqForVoice: (L: Float32Array, R: Float32Array, sr: number) => void;
  pastTitles: string[];
  /** Voice of the automation's presenter when it hosts the podcast. */
  hostVoice?: string;
  PipelineError: new (code: string, message: string) => Error;
}
/** Strong moments for the thumbnail (seconds; w = how strong). */
export interface Highlight { t: number; w: number }
const STRONG = new Set(['excited', 'surprised', 'laugh', 'happy', 'angry', 'scared', 'crying', 'shocked']);
const emotionMoments = (cues: Record<string, { t: number; tag: string }[]>, w = 0.6): Highlight[] => Object.values(cues).flat().filter((c) => STRONG.has(c.tag)).map((c) => ({ t: c.t, w }));
export interface AnimResult { title: string; description: string; hashtags: string[]; tags: string[]; script: string; durationSec: number; character: string; model: string; sources: string[]; showName?: string; highlights?: Highlight[] }

const SR = 48000;
const sleep = (ms: number) => new Promise((z) => setTimeout(z, ms));
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const clean = (s: any, n = 400) => String(s ?? '').replace(/\s+/g, ' ').trim().slice(0, n);

// ---------------------------------------------------------------------------
// Voices
// ---------------------------------------------------------------------------
const VOICE_POOL = {
  female: ['en-US-AvaMultilingualNeural', 'en-US-EmmaMultilingualNeural', 'en-US-AriaNeural', 'en-US-JennyNeural', 'en-GB-SoniaNeural', 'en-US-MichelleNeural'],
  male: ['en-US-AndrewMultilingualNeural', 'en-US-BrianMultilingualNeural', 'en-US-GuyNeural', 'en-US-ChristopherNeural', 'en-GB-RyanNeural', 'en-US-EricNeural'],
};
const NARRATOR = 'en-US-AndrewMultilingualNeural';

/**
 * Spoken languages for podcasts: native neural voices (the multilingual
 * English voices fill in when a language has few). A language with no
 * neural voice (Yoruba, Igbo, Hausa) is spoken in English with Nigerian voices.
 */
const SPOKEN: Record<string, { name: string; female: string[]; male: string[] }> = {
  korean: { name: 'Korean', female: ['ko-KR-SunHiNeural'], male: ['ko-KR-InJoonNeural', 'ko-KR-HyunsuMultilingualNeural'] },
  japanese: { name: 'Japanese', female: ['ja-JP-NanamiNeural'], male: ['ja-JP-KeitaNeural'] },
  chinese: { name: 'Mandarin Chinese', female: ['zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural'], male: ['zh-CN-YunxiNeural', 'zh-CN-YunjianNeural'] },
  french: { name: 'French', female: ['fr-FR-DeniseNeural', 'fr-FR-EloiseNeural'], male: ['fr-FR-HenriNeural', 'fr-FR-RemyMultilingualNeural'] },
  spanish: { name: 'Spanish', female: ['es-ES-ElviraNeural', 'es-MX-DaliaNeural'], male: ['es-ES-AlvaroNeural', 'es-MX-JorgeNeural'] },
  portuguese: { name: 'Portuguese', female: ['pt-BR-FranciscaNeural', 'pt-PT-RaquelNeural'], male: ['pt-BR-AntonioNeural', 'pt-PT-DuarteNeural'] },
  german: { name: 'German', female: ['de-DE-KatjaNeural', 'de-DE-AmalaNeural'], male: ['de-DE-ConradNeural', 'de-DE-KillianNeural'] },
  italian: { name: 'Italian', female: ['it-IT-ElsaNeural', 'it-IT-IsabellaNeural'], male: ['it-IT-DiegoNeural', 'it-IT-GiuseppeMultilingualNeural'] },
  arabic: { name: 'Arabic', female: ['ar-SA-ZariyahNeural', 'ar-EG-SalmaNeural'], male: ['ar-SA-HamedNeural', 'ar-EG-ShakirNeural'] },
  hindi: { name: 'Hindi', female: ['hi-IN-SwaraNeural'], male: ['hi-IN-MadhurNeural'] },
  swahili: { name: 'Swahili', female: ['sw-KE-ZuriNeural', 'sw-TZ-RehemaNeural'], male: ['sw-KE-RafikiNeural', 'sw-TZ-DaudiNeural'] },
  zulu: { name: 'Zulu', female: ['zu-ZA-ThandoNeural'], male: ['zu-ZA-ThembaNeural'] },
  amharic: { name: 'Amharic', female: ['am-ET-MekdesNeural'], male: ['am-ET-AmehaNeural'] },
  turkish: { name: 'Turkish', female: ['tr-TR-EmelNeural'], male: ['tr-TR-AhmetNeural'] },
  russian: { name: 'Russian', female: ['ru-RU-SvetlanaNeural', 'ru-RU-DariyaNeural'], male: ['ru-RU-DmitryNeural'] },
  indonesian: { name: 'Indonesian', female: ['id-ID-GadisNeural'], male: ['id-ID-ArdiNeural'] },
  tagalog: { name: 'Filipino (Tagalog)', female: ['fil-PH-BlessicaNeural'], male: ['fil-PH-AngeloNeural'] },
  vietnamese: { name: 'Vietnamese', female: ['vi-VN-HoaiMyNeural'], male: ['vi-VN-NamMinhNeural'] },
  thai: { name: 'Thai', female: ['th-TH-PremwadeeNeural'], male: ['th-TH-NiwatNeural'] },
  dutch: { name: 'Dutch', female: ['nl-NL-ColetteNeural', 'nl-NL-FennaNeural'], male: ['nl-NL-MaartenNeural'] },
  polish: { name: 'Polish', female: ['pl-PL-ZofiaNeural', 'pl-PL-AgnieszkaNeural'], male: ['pl-PL-MarekNeural'] },
  'nigerian pidgin': { name: 'Nigerian Pidgin', female: ['en-NG-EzinneNeural'], male: ['en-NG-AbeoNeural'] },
};
const NIGERIAN = { female: ['en-NG-EzinneNeural'], male: ['en-NG-AbeoNeural'] };
/** The podcast's spoken language: { name, voices } — null for English. */
function podcastLanguage(CFG: any): { key: string; name: string; female: string[]; male: string[]; english: boolean } | null {
  const raw = String(CFG.podcastLanguage || '').trim().toLowerCase();
  if (!raw || raw === 'english') return null;
  const key = Object.keys(SPOKEN).find((k) => raw === k || raw.startsWith(k) || raw.includes(k));
  if (key) return { key, ...SPOKEN[key], english: key === 'nigerian pidgin' };
  if (/yoruba|igbo|hausa/.test(raw)) return { key: raw, name: 'English', ...NIGERIAN, english: true };
  return null;
}

export const EMO_TAGS = ['neutral', 'happy', 'excited', 'sad', 'crying', 'serious', 'worried', 'scared', 'surprised', 'angry', 'calm', 'curious', 'laugh'];
export const GESTURE_TAGS = ['look_left', 'look_right', 'look_up', 'think', 'nod', 'shake_head', 'lean_in', 'point', 'explain', 'count', 'wave', 'shrug', 'hands_up', 'hand_chest', 'fist'];
const ALL_TAGS = new Set([...EMO_TAGS, ...GESTURE_TAGS]);

/** "[happy] Hello there [nod] friend" → text + tags at word indexes. */
function parseTags(raw: string): { text: string; tags: { word: number; tag: string }[] } {
  const tags: { word: number; tag: string }[] = [];
  const words: string[] = [];
  for (const tok of String(raw || '').split(/\s+/)) {
    const m = tok.match(/^\[([a-z_ ]+)\]$/i);
    if (m) { const tag = m[1].toLowerCase().replace(/\s+/g, '_'); if (ALL_TAGS.has(tag)) tags.push({ word: words.length, tag }); continue; }
    const cleaned = tok.replace(/\[[^\]]*\]/g, '');
    if (cleaned) words.push(cleaned);
  }
  return { text: words.join(' ').replace(/[*_#`>~]/g, '').trim(), tags };
}

interface Clip { file: string; duration: number; words: Word[] }

/** One line, one voice → a 48 kHz mono WAV + word timings (Edge TTS; flite offline). */
async function speak(kit: Kit, text: string, voice: string, rate: string, gender: 'female' | 'male', id: string): Promise<Clip> {
  const base = path.join(kit.WORK_DIR, `line_${id}`);
  fs.writeFileSync(`${base}.txt`, text);
  const wav = `${base}.wav`;
  if (!kit.CFG.offline) {
    for (let a = 0; a < 3; a++) {
      if (a) await sleep(1500 * a);
      const mp3 = `${base}.mp3`, wf = `${base}.json`;
      const r = await kit.run(process.env.PYTHON || 'python3', [path.join(kit.HERE, 'tts.py'), `--text-file=${base}.txt`, `--voice=${voice}`, `--rate=${rate}`, `--out-audio=${mp3}`, `--out-words=${wf}`], { timeoutMs: 120000 });
      if (r.code === 0 && fs.existsSync(mp3)) {
        await kit.run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', mp3, '-ac', '1', '-ar', String(SR), wav]);
        const duration = await kit.probeDuration(wav);
        let words: Word[] = [];
        try { words = (JSON.parse(fs.readFileSync(wf, 'utf8')).words || []).map((w: any) => ({ text: String(w.text), start: +w.start, end: +w.end })); } catch {}
        if (duration > 0.2) return { file: wav, duration, words: attachPunctuation(words, text, duration) };
      }
      kit.log(`TTS (${voice}) attempt ${a + 1} failed: ${r.stderr.trim().split('\n').slice(-1)[0] || r.code}`);
    }
  }
  // Offline / last resort: ffmpeg's flite voice.
  await kit.run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `flite=textfile=${base}.txt:voice=${gender === 'male' ? 'kal16' : 'slt'}`, '-ac', '1', '-ar', String(SR), wav]);
  const duration = await kit.probeDuration(wav);
  if (!(duration > 0.2)) throw new kit.PipelineError('tts_failed', `Could not record the line "${text.slice(0, 60)}".`);
  return { file: wav, duration, words: attachPunctuation([], text, duration), neural: false } as any;
}

/** The script's own tokens (with punctuation) on the TTS timings; estimated when there are none. */
function attachPunctuation(bounds: Word[], text: string, duration: number): Word[] {
  const tokens = text.split(/\s+/).filter(Boolean);
  const norm = (s: string) => String(s ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
  if (bounds.length >= Math.max(1, tokens.length * 0.6)) {
    const out: Word[] = [];
    let ti = 0;
    for (const b of bounds) {
      const nb = norm(b.text);
      if (!nb) continue;
      let j = ti;
      while (j < tokens.length && norm(tokens[j]) !== nb && j < ti + 3) j++;
      const tok = j < tokens.length && norm(tokens[j]) === nb ? tokens[j] : b.text;
      if (j < tokens.length && norm(tokens[j]) === nb) ti = j + 1;
      out.push({ text: tok, start: b.start, end: b.end });
    }
    return out;
  }
  const weight = (t: string) => Math.max(1, t.replace(/[^a-z]/gi, '').length) + (/[.!?]$/.test(t) ? 4 : /[,;:]$/.test(t) ? 2 : 0);
  const total = tokens.reduce((n, t) => n + weight(t), 0) || 1;
  let t = 0.05;
  const span = Math.max(0.3, duration - 0.15);
  return tokens.map((tok) => { const d = (weight(tok) / total) * span; const w = { text: tok, start: t, end: t + d * 0.85 }; t += d; return w; });
}

async function readPcm(kit: Kit, file: string): Promise<Float32Array> {
  const raw = `${file}.f32`;
  await kit.run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', file, '-ac', '1', '-ar', String(SR), '-f', 'f32le', raw]);
  const b = fs.readFileSync(raw);
  return new Float32Array(b.buffer, b.byteOffset, Math.floor(b.length / 4)).slice();
}

function writeWav(file: string, chans: Float32Array[]) {
  const n = chans[0].length, c = chans.length;
  const buf = Buffer.alloc(44 + n * c * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * c * 2, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(c, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * c * 2, 28); buf.writeUInt16LE(c * 2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * c * 2, 40);
  let o = 44;
  for (let i = 0; i < n; i++) for (let k = 0; k < c; k++) { buf.writeInt16LE(Math.round(clamp(chans[k][i], -1, 1) * 32767), o); o += 2; }
  fs.writeFileSync(file, buf);
}

/** Linear resample to 48 kHz, exactly n samples. */
function resample(x: Float32Array, rate: number, n: number): Float32Array {
  if (rate === SR && x.length >= n) return x.subarray(0, n);
  const out = new Float32Array(n), k = rate / SR;
  for (let i = 0; i < n; i++) { const p = i * k, j = Math.floor(p), f = p - j; out[i] = j + 1 < x.length ? x[j] * (1 - f) + x[j + 1] * f : 0; }
  return out;
}

// ---------------------------------------------------------------------------
// Sound effects (synthesised: no third-party samples)
// ---------------------------------------------------------------------------
function sfxInto(out: Float32Array, at: number, kind: string, seed: number) {
  let s = (seed * 9301 + 49297) % 233280;
  const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280 * 2 - 1; };
  const i0 = Math.max(0, Math.round(at * SR));
  const put = (k: number, v: number) => { if (i0 + k < out.length) out[i0 + k] += v; };
  const thump = (f0: number, f1: number, dur: number, gain: number) => {
    let ph = 0;
    for (let k = 0; k < dur * SR; k++) { const t = k / SR, f = f0 + (f1 - f0) * (t / dur); ph += (2 * Math.PI * f) / SR; put(k, Math.sin(ph) * gain * Math.exp(-t * 18 / dur * 0.25)); }
  };
  const noise = (dur: number, gain: number, lp: number, attack = 0.005, shape: 'decay' | 'swell' = 'decay') => {
    let y = 0;
    for (let k = 0; k < dur * SR; k++) {
      const t = k / SR;
      y += lp * (rnd() - y);
      const env = shape === 'swell' ? Math.sin(Math.PI * t / dur) ** 2 : Math.min(1, t / attack) * Math.exp(-t * 6 / dur);
      put(k, y * gain * env);
    }
  };
  switch (kind) {
    case 'jab': thump(140, 70, 0.1, 0.55); noise(0.05, 0.5, 0.5); break;
    case 'punch': thump(120, 50, 0.16, 0.8); noise(0.06, 0.7, 0.45); break;
    case 'kick': thump(95, 40, 0.22, 0.95); noise(0.08, 0.7, 0.35); break;
    case 'thud': thump(70, 35, 0.35, 0.8); noise(0.25, 0.35, 0.08); break;
    case 'block': thump(220, 160, 0.06, 0.45); noise(0.04, 0.6, 0.8); break;
    case 'slash': { noise(0.28, 0.5, 0.9, 0.02, 'swell'); let ph = 0; for (let k = 0; k < 0.9 * SR; k++) { ph += (2 * Math.PI * 1850) / SR; put(k + 0.12 * SR, Math.sin(ph) * 0.18 * Math.exp(-(k / SR) * 5)); } break; }
    case 'whoosh': noise(0.3, 0.45, 0.12, 0.05, 'swell'); break;
    case 'dash': noise(0.16, 0.55, 0.3, 0.01, 'swell'); break;
    case 'blink': { let ph = 0; for (let k = 0; k < 0.14 * SR; k++) { const f = 2400 - 1800 * (k / (0.14 * SR)); ph += (2 * Math.PI * f) / SR; put(k, Math.sin(ph) * 0.16 * (1 - k / (0.14 * SR))); } noise(0.1, 0.3, 0.6, 0.005); break; }
    case 'heavy': thump(90, 32, 0.4, 1.0); noise(0.12, 0.9, 0.3); noise(0.5, 0.25, 0.04); break;
    case 'boom': thump(60, 24, 0.9, 1.0); noise(0.9, 0.5, 0.03, 0.01); noise(0.15, 0.6, 0.5); break;
    case 'crumble': for (let i = 0; i < 9; i++) { const o = Math.round((0.02 + 0.07 * i + 0.03 * Math.abs(rnd())) * SR); let y = 0; for (let k = 0; k < 0.06 * SR; k++) { y += 0.3 * (rnd() - y); put(o + k, y * 0.4 * Math.exp(-k / (0.012 * SR))); } } break;
    case 'zap': { let ph = 0; for (let k = 0; k < 0.35 * SR; k++) { const tt = k / SR, f = 300 + 1700 * tt / 0.35; ph += (2 * Math.PI * f) / SR; put(k, (Math.sin(ph) > 0 ? 1 : -1) * 0.07 * Math.exp(-tt * 4)); } noise(0.3, 0.3, 0.7, 0.01, 'swell'); break; }
    case 'beam': { let ph = 0; for (let k = 0; k < 0.9 * SR; k++) { const tt = k / SR; ph += (2 * Math.PI * (140 + 8 * Math.sin(tt * 60))) / SR; const env = Math.min(1, tt / 0.05) * Math.max(0, 1 - Math.max(0, tt - 0.6) / 0.3); put(k, (Math.sin(ph) + 0.5 * Math.sin(ph * 3.01)) * 0.18 * env); } noise(0.9, 0.35, 0.8, 0.03, 'swell'); break; }
    case 'bell': for (const [f, g] of [[880, 0.22], [1320, 0.12], [2210, 0.07]] as [number, number][]) { let ph = 0; for (let k = 0; k < 1.2 * SR; k++) { ph += (2 * Math.PI * f) / SR; put(k, Math.sin(ph) * g * Math.exp(-(k / SR) * 3.2)); } } break;
    case 'slide': noise(0.5, 0.3, 0.05, 0.03, 'swell'); break;
    // Weapons (synthesised): a sharp crack with a body thump and a short room tail.
    case 'gun': noise(0.018, 1.0, 0.95, 0.0005); thump(180, 55, 0.12, 0.9); noise(0.25, 0.35, 0.18, 0.002); noise(0.6, 0.12, 0.03, 0.01); break;
    case 'bulletHit': thump(240, 90, 0.05, 0.45); noise(0.06, 0.5, 0.7, 0.001); break;
    case 'ricochet': { let ph = 0; for (let k = 0; k < 0.32 * SR; k++) { const tt = k / SR, f = 3200 - 2200 * (tt / 0.32); ph += (2 * Math.PI * f) / SR; put(k, Math.sin(ph) * 0.12 * Math.exp(-tt * 9)); } noise(0.04, 0.4, 0.8, 0.001); break; }
    case 'shell': for (let i = 0; i < 3; i++) { const o = Math.round((0.06 * i + 0.02 * Math.abs(rnd())) * SR), f = 4200 + 900 * i; let ph = 0; for (let k = 0; k < 0.07 * SR; k++) { ph += (2 * Math.PI * f) / SR; put(o + k, Math.sin(ph) * 0.06 * Math.exp(-k / (0.012 * SR)) / (i + 1)); } } break;
    case 'stab': noise(0.1, 0.55, 0.85, 0.01, 'swell'); thump(150, 70, 0.08, 0.5); break;
    case 'clang': for (const [f, g] of [[1250, 0.22], [2730, 0.14], [4100, 0.08], [5900, 0.05]] as [number, number][]) { let ph = 0; for (let k = 0; k < 0.7 * SR; k++) { ph += (2 * Math.PI * f) / SR; put(k, Math.sin(ph) * g * Math.exp(-(k / SR) * 7)); } } noise(0.03, 0.6, 0.9, 0.001); break;
  }
}

// ---------------------------------------------------------------------------
// LLM helper
// ---------------------------------------------------------------------------
async function askJson(kit: Kit, system: string, user: string, validate: (j: any) => string | null, task: string): Promise<{ j: any; model: string } | null> {
  if (kit.CFG.offline || !kit.LLM?.hasKeys) return null;
  for await (const a of kit.LLM.attempts({ system, user, json: true, temperature: 0.85, maxTokens: 8000, timeoutMs: 150000, task })) {
    try {
      const j = kit.extractJsonObject(a.text);
      const why = validate(j);
      if (why) throw new Error(why);
      return { j, model: `${a.provider}/${a.model}` };
    } catch (e: any) { kit.log(`Script from ${a.provider}/${a.model} unusable (${e?.message}).`); }
  }
  return null;
}

const avoidTitles = (kit: Kit) => kit.pastTitles.slice(-40).map((t) => `- ${t}`).join('\n') || '(none yet)';
const cleanTags = (a: any): string[] => (Array.isArray(a) ? a : []).map((h: any) => String(h).toLowerCase().replace(/[^a-z0-9]/g, '')).filter((h: string) => h.length > 2 && h.length < 30).slice(0, 8);

// ---------------------------------------------------------------------------
// Headlines for podcast topics (fresh, so hosts talk about what is happening)
// ---------------------------------------------------------------------------
async function headlinesFor(kit: Kit, topic: string): Promise<{ title: string; source: string }[]> {
  if (kit.CFG.offline || !topic) return [];
  try {
    const url = `${kit.CFG.newsBase || 'https://news.google.com/rss/search'}?q=${encodeURIComponent(`${topic} when:3d`)}&hl=en-US&gl=US&ceid=US:en`;
    const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 AnimatoAutoPoster/4.0' }, signal: AbortSignal.timeout(15000) });
    if (!r.ok) return [];
    const xml = await r.text();
    const strip = (s: string) => s.replace(/<!\[CDATA\[|\]\]>/g, '').replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, ' ').trim();
    const out: { title: string; source: string }[] = [];
    for (const m of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
      const title = strip((m[1].match(/<title>([\s\S]*?)<\/title>/) || [])[1] || '');
      const source = strip((m[1].match(/<source[^>]*>([\s\S]*?)<\/source>/) || [])[1] || '');
      if (title) out.push({ title: source && title.endsWith(` - ${source}`) ? title.slice(0, -(source.length + 3)) : title, source });
      if (out.length >= 8) break;
    }
    return out;
  } catch { return []; }
}

// ---------------------------------------------------------------------------
// Timeline assembly (dialogue + effects + music)
// ---------------------------------------------------------------------------
interface Placed { at: number; clip: Clip; speaker: string; tags: { word: number; tag: string }[] }

async function mixAudio(kit: Kit, placed: Placed[], sfx: { t: number; kind: string }[], duration: number, mood: string, musicLevelDb: number): Promise<{ dialogue: string; final: string; words: Word[]; cues: Record<string, { t: number; tag: string }[]> }> {
  const n = Math.ceil((duration + 0.5) * SR);
  const dia = new Float32Array(n);
  const words: Word[] = [];
  const cues: Record<string, { t: number; tag: string }[]> = {};
  for (const p of placed) {
    const pcm = await readPcm(kit, p.clip.file);
    const i0 = Math.round(p.at * SR);
    for (let k = 0; k < pcm.length && i0 + k < n; k++) dia[i0 + k] += pcm[k];
    for (const w of p.clip.words) words.push({ text: w.text, start: +(p.at + w.start).toFixed(3), end: +(p.at + w.end).toFixed(3), speaker: p.speaker });
    for (const tg of p.tags) {
      const w = p.clip.words[Math.min(tg.word, p.clip.words.length - 1)];
      (cues[p.speaker] ||= []).push({ t: +(p.at + (w ? w.start : 0) - 0.05).toFixed(3), tag: tg.tag });
    }
  }
  // Level the voices (peak ≈ -3 dBFS) before mixing.
  let peak = 0; for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(dia[i]));
  if (peak > 0) { const g = 0.7 / peak; for (let i = 0; i < n; i++) dia[i] *= g; }
  const dialogue = path.join(kit.WORK_DIR, 'dialogue.wav');
  writeWav(dialogue, [dia]);
  // Effects
  const fx = new Float32Array(n);
  sfx.forEach((e, i) => sfxInto(fx, e.t, e.kind, i + 1));
  // Music, ducked under speech, lifted in pauses.
  let L: Float32Array, R: Float32Array;
  try {
    const m = kit.composeMusic(mood, duration + 0.5, `${kit.CFG.campaignId}:${kit.CFG.partNumber}:${mood}`);
    kit.eqForVoice(m.L, m.R, m.sampleRate);
    L = resample(m.L, m.sampleRate, n); R = resample(m.R, m.sampleRate, n);
    const speech = placed.map((p) => ({ start: p.at, end: p.at + p.clip.duration }));
    const rms = (a: Float32Array) => { let s = 0, c = 0; for (let i = 0; i < a.length; i += 7) { s += a[i] * a[i]; c++; } return Math.sqrt(s / Math.max(1, c)) || 1e-6; };
    const voiceRms = (() => { let s = 0, c = 0; for (const p of speech) for (let i = Math.round(p.start * SR); i < Math.min(n, p.end * SR); i += 7) { s += dia[i] * dia[i]; c++; } return Math.sqrt(s / Math.max(1, c)) || 0.1; })();
    const g = (voiceRms / rms(L)) * 10 ** (-musicLevelDb / 20);
    kit.automateLevel(L, R, SR, { speech, speechGain: speech.length ? g : g * 2, pauseGain: g * 2 });
  } catch (e: any) {
    kit.log(`⚠️ Music could not be composed (${e?.message}) — voices and effects only.`);
    L = new Float32Array(n); R = new Float32Array(n);
  }
  const outL = new Float32Array(n), outR = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = dia[i] + fx[i] * 0.55;
    outL[i] = v + (L[i] || 0); outR[i] = v + (R[i] || 0);
  }
  // Soft limiter
  for (const ch of [outL, outR]) for (let i = 0; i < n; i++) { const x = ch[i]; ch[i] = Math.tanh(x * 1.1) * 0.92; }
  const final = path.join(kit.WORK_DIR, 'final_mix.wav');
  writeWav(final, [outL, outR]);
  return { dialogue, final, words: words.sort((a, b) => a.start - b.start), cues };
}

/** The film's idea: the typed title/idea, else a rotated genre topic (never the style name itself). */
const ideaOf = (CFG: any) => {
  const t = String(CFG.topic || '').trim();
  if (t) return t;
  const g = String(CFG.subGenre || '').trim();
  return g && !/stick|short film|2d/i.test(g) ? `a ${g.toLowerCase()} story` : '';
};
const PALETTE = ['#22d3ee', '#f472b6', '#facc15', '#a3e635', '#fb923c', '#c084fc'];
const AUTO_NAMES = {
  female: ['Amara', 'Zara', 'Ifeoma', 'Chiara', 'Aiko', 'Priya', 'Sofia', 'Nadia', 'Leila', 'Imani', 'Freya', 'Ayana', 'Mei', 'Elena', 'Thandi', 'Yara', 'Noor', 'Keira', 'Dalia', 'Halima', 'Ines', 'Talia', 'Nia', 'Adaeze', 'Farida', 'Camila', 'Zuri', 'Hana', 'Naomi', 'Bianca', 'Kiara', 'Sade', 'Isla', 'Marisol', 'Tolu', 'Kemi', 'Chioma', 'Lola', 'Esi', 'Ama', 'Rina', 'Jade', 'Ruby', 'Maya', 'Ada', 'Lina', 'Ivy', 'Nora'],
  male: ['Kofi', 'Mateo', 'Idris', 'Arjun', 'Lucas', 'Tobi', 'Hassan', 'Noel', 'Ezra', 'Kian', 'Andre', 'Chidi', 'Rafael', 'Oliver', 'Kenji', 'Malik', 'Diego', 'Femi', 'Nikhil', 'Samir', 'Jonas', 'Tariq', 'Elias', 'Bruno', 'Amani', 'Hugo', 'Zane', 'Emeka', 'Rowan', 'Yusuf', 'Caleb', 'Adrian', 'Omar', 'Theo', 'Ravi', 'Leo', 'Musa', 'Kwame', 'Enzo', 'Ibrahim', 'Tunde', 'Seun', 'Dayo', 'Max', 'Sam', 'Jay', 'Ben', 'Nate']
};
const AUTO_PERSONALITIES = ['warm and funny', 'sharp skeptic', 'hype energy', 'calm expert', 'curious newcomer', 'witty storyteller', 'straight-talking', 'optimistic dreamer', 'dry humour', 'passionate debater'];
/** A brand-new cast for one episode: names, genders, personalities and face seeds (deterministic per episode). */
function autoCast(seed: string, n: number, avoid: string[] = []): { autoHost: true; name: string; gender: 'female' | 'male'; personality: string; seed: string }[] {
  let h = 2166136261; for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  const r = () => { h = Math.imul(h ^ (h >>> 13), 0x5bd1e995) >>> 0; h ^= h >>> 15; return (h % 100000) / 100000; };
  const out: { autoHost: true; name: string; gender: 'female' | 'male'; personality: string; seed: string }[] = [];
  const used = new Set(avoid.map((x) => String(x ?? '').toLowerCase()));
  const usedP = new Set<string>();
  for (let i = 0; i < n; i++) {
    const gender: 'female' | 'male' = i === 0 ? (r() < 0.5 ? 'female' : 'male') : (out[i - 1].gender === 'female' ? (r() < 0.7 ? 'male' : 'female') : (r() < 0.7 ? 'female' : 'male'));
    const list = AUTO_NAMES[gender];
    let name = list[Math.floor(r() * list.length)];
    for (let k = 0; k < 10 && used.has(name.toLowerCase()); k++) name = list[Math.floor(r() * list.length)];
    used.add(name.toLowerCase());
    let personality = AUTO_PERSONALITIES[Math.floor(r() * AUTO_PERSONALITIES.length)];
    for (let k = 0; k < 6 && usedP.has(personality); k++) personality = AUTO_PERSONALITIES[Math.floor(r() * AUTO_PERSONALITIES.length)];
    usedP.add(personality);
    out.push({ autoHost: true, name, gender, personality, seed: `autohost:${seed}:${i}:${name}` });
  }
  return out;
}
const lengthOf = (kit: Kit) => kit.IS_SHORTS;

// ===========================================================================
// PODCAST
// ===========================================================================
async function podcast(kit: Kit): Promise<AnimResult> {
  const CFG = kit.CFG;
  let specs: any[] = (Array.isArray(CFG.castSpecs) && CFG.castSpecs.length ? CFG.castSpecs : [null, null]).slice(0, 3);
  // Auto cast: brand-new hosts (or guests, when the automation's presenter hosts) for every episode.
  // Musicals: the singer is the GUEST of her own show; two brand-new hosts interview her.
  const singerMode = !!CFG.podcastSinger && !!specs[0];
  if (singerMode) {
    const singer = { ...specs[0], gender: specs[0]?.gender === 'male' ? 'male' : 'female' };
    const auto = autoCast(`${CFG.campaignId}:${CFG.partNumber}:hosts`, 2, [String(singer?.name || '')]);
    specs = [...auto, singer];
    kit.log(`Podcast hosts this episode: ${auto.map((a) => `${a.name} (${a.gender}, ${a.personality})`).join(', ')}; guest: ${clean(singer?.name, 24) || 'the singer'}.`);
  } else if (CFG.podcastCast === 'auto') {
    const keep = CFG.podcastGuests && specs[0] ? [specs[0]] : [];
    const auto = autoCast(`${CFG.campaignId}:${CFG.partNumber}`, Math.max(keep.length ? 1 : 2, Math.min(3 - keep.length, Number(CFG.podcastCastCount) || 2)), keep.map((k: any) => String(k?.name || '')));
    specs = [...keep, ...auto];
    kit.log(`Auto cast this episode: ${auto.map((a) => `${a.name} (${a.gender}, ${a.personality})`).join(', ')}`);
  }
  const studio = normalizeStudio(CFG.studio || {});
  const hosts = specs.map((sp, i) => {
    const gender: 'female' | 'male' = sp?.gender === 'male' ? 'male' : sp?.gender === 'female' ? 'female' : i % 2 ? 'male' : 'female';
    // An auto host has no designed look: the stage builds a unique face and outfit from its seed.
    const spec = sp && !sp.autoHost ? sp : null;
    return { id: `h${i + 1}`, name: clean(sp?.name, 24) || ['Maya', 'Jordan', 'Sam'][i], gender, spec, seed: sp?.seed ? String(sp.seed) : undefined, personality: clean(sp?.personality, 40), color: PALETTE[i] };
  });
  // In an automation's rotation, a podcast with no topics of its own talks about the automation's topics.
  const songs: { title: string; genre: string; lyrics: string }[] = singerMode ? (Array.isArray(CFG.podcastSongs) ? CFG.podcastSongs : []).map((x: any) => ({ title: clean(x?.title, 100), genre: clean(x?.genre, 60), lyrics: String(x?.lyrics || '').replace(/\[[^\]]*\]/g, ' ').replace(/[ \t]+/g, ' ').trim().slice(0, 600) })).filter((x: any) => x.title) : [];
  const about = singerMode ? (songs.length ? `the inspiration behind ${songs.map((x) => `"${x.title}"`).join(', ')}` : 'her music and her latest concert') : clean(CFG.podcastAbout, 300);
  const topic = clean(CFG.topic || about || (/^podcast$/i.test(String(CFG.subGenre || '')) ? '' : CFG.subGenre) || 'the biggest story this week', 200);
  const queries = about && !CFG.topic ? about.split(/\s*,\s*/).filter(Boolean).slice(0, 3) : [topic];
  const news: { title: string; source: string }[] = [];
  if (!singerMode) for (const q of queries) for (const h of await headlinesFor(kit, q)) if (news.length < 10 && !news.some((n) => n.title === h.title)) news.push(h);
  const shorts = lengthOf(kit);
  const n = shorts ? '12-16 lines, 130-160 words in total (under 60 seconds)' : '40-60 lines, 480-620 words in total (3-4 minutes)';
  const guestMode = !singerMode && !!CFG.podcastGuests && hosts.length > 1;
  const star = hosts[hosts.length - 1];
  const guestNames = hosts.slice(1).map((h) => h.name).join(' and ');
  const cast = singerMode
    ? `hosted by ${hosts[0].name} (${hosts[0].gender}, ${String(hosts[0].personality || '').toLowerCase() || 'warm'}) and ${hosts[1].name} (${hosts[1].gender}, ${String(hosts[1].personality || '').toLowerCase() || 'curious'}). Today's GUEST is the singer ${star.name} (${star.gender}) — the star of this channel. She is NOT a host: the hosts interview HER.
Speakers: 1 = ${hosts[0].name} (lead host), 2 = ${hosts[1].name} (co-host), 3 = ${star.name} (the singer, guest).
THE HOSTS OPEN: welcome viewers back to ${studio.showName}, then introduce ${star.name} with real excitement and say she is here to talk about her latest songs. The two hosts take turns asking; ${star.name} answers in the first person ("I wrote it when…").`
    : guestMode
    ? `hosted by ${hosts[0].name} (${hosts[0].gender}), the channel's regular presenter. Today's GUEST${hosts.length > 2 ? 'S' : ''} on the show: ${hosts.slice(1).map((h, i) => `${i + 2} = ${h.name} (${h.gender}${h.personality ? `, ${h.personality.toLowerCase()}` : ''})`).join('; ')}.
Speakers: 1 = ${hosts[0].name} (the host), ${hosts.slice(1).map((h, i) => `${i + 2} = ${h.name} (guest)`).join(', ')}.
THE HOST OPENS: welcomes the viewers back, then introduces the guest by name — e.g. "We have ${hosts[1].name} on the show today" — and says what they will talk about. The host asks the questions and steers; the guest${hosts.length > 2 ? 's bring' : ' brings'} expertise, strong opinions and predictions. The host thanks ${guestNames} by name at the end and asks viewers to follow`
    : `with ${hosts.length} hosts: ${hosts.map((h, i) => `${i + 1} = ${h.name} (${h.gender}${h.personality ? `, personality: ${h.personality.toLowerCase()}` : ''}${i === 0 ? ', the lead host who opens and closes' : ', co-host with their own opinions'})`).join('; ')}. Each host speaks in a way that matches their personality.`;
  const lang = podcastLanguage(CFG);
  const foreign = !!lang && !lang.english;
  const langRule = lang ? (foreign
    ? `LANGUAGE: the whole conversation is spoken in natural, everyday ${lang.name} (as native speakers really talk — not a translation). Every line ALSO has "en": a faithful, natural English translation of that line (it is shown as the English subtitle). Keep every line short — at most about 14 English words — so each subtitle fits on screen.\n`
    : `LANGUAGE: ${lang.name === 'English' ? 'English with a warm Nigerian flavour (a few local expressions are welcome)' : lang.name}.\n`) : '';
  const user = `${langRule}Write one episode of "${studio.showName}", a video podcast ${cast}.
TOPIC: ${topic}
${news.length ? `FRESH HEADLINES (last 3 days). Facts may ONLY come from these; everything else is clearly the hosts' opinion, questions or reactions:\n${news.map((h, i) => `${i + 1}. ${h.title}${h.source ? ` (${h.source})` : ''}`).join('\n')}` : 'No live headlines were found: talk about the topic in general terms — opinions, experiences, tips — and do not invent news, numbers, dates or quotes.'}
EPISODES ALREADY MADE (never repeat one of these angles):
${avoidTitles(kit)}

${singerMode ? `THE SONGS SHE RELEASED SINCE THE LAST EPISODE (talk about EACH one, in this order, by its exact title):
${songs.length ? songs.map((x, i) => `${i + 1}. "${x.title}"${x.genre ? ` (${x.genre})` : ''}${x.lyrics ? `\n   lyrics: ${x.lyrics.replace(/\n+/g, ' / ')}` : ''}`).join('\n') : '(no song list was saved — talk about her music in general, her last concert and what she is working on, without naming song titles)'}
INTERVIEW: for every song the hosts ask about the INSPIRATION — the moment, person or feeling behind it, what the lyrics really mean (she may quote one short line ONLY from the lyrics given above), how the band and backup singers built it, and how the crowd reacted at the concert. Follow-up questions, laughs and honest emotion. ${star.name} speaks about 45% of the lines, each host about a quarter. End with what she is working on next, the hosts thank ${star.name} by name and ask viewers to follow.
Never invent awards, chart positions, sales or real collaborators.
` : ''}${about && !CFG.topic && !singerMode ? `ANGLE: a roundup of the latest in ${about} — what just happened, why it matters, the hosts' honest takes, and bold predictions about the next big thing (clearly framed as speculation).\n` : ''}HOW IT SOUNDS: a real conversation between friends who know the subject — quick back-and-forth, reactions ("wait, really?"), a disagreement, a laugh, a clear takeaway. Short lines (one or two sentences each). Line 1 is a HOOK (a surprising fact or bold opinion, max 14 words). The last line thanks the audience and asks them to follow.
Performance tags (optional, before the word they apply to): ${[...EMO_TAGS, 'nod', 'shake_head', 'lean_in', 'point', 'explain', 'count', 'shrug', 'hands_up', 'hand_chest', 'think'].map((t) => `[${t}]`).join(' ')}.
${CFG.performanceNotes ? `WHAT IS WORKING ON THIS CHANNEL (learn from it):\n${CFG.performanceNotes}\n` : ''}LENGTH: ${n}.
${foreign
  ? `Return ONLY JSON: {"title": "catchy episode title in ${lang!.name} (max 60 chars)", "title_en": "the same title in English", "description": "2-3 sentences for YouTube, in English", "hashtags": ["5-8 specific hashtags without #"], "lines": [{"host": 1, "text": "[excited] the line in ${lang!.name}", "en": "the English translation"}]}`
  : `Return ONLY JSON: {"title": "catchy episode title (max 70 chars)", "description": "2-3 sentences for YouTube", "hashtags": ["5-8 specific hashtags without #"], "lines": [{"host": 1, "text": "[excited] Line text"}]}`}`;
  const got = await askJson(kit, 'You write natural, factual podcast conversations. You answer with one JSON object.', user, (j) => {
    if (!Array.isArray(j.lines) || j.lines.length < (shorts ? 8 : 25)) return 'too few lines';
    if (j.lines.some((l: any) => !(Number(l.host) >= 1 && Number(l.host) <= hosts.length) || !clean(l.text))) return 'bad line';
    if (foreign && j.lines.filter((l: any) => clean(l.en)).length < j.lines.length * 0.8) return 'missing English translations';
    return null;
  }, 'podcast_script');
  const script = got?.j || {
    title: singerMode ? `${star.name}: the stories behind the songs` : `${topic}: what everyone is missing`, description: singerMode ? `${star.name} joins ${studio.showName} to talk about ${topic}.` : `The hosts of ${studio.showName} talk about ${topic}.`, hashtags: singerMode ? ['podcast', 'music', 'interview'] : ['podcast', 'talk'],
    lines: [
      ...(singerMode ? [
      { host: 1, text: `[excited] Welcome back to ${studio.showName}! And today the star herself is here — ${star.name}!` },
      { host: 3, text: `[happy] Thank you so much, I'm so happy to be here.` },
      { host: 2, text: `[curious] So tell us — where did ${songs[0] ? `"${songs[0].title}"` : 'your latest song'} come from?` },
      { host: 3, text: `[hand_chest] Honestly, it came from a real moment in my life. I wrote it for everyone who needed to hear it.` },
      { host: 1, text: `[laugh] And the crowd sang every word at the concert!` },
      { host: 3, text: `[happy] That night I will never forget. Thank you all for listening.` },
      { host: 2, text: `[happy] Thank you, ${star.name}! Follow for the next episode.` },
      ] : [
      { host: 1, text: guestMode ? `[excited] Welcome back to ${studio.showName}! We have ${hosts[1].name} on the show today, and we are talking about ${topic}.` : `[excited] Welcome back to ${studio.showName}! Today we are talking about ${topic}.` },
      { host: 2, text: `[curious] Honestly, I have a strong opinion on this one.` },
      { host: 1, text: `[laugh] Of course you do. Go on then.` },
      { host: 2, text: `[explain] Most people only look at the headline, not what it means for them.` },
      { host: 1, text: `[nod] That's fair. So what should people actually do?` },
      { host: 2, text: `[count] Start small, stay curious, and check more than one source.` },
      { host: 1, text: `[happy] Great advice. Thanks for listening, and follow for the next episode!` },
    ]),
    ],
  };
  if (!got && !CFG.allowFallbackPublish && !CFG.offline && !CFG.dryRun) throw new kit.PipelineError('script_retry', 'The AI model on the runner could not write the podcast script this time. Nothing was posted; the next attempt runs automatically.');
  await kit.reportStatus('running', '2/5 Recording the hosts', 22, `Script ready: "${clean(script.title, 90)}" (${script.lines.length} lines${got ? `, ${got.model}` : ''}).`);

  // Voices: one per host, distinct.
  const used = new Set<string>();
  // Another language: native voices first, then the multilingual voices (they speak it too).
  const voicePool = lang ? { female: [...lang.female, 'en-US-AvaMultilingualNeural', 'en-US-EmmaMultilingualNeural'], male: [...lang.male, 'en-US-AndrewMultilingualNeural', 'en-US-BrianMultilingualNeural'] } : VOICE_POOL;
  const voiceOf = (g: 'female' | 'male', i: number) => { const v = voicePool[g].find((x) => !used.has(x)) || voicePool[g][i % voicePool[g].length]; used.add(v); return v; };
  const ownVoice = lang ? '' : kit.hostVoice;
  if ((guestMode || singerMode) && ownVoice) used.add(ownVoice);
  const voices = hosts.map((h, i) => (i === 0 && guestMode && ownVoice ? ownVoice : singerMode && i === hosts.length - 1 && ownVoice ? ownVoice : voiceOf(h.gender, i)));
  if (lang) kit.log(`Podcast language: ${lang.name}${foreign ? ' with English subtitles' : ''} — voices ${voices.join(', ')}.`);
  const lines = script.lines.map((l: any, i: number) => ({ host: clamp(Math.round(Number(l.host)), 1, hosts.length) - 1, ...parseTags(clean(l.text, 400)), en: clean(String(l.en || '').replace(/\[[^\]]*\]/g, ' '), 300), i })).filter((l: any) => l.text);
  const clips = await pool(lines, 4, (l: any) => speak(kit, l.text, voices[l.host], '+4%', hosts[l.host].gender, `p${l.i}`));
  // Natural pacing: quick exchanges, a beat longer on a change of speaker.
  let t = 0.5;
  const placed: Placed[] = [];
  lines.forEach((l: any, i: number) => {
    placed.push({ at: t, clip: clips[i], speaker: hosts[l.host].id, tags: l.tags });
    const next = lines[i + 1];
    t += clips[i].duration + (next && next.host !== l.host ? 0.18 : 0.32);
  });
  const duration = +(t + 1.6).toFixed(2);
  await kit.reportStatus('running', '3/5 Mixing the sound', 40, `Recorded ${lines.length} lines (${duration.toFixed(0)}s) with ${voices.join(', ')}.`);
  const mix = await mixAudio(kit, placed, [], duration, 'tech', 16);
  const job = {
    mode: 'podcast', width: kit.W, height: kit.H, fps: kit.FPS, duration, title: clean(script.title, 90), badge: `🎙 ${studio.showName.toUpperCase()}`, endCard: 'Follow for the next episode', accent: studio.accent,
    fontUrl: '/font/Poppins-Bold.ttf', audio: '/audio/dialogue.wav', words: mix.words,
    hosts: hosts.map((h) => ({ ...h, cues: mix.cues[h.id] || [] })), studio,
    // Another language: each spoken line on screen with its English translation underneath.
    ...(foreign ? { subtitles: lines.map((l: any, i: number) => ({ start: placed[i].at, end: placed[i].at + clips[i].duration, text: l.text, en: l.en, speaker: hosts[l.host].id })) } : {}),
  };
  const res = await render(kit, job, mix, duration);
  const text = lines.map((l: any) => `${hosts[l.host].name}: ${l.text}`).join('\n');
  return {
    highlights: emotionMoments(mix.cues),
    title: foreign && clean(script.title_en) ? clean(`${clean(script.title, 50)} | ${clean(script.title_en, 60)}`, 95) : clean(script.title, 95), description: `${foreign ? `🌐 ${lang!.name} with English subtitles.\n` : ''}${clean(script.description, 900)}\n\n${singerMode ? `Hosts: ${hosts[0].name} & ${hosts[1].name}. Guest: ${star.name}.${songs.length ? `\n\nSongs in this episode:\n${songs.map((x) => `• ${x.title}`).join('\n')}` : ''}` : guestMode ? `Host: ${hosts[0].name}. Guest${hosts.length > 2 ? 's' : ''}: ${guestNames}.` : `Hosts: ${hosts.map((h) => h.name).join(', ')}.`}${news.length ? `\n\nIn the news:\n${news.slice(0, 4).map((h) => `• ${h.title}${h.source ? ` (${h.source})` : ''}`).join('\n')}` : ''}`,
    hashtags: [...cleanTags(script.hashtags), 'podcast'], tags: [topic, studio.showName, 'podcast', ...hosts.map((h) => h.name)].map((x) => clean(x, 40)).filter(Boolean),
    script: text, durationSec: duration, character: res.character || 'podcast', model: got?.model || 'template', sources: news.slice(0, 4).map((h) => h.title), showName: studio.showName,
  };
}

// ===========================================================================
// FILM
// ===========================================================================
export const FILM_LOCATIONS = ['bedroom', 'living_room', 'office', 'coffee_shop', 'library', 'cabin', 'street', 'rooftop', 'city_night', 'garden', 'beach'];

/** Film genres offered in the app (and what they sound and look like). */
export const FILM_GENRES: Record<string, { mood: string; grade: string; brief: string }> = {
  romance: { mood: 'emotional', grade: 'romance', brief: 'a romance: longing, a meet-cute or a second chance, a heartfelt confession, a warm ending' },
  horror: { mood: 'horror', grade: 'horror', brief: 'horror: creeping dread, something wrong in an ordinary place, a scare, an unsettling final twist (frightening, never gory)' },
  comedy: { mood: 'comedy', grade: '', brief: 'a comedy: a silly misunderstanding that escalates, quick banter, a big funny payoff' },
  action: { mood: 'action', grade: '', brief: 'action: a chase or a race against time, high stakes, a daring plan, a triumphant finish' },
  mystery: { mood: 'mystery', grade: 'mystery', brief: 'a mystery: a strange event, clues the viewer can follow, a clever reveal' },
  drama: { mood: 'emotional', grade: '', brief: 'a drama: a hard choice between people who care about each other, an honest, moving resolution' },
  thriller: { mood: 'mystery', grade: 'mystery', brief: 'a thriller: a secret, a threat, rising tension, a sharp twist' },
  'sci-fi': { mood: 'tech', grade: '', brief: 'science fiction: one surprising invention or future idea changes an ordinary day' },
  fantasy: { mood: 'story', grade: '', brief: 'fantasy: a touch of magic enters the everyday world, a lesson learned' },
  family: { mood: 'story', grade: '', brief: 'a family story: siblings, parents or grandparents, a misunderstanding, love wins' },
  adventure: { mood: 'upbeat', grade: '', brief: 'an adventure: a quest, a discovery, friends who get each other through it' },
  friendship: { mood: 'upbeat', grade: '', brief: 'a friendship story: two friends tested, loyalty and laughter' },
};
const genreKey = (g: string) => { const x = String(g ?? '').toLowerCase(); return Object.keys(FILM_GENRES).find((k) => x.includes(k.replace('-', '')) || x.includes(k)) || (/love|romantic/.test(x) ? 'romance' : /scary|horror|ghost/.test(x) ? 'horror' : /funny|comed/.test(x) ? 'comedy' : /scifi|science|space|robot/.test(x) ? 'sci-fi' : ''); };

/** A fresh, random cast for every film: who they are is decided here, the writer names and styles them. */
function castBrief(seed: string): string {
  let h = 2166136261; for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  const r = () => { h = Math.imul(h ^ (h >>> 13), 0x5bd1e995) >>> 0; return (h % 10000) / 10000; };
  const pick = <T>(a: T[]) => a[Math.floor(r() * a.length) % a.length];
  const n = 2 + Math.floor(r() * 2.2);
  const ages = ['a teenager', 'a young adult', 'someone in their thirties', 'a middle-aged person', 'an older person'];
  const roles = ['student', 'nurse', 'chef', 'mechanic', 'artist', 'teacher', 'delivery rider', 'musician', 'shop owner', 'engineer', 'photographer', 'barista', 'detective', 'farmer', 'pilot', 'gamer', 'baker', 'coach', 'librarian', 'street dancer'];
  const letters = 'ABCDEFGHIJKLMNOPRSTVWYZ';
  const used = new Set<string>();
  return Array.from({ length: n }, (_, i) => {
    let L = pick(letters.split('')); while (used.has(L)) L = pick(letters.split('')); used.add(L);
    return `${i + 1}) ${r() < 0.5 ? 'female' : 'male'}, ${pick(ages)}, a ${pick(roles)}, first name starting with "${L}"`;
  }).join('; ');
}

async function film(kit: Kit): Promise<AnimResult> {
  const CFG = kit.CFG;
  const shorts = lengthOf(kit);
  const typed = String(CFG.topic || '').trim();
  const idea = clean(typed, 300);
  const gRaw = clean(CFG.storyGenre || (/stick|short film|2d/i.test(String(CFG.subGenre || '')) ? '' : CFG.subGenre) || '', 60);
  const gk = genreKey(gRaw);
  const G = gk ? FILM_GENRES[gk] : null;
  const genre = gk || gRaw;
  const cast = castBrief(`${CFG.campaignId}:${CFG.partNumber}:${Date.now() >> 16}`);
  const size = shorts ? '2-3 scenes, 10-16 dialogue lines in total, 110-150 words of dialogue (under 60 seconds)' : '5-8 scenes, 45-70 dialogue lines, 500-700 words of dialogue (3-4 minutes)';
  const user = `Write an ORIGINAL 2D animated short film${idea ? ` from this idea / title: "${idea}"` : ''}${genre ? `. GENRE: ${G ? G.brief : genre}` : ''}.
It is a complete story with a beginning, a turn and a satisfying ending — told almost entirely through dialogue between a small cast (all original; no existing characters, brands or celebrities).
THIS FILM'S CAST (brand-new characters — give each a fresh, uncommon first name and a distinct look; never reuse names from earlier films): ${cast}.
FILMS ALREADY MADE (make something different):
${avoidTitles(kit)}

RULES
- Locations (use these ids only): ${FILM_LOCATIONS.join(', ')}. Each scene has "time": "day" or "night", and a short "card" like "Maya's bedroom — 2 a.m.".
- "cast": the ids of the characters on screen in that scene (1-3). Only cast members speak, plus an optional "narrator" (use the narrator for at most one line per scene).
- Each character: id (c1, c2…), name, gender (female|male), and "look": a short visual description the animator uses — hair (style + colour), skin tone, clothes with colours, glasses/beard if any (e.g. "curly black hair, dark skin, yellow hoodie, round glasses").
- Lines are short and natural (max 2 sentences). Performance tags before words: ${[...EMO_TAGS, 'nod', 'shake_head', 'think', 'shrug', 'hands_up', 'hand_chest', 'point', 'fist', 'wave'].map((t) => `[${t}]`).join(' ')}.
- Scene 1 opens with a hook. The final scene resolves the story.
- Characters move like in a real film: they arrive, cross the room, leave. Write scenes so people can walk in and out (e.g. someone arrives mid-scene, someone storms off).
- LENGTH: ${size}.
Return ONLY JSON: {"title": "max 60 chars", "logline": "one sentence", "description": "2-3 sentences for YouTube", "hashtags": ["5-8 specific hashtags without #"], "mood": "story|emotional|mystery|horror|upbeat", "characters": [{"id": "c1", "name": "", "gender": "female", "look": ""}], "scenes": [{"location": "bedroom", "time": "night", "card": "", "cast": ["c1", "c2"], "lines": [{"speaker": "c1", "text": "[worried] …"}]}]}`;
  // Studio → Create animation: the user's own script and characters, used exactly as written.
  const injected = (CFG as any).studioFilm;
  const got = injected ? { j: injected, model: 'your script' } : await askJson(kit, 'You are an award-winning animation screenwriter. You answer with one JSON object.', user, (j) => {
    if (!Array.isArray(j.characters) || j.characters.length < 1 || !Array.isArray(j.scenes) || j.scenes.length < (shorts ? 1 : 3)) return 'missing cast or scenes';
    const ids = new Set(j.characters.map((c: any) => String(c.id)));
    const lines = j.scenes.flatMap((s: any) => (Array.isArray(s.lines) ? s.lines : []));
    if (lines.length < (shorts ? 6 : 25)) return 'too few lines';
    if (lines.some((l: any) => !clean(l.text) || (l.speaker !== 'narrator' && !ids.has(String(l.speaker))))) return 'a line has an unknown speaker';
    return null;
  }, 'film_script');
  const s = got?.j || {
    title: idea || 'The Last Train Home', logline: 'Two friends miss the last train and find out what really matters.', description: 'An original 2D short film.', hashtags: ['animation', 'shortfilm'], mood: 'emotional',
    characters: [{ id: 'c1', name: 'Maya', gender: 'female', look: 'curly black hair, yellow hoodie, round glasses' }, { id: 'c2', name: 'Leo', gender: 'male', look: 'short brown hair, green jacket, light stubble' }],
    scenes: [
      { location: 'street', time: 'night', card: 'Main Street — 11:58 p.m.', cast: ['c1', 'c2'], lines: [{ speaker: 'c1', text: '[worried] Leo, that was the last train.' }, { speaker: 'c2', text: '[shrug] Then I guess we walk.' }, { speaker: 'c1', text: '[surprised] Walk? It is ten miles!' }] },
      { location: 'rooftop', time: 'night', card: 'Later', cast: ['c1', 'c2'], lines: [{ speaker: 'c2', text: '[calm] You know, this is the best night we have had in years.' }, { speaker: 'c1', text: '[happy] Yeah. Missing that train was the best mistake.' }] },
    ],
  };
  if (!got && !CFG.allowFallbackPublish && !CFG.offline && !CFG.dryRun) throw new kit.PipelineError('script_retry', 'The AI model on the runner could not write the film script this time. Nothing was posted; the next attempt runs automatically.');
  const chars = (s.characters as any[]).slice(0, 4).map((c, i) => ({ id: clean(c.id, 12) || `c${i + 1}`, name: clean(c.name, 24) || `Character ${i + 1}`, gender: c.gender === 'male' ? 'male' as const : 'female' as const, look: clean(c.look, 200), color: PALETTE[i], seed: `${CFG.campaignId}:${CFG.partNumber}:${c.name}:${i}:${Date.now() >> 20}`, ...(c.spec ? { spec: c.spec } : {}) }));
  await kit.reportStatus('running', '2/5 Recording the cast', 22, `Script ready: "${clean(s.title, 80)}" — ${chars.map((c) => c.name).join(', ')}; ${s.scenes.length} scenes${got ? ` (${got.model})` : ''}.`);
  const used = new Set<string>([NARRATOR]);
  const voice: Record<string, string> = { narrator: NARRATOR };
  chars.forEach((c, i) => { const v = VOICE_POOL[c.gender].find((x) => !used.has(x)) || VOICE_POOL[c.gender][i % 6]; used.add(v); voice[c.id] = v; });
  const scenesIn = (s.scenes as any[]).map((sc, si) => ({
    location: FILM_LOCATIONS.includes(sc.location) ? sc.location : 'living_room', time: sc.time === 'night' ? 'night' as const : 'day' as const, card: clean(sc.card, 50),
    cast: (Array.isArray(sc.cast) ? sc.cast : []).map(String).filter((id: string) => chars.some((c) => c.id === id)).slice(0, 3),
    lines: (Array.isArray(sc.lines) ? sc.lines : []).map((l: any, li: number) => ({ speaker: l.speaker === 'narrator' ? 'narrator' : String(l.speaker), ...parseTags(clean(l.text, 400)), key: `f${si}_${li}` })).filter((l: any) => l.text),
  }));
  // Whoever speaks in a scene is in it.
  for (const sc of scenesIn) for (const l of sc.lines) if (l.speaker !== 'narrator' && !sc.cast.includes(l.speaker) && sc.cast.length < 3) sc.cast.push(l.speaker);
  const all = scenesIn.flatMap((sc) => sc.lines);
  const clipList = await pool(all, 4, (l: any) => speak(kit, l.text, voice[l.speaker] || NARRATOR, l.speaker === 'narrator' ? '-3%' : '+0%', l.speaker === 'narrator' ? 'male' : (chars.find((c) => c.id === l.speaker)?.gender || 'female'), l.key));
  const clipOf = new Map(all.map((l: any, i: number) => [l.key, clipList[i]]));
  const placed: Placed[] = [];
  const scenes: any[] = [];
  let t = 0;
  for (const sc of scenesIn) {
    const start = t;
    t += 2.3; // establishing shot while the characters walk in
    const lines: any[] = [];
    sc.lines.forEach((l: any, i: number) => {
      const c = clipOf.get(l.key)!;
      placed.push({ at: t, clip: c, speaker: l.speaker, tags: l.tags });
      lines.push({ speaker: l.speaker, start: t, end: t + c.duration });
      const next = sc.lines[i + 1];
      t += c.duration + (next && next.speaker !== l.speaker ? 0.22 : 0.4);
    });
    t += 1.7; // room to walk out
    scenes.push({ start, end: t, location: sc.location, time: sc.time, card: sc.card, cast: sc.cast.length ? sc.cast : [chars[0].id], lines });
  }
  const duration = +(t + 1.2).toFixed(2);
  scenes[scenes.length - 1].end = duration;
  await kit.reportStatus('running', '3/5 Mixing the sound', 40, `Recorded ${all.length} lines (${duration.toFixed(0)}s).`);
  const mood = G ? G.mood : ['story', 'emotional', 'mystery', 'horror', 'upbeat', 'comedy', 'action'].includes(s.mood) ? s.mood : 'story';
  const mix = await mixAudio(kit, placed, [], duration, mood, 14);
  const job = {
    mode: 'film', width: kit.W, height: kit.H, fps: kit.FPS, duration, title: clean(s.title, 80), badge: injected ? '' : 'SHORT FILM', endCard: injected ? '' : 'Follow for the next film', accent: '#facc15',
    fontUrl: '/font/Poppins-Bold.ttf', audio: '/audio/dialogue.wav', words: mix.words,
    characters: chars.map((c) => ({ ...c, cues: mix.cues[c.id] || [] })), scenes, grade: G?.grade || (s.mood === 'horror' ? 'horror' : ''),
  };
  const res = await render(kit, job, mix, duration);
  return {
    // Big reactions, but not while the camera moves between places.
    highlights: emotionMoments(mix.cues, 0.7).filter((h) => !scenes.some((sc, i) => i > 0 && Math.abs(h.t - sc.start) < 0.8)),
    title: clean(s.title, 95), description: `${clean(s.logline, 300)}\n\n${clean(s.description, 800)}\n\nCast: ${chars.map((c) => c.name).join(', ')}. An original 2D animated short film.`,
    hashtags: [...cleanTags(s.hashtags), 'animation', 'shortfilm'], tags: ['2d animation', 'short film', 'animated story', ...(genre ? [`${genre} short film`] : []), ...chars.map((c) => c.name)],
    script: scenesIn.map((sc) => sc.lines.map((l: any) => `${l.speaker === 'narrator' ? 'Narrator' : chars.find((c) => c.id === l.speaker)?.name}: ${l.text}`).join('\n')).join('\n\n'),
    durationSec: duration, character: res.character || 'film', model: got?.model || 'template', sources: [],
  };
}

// ===========================================================================
// STICKMAN — "VS" fight edits: two sides clash over rounds (speed vs strength,
// a king vs an army, an engineer vs AI coding agents…), one side wins.
// ===========================================================================
/** 3D fight arenas (anim/arenas.ts): one per video, used for every stage of it. */
export const FIGHT_LOCATIONS = ARENA_IDS;

/** Built-in match-ups (offline, and when the writer is unavailable in dry runs). */
const MATCHUPS: { sides: VsSide[]; title: string; hook: string; verdict: string; winners: ('A' | 'B')[] }[] = [
  { title: 'SPEED vs STRENGTH | Who Really Wins?', hook: 'Speed versus strength. Who really wins?', verdict: 'Speed wins. Strength never landed the big one.', winners: ['B', 'A', 'A'],
    sides: [{ id: 'A', name: 'Speed', label: 'SPEED', archetype: 'speed', color: '#2f45b8', count: 1 }, { id: 'B', name: 'Strength', label: 'STRENGTH', archetype: 'strength', color: '#b3150f', count: 1 }] },
  { title: 'ONE KING vs AN ENTIRE ARMY', hook: 'One king against an entire army.', verdict: 'The king stands alone.', winners: ['A', 'B', 'A'],
    sides: [{ id: 'A', name: 'The King', label: 'KING', archetype: 'sword', color: '#6d28d9', count: 1, gear: 'crown' }, { id: 'B', name: 'Soldier', label: 'ARMY', archetype: 'brawler', color: '#374151', count: 6, gear: 'helmet' }] },
  { title: 'ENGINEER vs AI CODING AGENTS', hook: 'One engineer versus four AI coding agents.', verdict: 'The engineer still ships it.', winners: ['B', 'A', 'A'],
    sides: [{ id: 'A', name: 'Engineer', label: 'ENGINEER', archetype: 'tech', color: '#0f766e', count: 1, gear: 'hardhat' }, { id: 'B', name: 'AI Agent', label: 'AI AGENTS', archetype: 'tech', color: '#c2410c', count: 4, gear: 'antenna' }] },
  { title: 'NINJA vs SAMURAI | Who Wins?', hook: 'Ninja versus samurai. Only one walks away.', verdict: 'The samurai wins by one blade.', winners: ['A', 'B', 'B'],
    sides: [{ id: 'A', name: 'Ninja', label: 'NINJA', archetype: 'ninja', color: '#111827', count: 1, gear: 'hood' }, { id: 'B', name: 'Samurai', label: 'SAMURAI', archetype: 'sword', color: '#b91c1c', count: 1, gear: 'helmet' }] },
  { title: 'WIZARD vs ROBOT', hook: 'Magic versus machine.', verdict: 'Magic wins this time.', winners: ['B', 'A', 'A'],
    sides: [{ id: 'A', name: 'Wizard', label: 'MAGIC', archetype: 'magic', color: '#7c3aed', count: 1, gear: 'hood' }, { id: 'B', name: 'Robot', label: 'MACHINE', archetype: 'tech', color: '#475569', count: 1, gear: 'antenna' }] },
];

/** A colour that reads on a white stage (dark enough, saturated). */
function inkOn(hex: string, fallback: string): string {
  if (!/^#[0-9a-f]{6}$/i.test(hex || '')) return fallback;
  const n = parseInt(hex.slice(1), 16), r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  if (lum < 0.6) return hex;
  const k = 0.55 / lum;
  return `#${[r, g, b].map((v) => Math.round(v * k).toString(16).padStart(2, '0')).join('')}`;
}

async function stickman(kit: Kit): Promise<AnimResult> {
  const CFG = kit.CFG;
  const shorts = lengthOf(kit);
  const idea = clean(ideaOf(CFG), 300);
  const nRounds = shorts ? 3 : 5;
  const gears = ['none', 'headband', 'crown', 'cape', 'visor', 'glasses', 'hardhat', 'helmet', 'hood', 'antenna'];
  const user = `Plan a viral stickman "VS" fight edit${idea ? ` about: "${idea}"` : ''}.
The video is a fight simulation between two sides to answer "which one is better?" — like "Speed vs Strength", "One King vs an Army", "Engineer vs AI Coding Agents", "Ninja vs Samurai", "Boxer vs Karate Master", "Introvert vs Extrovert", "Coffee vs Energy Drink", "Cat Reflexes vs Dog Loyalty". Pick a match-up people argue about and would comment on. It can be a concept (speed, strength), a job, an idea, or one fighter against a crowd.
ALREADY MADE (pick a different match-up):
${avoidTitles(kit)}

- sides: exactly 2, ids "A" and "B". Each: name (short, e.g. "Speed", "AI Agent"), label (1-2 WORDS IN CAPITALS for the health bar, e.g. "SPEED", "AI AGENTS"), archetype (how it fights: ${ARCHETYPES.join(', ')}), color (strong hex colour that reads on white — e.g. red #b3150f, blue #2f45b8), count (1, or 3-8 for a crowd/army/swarm), gear (${gears.join(', ')}).
- rounds: exactly ${nRounds}. Each has "winner" ("A" or "B") and optional "lines": at most 2 very short trash-talk lines (max 6 words) like {"side": "B", "when": "start", "text": "Too slow."} or {"side": "A", "when": "end", "text": "Blink and you miss it."}.
- The overall winner wins most rounds and the FINAL round. Make it close (a comeback is best).
- hook: one spoken line to open the video (max 10 words), a question or a bold claim.
- verdict: one short spoken line for the end (max 10 words).
Return ONLY JSON: {"title": "e.g. SPEED vs STRENGTH | Who Really Wins? (max 70 chars)", "hook": "", "verdict": "", "description": "2 sentences for YouTube, ends with a question for the comments", "hashtags": ["5-8 hashtags without #"], "sides": [{"id": "A", "name": "", "label": "", "archetype": "speed", "color": "#2f45b8", "count": 1, "gear": "none"}], "rounds": [{"winner": "A", "lines": []}]}`;
  const got = await askJson(kit, 'You design viral stick-figure "VS" fight videos. You answer with one JSON object.', user, (j) => {
    if (!Array.isArray(j.sides) || j.sides.length !== 2) return 'need two sides';
    if (!Array.isArray(j.rounds) || j.rounds.length < Math.max(1, nRounds - 1)) return 'too few rounds';
    if (j.rounds.some((r: any) => r.winner !== 'A' && r.winner !== 'B')) return 'bad round winner';
    return null;
  }, 'stickman_script');
  if (!got && !CFG.allowFallbackPublish && !CFG.offline && !CFG.dryRun) throw new kit.PipelineError('script_retry', 'The AI model on the runner could not write the fight plan this time. Nothing was posted; the next attempt runs automatically.');
  const pick = MATCHUPS[Math.abs(Math.floor(Number(CFG.partNumber) || 0)) % MATCHUPS.length];
  const j = got?.j || { title: pick.title, hook: pick.hook, verdict: pick.verdict, description: `${pick.sides[0].label} vs ${pick.sides[1].label} — who really wins? Tell us in the comments.`, hashtags: ['stickman', 'whowins', 'vs'], sides: pick.sides, rounds: pick.winners.map((w) => ({ winner: w })) };
  const defaults = [{ color: '#2f45b8' }, { color: '#b3150f' }];
  let sides: VsSide[] = (j.sides as any[]).slice(0, 2).map((sd, i) => ({
    id: i ? 'B' : 'A', name: clean(sd.name, 20) || (i ? 'Strength' : 'Speed'), label: clean(sd.label || sd.name, 18).toUpperCase() || (i ? 'B' : 'A'),
    archetype: ARCHETYPES.includes(sd.archetype) ? sd.archetype : (i ? 'strength' : 'speed'),
    color: inkOn(String(sd.color || ''), defaults[i].color), count: clamp(Math.round(Number(sd.count) || 1), 1, shorts ? 8 : 10),
    gear: gears.includes(sd.gear) ? sd.gear : 'none',
  }));
  // Two sides in the same colour can't be told apart.
  if (String(sides[0].color || '').toLowerCase() === String(sides[1].color || '').toLowerCase()) { sides[0].color = defaults[0].color; sides[1].color = defaults[1].color; }
  // Guns, blades and costumes for this video (seeded: every video is equipped differently).
  sides = equipSides(sides, `${CFG.campaignId}:${CFG.partNumber}:${clean(j.title, 90)}`);
  // Rounds: best of N, the overall winner takes the last one.
  let rounds: VsRound[] = (j.rounds as any[]).slice(0, nRounds).map((r) => ({
    winner: r.winner === 'B' ? 'B' : 'A',
    lines: (Array.isArray(r.lines) ? r.lines : []).slice(0, 2).map((l: any) => ({ side: l.side === 'B' ? 'B' as const : 'A' as const, when: l.when === 'end' ? 'end' as const : 'start' as const, text: clean(l.text, 40) })).filter((l: any) => l.text),
  }));
  while (rounds.length < nRounds) rounds.push({ winner: rounds[rounds.length - 1]?.winner || 'A' });
  const wins = { A: rounds.filter((r) => r.winner === 'A').length, B: rounds.filter((r) => r.winner === 'B').length };
  const champ: 'A' | 'B' = wins.A >= wins.B ? 'A' : 'B';
  rounds[rounds.length - 1].winner = champ;
  const champSide = sides.find((x) => x.id === champ)!;
  const fighters = vsFighters(sides);
  const title = clean(j.title, 90) || `${sides[0].label} vs ${sides[1].label}`;
  await kit.reportStatus('running', '2/5 Choreographing the fights', 22, `Plan ready: "${title}" — ${sides.map((x) => `${x.label}${x.count > 1 ? ` ×${x.count}` : ''} (${x.archetype})`).join(' vs ')}, ${nRounds} rounds, ${champSide.label} wins${got ? ` (${got.model})` : ''}.`);

  // Voices: a narrator for the hook and verdict, one voice per side for trash talk.
  const vf: Record<string, string> = { narrator: NARRATOR };
  const used = new Set<string>([NARRATOR]);
  for (const f of fighters) { const key = f.side; if (!vf[key]) { const v = VOICE_POOL.male.find((x) => !used.has(x)) || VOICE_POOL.male[0]; used.add(v); vf[key] = v; } vf[f.id] = vf[key]; }
  const pieces: { key: string; text: string; who: string }[] = [];
  const hook = clean(j.hook, 90), verdict = clean(j.verdict, 90);
  if (hook) pieces.push({ key: 'hook', text: hook, who: 'narrator' });
  if (verdict) pieces.push({ key: 'verdict', text: verdict, who: 'narrator' });
  rounds.forEach((r, ri) => (r.lines || []).forEach((l, li) => pieces.push({ key: `r${ri}_${li}`, text: l.text, who: l.side })));
  const recorded = await pool(pieces, 4, (p) => speak(kit, p.text, vf[p.who] || NARRATOR, p.who === 'narrator' ? '+4%' : '+8%', 'male', p.key));
  const clipOf = new Map(pieces.map((p, i) => [p.key, recorded[i]]));
  const lineClip = new Map<string, Clip>();
  rounds.forEach((r, ri) => (r.lines || []).forEach((l, li) => { const c = clipOf.get(`r${ri}_${li}`); if (c) lineClip.set(l.text, c); }));

  const placed: Placed[] = [];
  const sfx: { t: number; kind: string }[] = [];
  const hookClip = clipOf.get('hook');
  const introEnd = +Math.max(2.8, hookClip ? hookClip.duration + 0.9 : 0).toFixed(2);
  if (hookClip) placed.push({ at: 0.35, clip: hookClip, speaker: 'narrator', tags: [] });
  sfx.push({ t: 0.25, kind: 'boom' });
  const fightLen = shorts ? 12 : 20;
  const fights: any[] = [];
  // ONE arena for the whole video (every stage, the intro and the outro); the next video gets another.
  const arenaId = arenaOrder(`${CFG.campaignId}:${CFG.partNumber}:${title}`, 1)[0];
  let t = introEnd;
  rounds.forEach((r, ri) => {
    const start = t;
    const cardEnd = start + 1.15;
    sfx.push({ t: start + 0.08, kind: 'bell' });
    const beats = planRound(sides, r, ri, rounds.length, fightLen * 0.72, `${CFG.campaignId}:${CFG.partNumber}:${title}`);
    const fs0 = fighters.map((f) => ({ ...f }));
    const ch = choreograph(fs0, beats, { start: cardEnd + 0.25, lineDur: (text) => (lineClip.get(text)?.duration || 1) });
    ch.lines.forEach((l) => { const c = lineClip.get(l.text); if (c) { placed.push({ at: l.t + 0.05, clip: c, speaker: l.actor, tags: [] }); l.end = l.t + c.duration + 0.2; } });
    sfx.push(...ch.sfx);
    // Bullet hits and ricochets, brass falling after every shot.
    for (const p of ch.projectiles || []) if (p.kind === 'bullet') { sfx.push({ t: p.t1, kind: p.miss ? 'ricochet' : 'bulletHit' }); sfx.push({ t: p.t0 + 0.42, kind: 'shell' }); }
    for (const im of ch.impacts) {
      if (im.kind === 'down' || im.kind === 'launchFar') sfx.push({ t: im.t + (im.kind === 'launchFar' ? 0.7 : 0.4), kind: 'thud' });
      if (im.kind === 'land') sfx.push({ t: im.t, kind: im.strength >= 1.5 ? 'heavy' : 'thud' });
      if (im.kind === 'parry') sfx.push({ t: im.t, kind: 'clang' });
      if (im.ground) sfx.push({ t: im.t + 0.05, kind: 'crumble' });
    }
    const end = cardEnd + 0.25 + ch.duration;
    fights.push({ start, cardEnd, end, location: arenaId, time: 'day', card: '', round: ri + 1, final: ri === rounds.length - 1, winner: r.winner, choreo: ch, health: roundHealth(ch, fighters, r.winner) });
    t = end;
  });
  const outroStart = t;
  const verdictClip = clipOf.get('verdict');
  if (verdictClip) placed.push({ at: outroStart + 0.45, clip: verdictClip, speaker: 'narrator', tags: [] });
  sfx.push({ t: outroStart + 0.05, kind: 'boom' });
  const duration = +(outroStart + Math.max(3.4, verdictClip ? verdictClip.duration + 1.2 : 0)).toFixed(2);
  fights[fights.length - 1].end = outroStart;
  await kit.reportStatus('running', '3/5 Sound effects and music', 40, `${rounds.length} rounds choreographed (${duration.toFixed(0)}s, ${sfx.length} sound effects).`);
  const mix = await mixAudio(kit, placed, sfx, duration, 'action', 8);
  const job = {
    mode: 'stickman', width: kit.W, height: kit.H, fps: kit.FPS, duration, title, badge: '', endCard: '', accent: champSide.color, hideTitle: true,
    fontUrl: '/font/Poppins-Bold.ttf', words: mix.words, fighters, fights,
    vs: { sides, champion: champ, introEnd, outroStart, verdict },
  };
  const res = await render(kit, job, mix, duration);
  const score = `${champSide.label} wins ${Math.max(wins.A, wins.B)}-${Math.min(wins.A, wins.B)}`;
  return {
    // The VS face-off card and the big hits.
    highlights: [{ t: Math.max(0.6, introEnd - 0.9), w: 0.5 }, ...fights.flatMap((f: any) => (f.choreo?.impacts || []).filter((im: any) => im.ko || im.ground || ['down', 'launchFar', 'clash', 'spiked', 'thrown'].includes(im.kind) || (!['block', 'parry', 'land'].includes(im.kind) && im.strength >= 0.9)).map((im: any) => ({ t: im.t - 0.1, w: im.ko ? 0.8 : im.kind === 'clash' ? 0.7 : 0.5 })))],
    title,
    description: `${clean(j.description, 700)}\n\n${rounds.map((r, i) => `Round ${i + 1}: ${sides.find((x) => x.id === r.winner)!.label}`).join('\n')}\n🏆 ${score}.\n\nWho should fight next? Tell us in the comments.\nAn original stickman animation, just for fun — not a real-world test.`,
    hashtags: [...cleanTags(j.hashtags), 'stickman', 'stickfight', 'whowins'], tags: ['stickman', 'stick fight', 'vs', 'who wins', 'animation', ...sides.map((x) => String(x.label || '').toLowerCase())],
    script: [hook, ...rounds.map((r, i) => `Round ${i + 1}: ${sides.find((x) => x.id === r.winner)!.label}${(r.lines || []).map((l) => ` — ${sides.find((x) => x.id === l.side)!.label}: "${l.text}"`).join('')}`), verdict].filter(Boolean).join('\n'),
    durationSec: duration, character: res.character || 'stickman', model: got?.model || 'template', sources: [],
  };
}

// ---------------------------------------------------------------------------
async function render(kit: Kit, job: any, mix: { dialogue: string; final: string }, duration: number) {
  await kit.reportStatus('running', '4/5 Animating', 55, `Animating ${duration.toFixed(0)}s at ${kit.W}x${kit.H}…`);
  const res = await kit.runStage({ stageFile: 'anim.js', job, audioFinal: mix.final, files: { '/audio/dialogue.wav': mix.dialogue }, duration });
  if (!res.ok) throw new kit.PipelineError('render_failed', `The animation could not be rendered: ${res.reason}`);
  return res;
}

async function pool<T, R>(items: T[], n: number, fn: (x: T, i: number) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); } }));
  return out;
}

export const ANIM_CATEGORIES = new Set(['animation', 'podcast']);

export async function runAnimated(kit: Kit): Promise<AnimResult> {
  const cat = String(kit.CFG.category || '').toLowerCase();
  if (cat === 'podcast') return podcast(kit);
  return /stick/i.test(String(kit.CFG.animStyle || kit.CFG.subGenre || '')) ? stickman(kit) : film(kit);
}
