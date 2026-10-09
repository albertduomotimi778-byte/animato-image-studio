/**
 * Animato AutoPoster — cloud renderer (runs on GitHub Actions).
 *
 * One run = one episode:
 *   1. read the job from the repository_dispatch payload
 *   2. ask the app whether the automation still exists / isn't paused
 *   3. write the episode as timed scenes with free Google Gemini models
 *      (10 rotating keys, instant failover) and Groq as the fallback —
 *      grounded in fresh, never-repeated headlines for tech/news
 *   4. narrate it with a neural voice, keeping word-level timings
 *   5. find an image for every scene (AI images for stories/cooking, real
 *      photos first for tech/news)
 *   6. render in headless Chrome with the app's own character engine:
 *      real lip-sync, blinking, glances, head motion, looking at each image,
 *      word-by-word captions — then encode with FFmpeg
 *   7. publish to YouTube as a Short or a regular video
 *   8. report the episode back to the app, which schedules the next one
 *
 * No npm dependencies: Node 22 built-ins, FFmpeg, Chrome, Python edge-tts.
 * The presenter is the CSS character designed in the app (only its small
 * JSON spec travels in the job; the stage draws it).
 * Run with:  node --experimental-strip-types animato-cloud/renderer.mts
 */

import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { LlmPool, extractJsonObject } from './llm.ts';
import { researchNews, bingNews, researchRecipe, factSheet, unsupportedNumbers, visionMatches, visionAnatomy, metadataMatches, identifierTokens, type FactPack, type ResearchCtx } from './research.ts';
import { composeBuffers, eqForVoice, automateLevel, levelDb, encodeWav, moodFor } from './music.ts';
import { runAnimated, ANIM_CATEGORIES } from './anim/runner.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ENV = process.env;
const IN_ACTIONS = ENV.GITHUB_ACTIONS === 'true';

function readEvent(): any {
  try {
    if (ENV.GITHUB_EVENT_PATH && fs.existsSync(ENV.GITHUB_EVENT_PATH)) {
      return JSON.parse(fs.readFileSync(ENV.GITHUB_EVENT_PATH, 'utf8'));
    }
  } catch (err: any) {
    console.warn('Could not read the GitHub event payload:', err?.message);
  }
  return {};
}

const EVENT = readEvent();
const CP = EVENT.client_payload || {};
const INPUTS = EVENT.inputs || {};
const JOB = CP.job || {};
const AUTH = CP.auth || {};

function pick(...values: any[]): string {
  for (const v of values) {
    if (v === undefined || v === null) continue;
    const s = String(v).trim();
    if (s) return s;
  }
  return '';
}

const DEFAULT_YT_CLIENT_ID = '592242596648-am9pri2j11vmu44fdc6oau5p34aklkj8.apps.googleusercontent.com';
// API keys are never stored in the render repository: the app sends them in
// every dispatch (auth.*), or they come from repository secrets.

const DIMENSIONS: Record<string, [number, number]> = { '9:16': [1080, 1920], '16:9': [1920, 1080], '1:1': [1080, 1080], '4:3': [1440, 1080] };

function resolveFormat(): { format: 'shorts' | 'video'; aspect: string } {
  const rawAspect = pick(JOB.aspect_ratio, INPUTS.aspect_ratio, ENV.ASPECT_RATIO, '9:16').replace('/', ':');
  let aspect = DIMENSIONS[rawAspect] ? rawAspect : '9:16';
  const rawFormat = pick(JOB.format, INPUTS.format, ENV.VIDEO_FORMAT).toLowerCase();
  const format: 'shorts' | 'video' = rawFormat === 'video' ? 'video' : rawFormat === 'shorts' ? 'shorts' : (aspect === '16:9' || aspect === '4:3' ? 'video' : 'shorts');
  if (format === 'shorts' && (aspect === '16:9' || aspect === '4:3')) aspect = '9:16';
  if (format === 'video' && (aspect === '9:16' || aspect === '1:1')) aspect = '16:9';
  return { format, aspect };
}

const rawGender = pick(JOB.gender, INPUTS.gender, ENV.CHARACTER_GENDER, 'female').toLowerCase();
function keyList(...values: any[]): string[] {
  return Array.from(new Set(values
    .flatMap((v) => (Array.isArray(v) ? v : String(v || '').split(/[\s,;]+/)))
    .map((k) => String(k).trim())
    .filter((k) => k.length > 8)));
}
const listOf = (v: string): string[] | undefined => { const l = String(v || '').split(/[\s,;]+/).map((x) => x.trim()).filter(Boolean); return l.length ? l : undefined; };
function parseSpec(v: any): any | null {
  if (!v) return null;
  if (typeof v === 'object') return v;
  try { const o = JSON.parse(String(v)); return o && typeof o === 'object' ? o : null; } catch { return null; }
}
const FMT = resolveFormat();

const CFG = {
  campaignId: pick(CP.campaign_id, INPUTS.campaign_id, ENV.CAMPAIGN_ID),
  partNumber: Math.max(1, parseInt(pick(CP.part_number, INPUTS.part_number, ENV.PART_NUMBER, '1'), 10) || 1),
  appUrl: pick(CP.app_url, INPUTS.app_url, ENV.APP_URL).replace(/\/+$/, ''),
  category: pick(JOB.category, INPUTS.category, ENV.CATEGORY, 'stories').toLowerCase(),
  subGenre: pick(JOB.sub_genre, INPUTS.sub_genre, ENV.SUB_GENRE),
  topic: pick(JOB.topic, INPUTS.topic, ENV.PROMPT, ENV.TOPIC),
  campaignName: pick(JOB.name, ENV.CAMPAIGN_NAME),
  gender: (rawGender === 'male' ? 'male' : 'female') as 'male' | 'female',
  format: FMT.format,
  aspect: FMT.aspect,
  autoPost: pick(JOB.auto_post_youtube, INPUTS.auto_post_youtube, ENV.AUTO_POST_YOUTUBE, 'false').toLowerCase() === 'true',
  // Publishing destinations selected by the automation. Credentials are fetched
  // from the app server with the runner key; they are never embedded in the job payload.
  // Only YouTube and Instagram are published to (Facebook and Threads were retired).
  targets: new Set(String(pick(JOB.publish_targets, '')).split(',').map((x) => x.trim()).filter((x) => ['youtube', 'instagram'].includes(x))),
  previousScript: pick(JOB.previous_script, ENV.PREVIOUS_SCRIPT),
  privacy: pick(JOB.privacy, ENV.YOUTUBE_PRIVACY, 'public'),
  ytRefreshToken: pick(AUTH.youtube_refresh_token, ENV.YOUTUBE_REFRESH_TOKEN),
  // Expected destination supplied by the campaign. Used as a final routing
  // guard immediately before upload.
  ytChannelId: pick(AUTH.youtube_channel_id, ENV.YOUTUBE_CHANNEL_ID),
  ytClientId: pick(AUTH.youtube_client_id, ENV.YOUTUBE_CLIENT_ID, DEFAULT_YT_CLIENT_ID),
  ytClientSecret: pick(AUTH.youtube_client_secret, ENV.YOUTUBE_CLIENT_SECRET),
  // The script writer is the self-hosted model started by the workflow on this runner (no API keys).
  localLlmUrl: pick(ENV.LOCAL_LLM_URL, 'http://127.0.0.1:8080/v1'),
  localVlmUrl: pick(ENV.LOCAL_VLM_URL),
  geminiKeys: [] as string[],
  groqKeys: [] as string[],
  airforceKey: '',
  // Musical: the concert stage (a fixed stage number, or 0 = a new stage every video) and the song language.
  stageId: Math.max(0, parseInt(pick(JOB.stage_id, ENV.STAGE_ID, '0'), 10) || 0),
  musicLanguage: pick(JOB.music_language, ENV.MUSIC_LANGUAGE, 'English'),
  // Music engine: ACE-Step 1.5 official cloud (free API key from https://acemusic.ai/api-key).
  // Personal project: built-in ACE-Step keys (rotated); a repository secret still overrides them.
  // Set to the local ACE-Step server on the runner while a song is being made.
  acestepBase: 'http://127.0.0.1:8011',
  acestepModel: pick(ENV.ACESTEP_MODEL, ''),
  // Musical song length (seconds), at most 60: intro → verse → pre-chorus → chorus → chorus again → outro.
  musicSeconds: Math.max(20, Math.min(60, parseInt(pick(ENV.MUSIC_SECONDS, '60'), 10) || 60)),
  // Optional paid last resort: Google Lyria via the Gemini API (needs a billing-enabled Gemini key).
  lyriaModel: pick(ENV.LYRIA_MODEL, 'lyria-3.5'),
  lyriaEnabled: pick(ENV.LYRIA_ENABLED, 'false').toLowerCase() === 'true',
  characterSpec: (() => { const s = parseSpec(pick(JOB.character_spec, ENV.CHARACTER_SPEC)); return s?.kind === 'ai-rig' ? { ...s, assetBaseUrl: pick(s.assetBaseUrl, pick(CP.app_url, INPUTS.app_url, ENV.APP_URL)).replace(/\/+$/, '') } : s; })(),
  // Podcasts: the hosts designed in the app, and the studio.
  castSpecs: (() => { try { const v = JSON.parse(pick(JOB.cast_specs, ENV.CAST_SPECS) || '[]'); const base = pick(CP.app_url, INPUTS.app_url, ENV.APP_URL).replace(/\/+$/, ''); return Array.isArray(v) ? v.slice(0, 3).map((s: any) => s?.kind === 'ai-rig' ? { ...s, assetBaseUrl: s.assetBaseUrl || base } : s) : []; } catch { return []; } })(),
  studio: (() => { try { return JSON.parse(pick(JOB.studio, ENV.STUDIO) || 'null'); } catch { return null; } })(),
  storyGenre: pick(JOB.story_genre, ENV.STORY_GENRE),
  animStyle: pick(JOB.anim_style, ENV.ANIM_STYLE),
  podcastAbout: pick(JOB.podcast_about, ENV.PODCAST_ABOUT),
  podcastGuests: pick(JOB.podcast_guests, ENV.PODCAST_GUESTS) === 'true',
  /** Musicals: the singer is the guest of her own show, interviewed about her latest songs. */
  podcastSinger: pick(JOB.podcast_singer, ENV.PODCAST_SINGER) === 'true',
  /** The language the podcast is spoken in (e.g. a Korean musical's podcast stays Korean, with English subtitles). */
  podcastLanguage: pick(JOB.podcast_language, ENV.PODCAST_LANGUAGE),
  podcastSongs: (() => { try { const v = JSON.parse(pick(JOB.podcast_songs, ENV.PODCAST_SONGS) || '[]'); return Array.isArray(v) ? v.slice(0, 6) : []; } catch { return []; } })(),
  /** The presenter's name: the YouTube channel's first word ("Lola Cooks" → Lola). */
  presenterName: pick(JOB.presenter_name, ENV.PRESENTER_NAME),
  /** What is working on this channel (real YouTube numbers) — the writer leans into it. */
  performanceNotes: [pick(JOB.performance_notes, ENV.PERFORMANCE_NOTES), pick(JOB.creative_angle) ? `THIS VIDEO'S CREATIVE ANGLE (make it feel fresh — never a generic take): ${pick(JOB.creative_angle)}.` : ''].filter(Boolean).join('\n'),
  /** Titles other creators made recently in this category (across the app) — never repeat or closely copy them. */
  globalTitles: String(pick(JOB.global_titles) || '').split('\n').map((x) => x.trim()).filter(Boolean).slice(-50),
  /** Podcasts: 'auto' = brand-new hosts every episode (podcastCastCount of them, besides the presenter in guest mode). */
  podcastCast: pick(JOB.podcast_cast, ENV.PODCAST_CAST, 'custom'),
  podcastCastCount: Math.max(1, Math.min(3, parseInt(pick(JOB.podcast_cast_count, '2'), 10) || 2)),
  hostCategory: pick(JOB.host_category, ENV.HOST_CATEGORY),
  // Story arcs: every story is told in at most 3 parts and then finished for good.
  arcParts: Math.max(1, parseInt(pick(JOB.arc_parts, '3'), 10) || 3),
  storyPremise: pick(JOB.story_premise),
  storyCharacters: pick(JOB.story_characters),
  storyTitle: pick(JOB.story_title),
  adBrief: pick(JOB.ad_brief),
  adImages: String(pick(JOB.ad_images) || '').split(',').map((x) => x.trim()).filter(Boolean),
  /** Ads: the advertiser's name, whether it is a service (no app/website — people message them) and how to reach them. */
  adProfile: (() => { try { const j = JSON.parse(pick(JOB.ad_profile) || '{}'); return { company: String(j.company || '').slice(0, 80), service: j.service === true, contact: String(j.contact || '').slice(0, 160) }; } catch { return { company: '', service: false, contact: '' }; } })(),
  /** Ads promo for the current product: only mentioned when switched on in the app. */
  adPromo: (() => { try { const j = JSON.parse(pick(JOB.ad_promo) || '{}'); return j && j.enabled === true ? { enabled: true, discount: String(j.discount || '').slice(0, 60), details: String(j.details || '').slice(0, 400), code: String(j.code || '').slice(0, 40), ends: String(j.ends || '').slice(0, 60) } : null; } catch { return null; } })(),
  usedHeadlines: String(pick(JOB.used_headlines)).split('\n').map((x) => x.trim()).filter(Boolean),
  pexelsKey: pick(AUTH.pexels_key, ENV.PEXELS_API_KEY),
  pixabayKey: pick(AUTH.pixabay_key, ENV.PIXABAY_API_KEY),
  runnerKey: pick(AUTH.runner_key, ENV.ANIMATO_RUNNER_KEY),
  runId: pick(ENV.GITHUB_RUN_ID),
  runUrl: ENV.GITHUB_RUN_ID
    ? `${ENV.GITHUB_SERVER_URL || 'https://github.com'}/${ENV.GITHUB_REPOSITORY}/actions/runs/${ENV.GITHUB_RUN_ID}`
    : '',
  // Test hooks (never set in production).
  dryRun: ENV.ANIMATO_DRY_RUN === 'true',
  offline: ENV.ANIMATO_OFFLINE === 'true',
  geminiBase: pick(ENV.GEMINI_BASE_URL, 'https://generativelanguage.googleapis.com/v1beta'),
  groqBase: pick(ENV.GROQ_BASE_URL, 'https://api.groq.com/openai/v1'),
  googleTokenUrl: pick(ENV.GOOGLE_TOKEN_URL, 'https://oauth2.googleapis.com/token'),
  // Real-image sources (news, tech, tutorials, cooking, ads). Overridable for tests.
  wikiApiBase: pick(ENV.WIKI_API_BASE, 'https://en.wikipedia.org/w/api.php'),
  commonsApiBase: pick(ENV.COMMONS_API_BASE, 'https://commons.wikimedia.org/w/api.php'),
  openverseBase: pick(ENV.OPENVERSE_BASE, 'https://api.openverse.org/v1/images/'),
  wikidataApiBase: pick(ENV.WIKIDATA_API_BASE, 'https://www.wikidata.org/w/api.php'),
  youtubeUploadBase: pick(ENV.YOUTUBE_UPLOAD_BASE, 'https://www.googleapis.com/upload/youtube/v3'),
  newsBase: pick(ENV.NEWS_RSS_BASE, 'https://news.google.com/rss/search'),
  allowFallbackPublish: ENV.PUBLISH_WITH_FALLBACK_CONTENT === 'true',
  maxRenderSeconds: parseInt(pick(ENV.MAX_VIDEO_SECONDS, '0'), 10) || 0,
  // Longest time a run spends waiting for ACE-Step before giving up (the next run retries).
  // Max minutes for ACE-Step on the runner (setup on the first run + making the song on the CPU).
  acestepLocalMin: parseInt(pick(ENV.ACESTEP_LOCAL_MINUTES, '35'), 10) || 35
};

/**
 * "Create animation" (Studio → Create animation): ONE video made from the user's own script
 * and/or recorded voice, with the character they picked. It is handed back to the app (never
 * published) — see produceStudio().
 */
interface StudioJob {
  id: string; kind: string; title: string; script: string; voiceUrl: string; voiceGender: 'female' | 'male';
  cast: { name: string; gender: 'female' | 'male'; spec: any }[]; watermark: boolean; maxSeconds: number;
  location: string; time: 'day' | 'night'; scenes: { location?: string; time?: string; lines: { speaker: string; text: string }[] }[];
}
const STUDIO: StudioJob | null = (() => {
  const s = parseSpec(JOB.studio_job);
  if (!s?.id) return null;
  const gender = (g: any): 'female' | 'male' => (g === 'male' ? 'male' : 'female');
  return {
    id: String(s.id).replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40), kind: String(s.kind || 'talk'), title: String(s.title || '').slice(0, 90),
    script: String(s.script || '').slice(0, 6000), voiceUrl: String(s.voiceUrl || ''), voiceGender: gender(s.voiceGender),
    cast: (Array.isArray(s.cast) ? s.cast : []).slice(0, 3).map((c: any, i: number) => ({ name: String(c?.name || `Character ${i + 1}`).slice(0, 24), gender: gender(c?.gender), spec: c?.spec || null })),
    watermark: s.watermark !== false, maxSeconds: Math.max(15, Math.min(300, Number(s.maxSeconds) || 60)),
    location: String(s.location || 'living_room'), time: s.time === 'night' ? 'night' : 'day',
    scenes: Array.isArray(s.scenes) ? s.scenes.slice(0, 8) : [],
  };
})();

if (IN_ACTIONS) {
  for (const secret of [CFG.ytRefreshToken, CFG.ytClientSecret, ...CFG.geminiKeys, ...CFG.groqKeys, CFG.runnerKey, CFG.pexelsKey, CFG.pixabayKey, CFG.airforceKey]) {
    if (secret && secret.length > 6) console.log(`::add-mask::${secret}`);
  }
}

const OUTPUT_DIR = path.resolve(ENV.ANIMATO_OUTPUT_DIR || path.join(process.cwd(), 'output'));
const WORK_DIR = path.join(OUTPUT_DIR, 'work');
const OUTPUT_VIDEO = path.join(OUTPUT_DIR, 'rendered_video.mp4');
const OUTPUT_META = path.join(OUTPUT_DIR, 'video_metadata.json');
fs.mkdirSync(WORK_DIR, { recursive: true });

// Older jobs had no target list: YouTube when auto-post is on.
if (!CFG.targets.size && CFG.autoPost) CFG.targets.add('youtube');
const WANT = { youtube: CFG.targets.has('youtube'), facebook: CFG.targets.has('facebook'), instagram: CFG.targets.has('instagram'), threads: CFG.targets.has('threads') };
const PUBLISH = WANT.youtube || WANT.facebook || WANT.instagram || WANT.threads;
let SOCIAL: Record<string, any> = {};

const [W, H] = DIMENSIONS[CFG.aspect];
const FRAME_W = W, FRAME_H = H;
const FPS = 30;
/** Minutes left before GitHub stops the job (JOB_TIMEOUT_MINUTES, default 60, minus setup time already used). */
const PROCESS_T0 = Date.now();
const jobMinutesLeft = () => (parseInt(process.env.JOB_TIMEOUT_MINUTES || '60', 10) || 60) - 3 - (Date.now() - PROCESS_T0) / 60000;
const IS_SHORTS = CFG.format === 'shorts';

class PipelineError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

function log(msg: string) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`);
}

// ---------------------------------------------------------------------------
// Talking to the app
// ---------------------------------------------------------------------------
async function appRequest(method: string, route: string, body?: any, timeoutMs = 20000): Promise<{ status: number; data: any } | null> {
  if (!CFG.appUrl || (!CFG.campaignId && !STUDIO)) return null;
  // The app may be briefly busy (Cloudflare per-request CPU limit, a deploy, a
  // network blip): retry with growing pauses. The episode report matters most
  // (it records the video and schedules the next one), so it tries longest.
  const tries = /\/episodes$/.test(route) ? 7 : /\/status$/.test(route) ? 3 : 4;
  let last: { status: number; data: any } | null = null;
  for (let a = 0; a < tries; a++) {
    if (a) await new Promise((z) => setTimeout(z, Math.min(30000, 2000 * 2 ** (a - 1))));
    try {
      const res = await fetch(`${CFG.appUrl}${route}`, {
        method,
        headers: { 'Content-Type': 'application/json', 'X-Animato-Runner-Key': CFG.runnerKey || '' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs)
      });
      const text = await res.text();
      let data: any = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text.slice(0, 300) }; }
      last = { status: res.status, data };
      if (res.status !== 429 && res.status < 500) return last;
      console.warn(`App request ${method} ${route}: HTTP ${res.status} (attempt ${a + 1}/${tries}).`);
    } catch (err: any) {
      console.warn(`App request ${method} ${route} failed: ${err?.message} (attempt ${a + 1}/${tries}).`);
    }
  }
  return last;
}

const campaignPath = () => `/api/automation/campaigns/${encodeURIComponent(CFG.campaignId)}`;

/**
 * Keeps a copy of the finished video in the app's video library (the same
 * storage bucket as the AI video editor), so it can be watched, downloaded
 * and deleted from the app's "Videos" screen. Never fails the run.
 */
async function saveToLibrary(meta: { title: string; topic?: string; kind?: string; durationSec: number; width: number; height: number }): Promise<string> {
  if (!CFG.appUrl || !CFG.campaignId || CFG.offline || !fs.existsSync(OUTPUT_VIDEO)) return '';
  try {
    const size = fs.statSync(OUTPUT_VIDEO).size;
    const r = await appRequest('POST', `${campaignPath()}/videos`, { ...meta, size, partNumber: CFG.partNumber, format: CFG.format }, 30000);
    if (!r || r.status >= 300 || !r.data?.video?.id) { log(`⚠️ Video library: not saved (${r ? `HTTP ${r.status} ${r.data?.error || ''}` : 'app unreachable'}).`); return ''; }
    const id = String(r.data.video.id), up = r.data.upload, chunk = Number(up.chunkSize) || 6 * 1024 * 1024;
    const n = Math.max(1, Math.ceil(size / chunk));
    const etags: string[] = new Array(n).fill('');
    const fd = fs.openSync(OUTPUT_VIDEO, 'r');
    let next = 0;
    const worker = async () => {
      while (next < n) {
        const i = next++;
        const len = Math.min(chunk, size - i * chunk);
        const buf = Buffer.alloc(len);
        fs.readSync(fd, buf, 0, len, i * chunk);
        for (let a = 0; a < 4 && !etags[i]; a++) {
          try {
            const res = await fetch(`${CFG.appUrl}${up.base}/${i}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream', 'X-Animato-Runner-Key': CFG.runnerKey || '' }, body: buf, signal: AbortSignal.timeout(120000) });
            const d: any = await res.json().catch(() => ({}));
            if (res.ok && d.etag) etags[i] = String(d.etag);
            else await new Promise((z) => setTimeout(z, 1500 * (a + 1)));
          } catch { await new Promise((z) => setTimeout(z, 1500 * (a + 1))); }
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(3, n) }, worker));
    fs.closeSync(fd);
    if (etags.some((e) => !e)) { log('⚠️ Video library: some pieces did not arrive — not saved.'); return ''; }
    const done = await appRequest('POST', `/api/videos/${id}/complete`, { parts: etags }, 60000);
    if (!done || done.status >= 300) { log(`⚠️ Video library: could not finish (${done ? `HTTP ${done.status}` : 'app unreachable'}).`); return ''; }
    log(`Saved to the video library (${(size / 1e6).toFixed(1)} MB).`);
    return id;
  } catch (e: any) { log(`⚠️ Video library: ${e?.message || e}`); return ''; }
}
const linkLibraryVideo = async (id: string, url: string) => { if (id && url) await appRequest('POST', `/api/videos/${id}/youtube`, { youtubeUrl: url }); };

async function loadSocialCredentials(): Promise<void> {
  if (!CFG.appUrl || !CFG.campaignId || CFG.offline) return;
  if (!(WANT.facebook || WANT.instagram || WANT.threads)) return;
  const r = await appRequest('GET', `${campaignPath()}/social-credentials`, undefined, 30000);
  if (!r || r.status >= 300 || !r.data?.social) throw new PipelineError('social_credentials', `The app could not provide the selected social publishing credentials (HTTP ${r?.status || 'no response'}).`);
  SOCIAL = r.data.social || {};
  for (const p of ['facebook', 'instagram', 'threads']) {
    if (!WANT[p as keyof typeof WANT]) continue;
    const v = SOCIAL[p];
    if (p === 'facebook' ? !(v?.pageId && v?.pageAccessToken) : !(v?.userId && v?.accessToken)) {
      throw new PipelineError(`${p}_not_connected`, `${p[0].toUpperCase()}${p.slice(1)} is selected but its publishing credential is unavailable. Reconnect it in the automation dashboard.`);
    }
  }
}

async function publicVideoUrl(libId: string): Promise<string> {
  if (!libId) throw new PipelineError('social_media_url', 'The rendered video was not saved to the app storage, so social platforms cannot fetch it.');
  // The completion response includes a short-lived public URL. Reuse the endpoint
  // so this remains compatible with older runners that only returned the video id.
  const r = await appRequest('GET', `/api/videos/${encodeURIComponent(libId)}/publish-url`, undefined, 30000);
  if (!r || r.status >= 300 || !r.data?.publishUrl) throw new PipelineError('social_media_url', `The app could not create a public video URL (HTTP ${r?.status || 'no response'}).`);
  return String(r.data.publishUrl);
}

function socialCaption(title: string, description: string): string {
  const clean = String(description || title || '').replace(/\s+/g, ' ').trim();
  return clean.slice(0, 2200);
}

/**
 * Facebook Page Reels (vertical Shorts). Needs only pages_manage_posts +
 * pages_read_engagement + pages_show_list — NOT publish_video, which is what
 * the classic /videos endpoint rejects with "(#100) No permission to publish
 * the video" until Meta approves that permission.
 */
/**
 * Threads rejects text over 500 characters — and it counts emoji / some
 * symbols as more than one. Measure in UTF-8 bytes (always >= Threads' count)
 * and cut on a word boundary, so the limit can never be exceeded.
 */
function threadsText(title: string, description: string, max = 480): string {
  const t = socialCaption(title, description);
  const enc = new TextEncoder();
  if (enc.encode(t).length <= max) return t;
  let cut = '';
  for (const ch of Array.from(t)) { if (enc.encode(cut + ch).length > max - 3) break; cut += ch; }
  const sp = cut.lastIndexOf(' ');
  return `${(sp > cut.length * 0.6 ? cut.slice(0, sp) : cut).trimEnd()}…`;
}

async function publishFacebookReel(meta: { title: string; description: string; videoUrl: string }): Promise<string> {
  const c = SOCIAL.facebook;
  const g = `https://graph.facebook.com/v24.0/${encodeURIComponent(c.pageId)}/video_reels`;
  const start = await fetch(g, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ upload_phase: 'start', access_token: c.pageAccessToken }), signal: AbortSignal.timeout(60000) });
  const sd: any = await start.json().catch(() => ({}));
  if (!start.ok || !sd.video_id) throw new PipelineError('facebook_publish', sd?.error?.message || `Facebook Reel start failed (HTTP ${start.status}).`);
  const up = await fetch(`https://rupload.facebook.com/video-upload/v24.0/${encodeURIComponent(sd.video_id)}`, { method: 'POST', headers: { Authorization: `OAuth ${c.pageAccessToken}`, file_url: meta.videoUrl }, signal: AbortSignal.timeout(300000) });
  const ud: any = await up.json().catch(() => ({}));
  if (!up.ok || ud?.success === false) throw new PipelineError('facebook_publish', ud?.debug_info?.message || ud?.error?.message || `Facebook Reel upload failed (HTTP ${up.status}).`);
  const fin = await fetch(g, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ upload_phase: 'finish', video_id: String(sd.video_id), video_state: 'PUBLISHED', title: meta.title.slice(0, 200), description: socialCaption(meta.title, meta.description), access_token: c.pageAccessToken }), signal: AbortSignal.timeout(120000) });
  const fd: any = await fin.json().catch(() => ({}));
  if (!fin.ok || fd?.success === false) throw new PipelineError('facebook_publish', fd?.error?.message || `Facebook Reel publish failed (HTTP ${fin.status}).`);
  return `https://www.facebook.com/reel/${sd.video_id}`;
}

async function publishFacebookVideo(meta: { title: string; description: string; videoUrl: string }): Promise<string> {
  // Vertical Shorts go up as Reels (no publish_video permission needed).
  // If Reels is refused, fall back to a normal Page video below.
  let reelErr: any = null;
  if (CFG.format === 'shorts') {
    try { return await publishFacebookReel(meta); } catch (e) { reelErr = e; log(`Facebook Reel failed (${(e as any)?.message || e}); trying a normal Page video.`); }
  }
  try { return await publishFacebookPageVideo(meta); }
  catch (e) { if (reelErr) throw new PipelineError('facebook_publish', `Reel: ${reelErr.message} · Video: ${(e as any)?.message || e}`); throw e; }
}

async function publishFacebookPageVideo(meta: { title: string; description: string; videoUrl: string }): Promise<string> {
  const c = SOCIAL.facebook;
  const body = new URLSearchParams({ file_url: meta.videoUrl, title: meta.title.slice(0, 200), description: socialCaption(meta.title, meta.description), published: 'true', access_token: c.pageAccessToken });
  const r = await fetch(`https://graph.facebook.com/${encodeURIComponent(c.pageId)}/videos`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body, signal: AbortSignal.timeout(120000) });
  const d: any = await r.json().catch(() => ({}));
  if (!r.ok || !d.id) throw new PipelineError('facebook_publish', d?.error?.message || `Facebook video publish failed (HTTP ${r.status}).`);
  try {
    const pr = await fetch(`https://graph.facebook.com/${encodeURIComponent(d.id)}?fields=permalink_url&access_token=${encodeURIComponent(c.pageAccessToken)}`, { signal: AbortSignal.timeout(30000) });
    const pd: any = await pr.json().catch(() => ({}));
    if (pr.ok && pd.permalink_url) return String(pd.permalink_url);
  } catch {}
  return `https://www.facebook.com/${d.id}`;
}

async function waitForContainer(url: string, token: string, kind: 'instagram' | 'threads'): Promise<void> {
  const deadline = Date.now() + 8 * 60 * 1000;
  let last = '';
  while (Date.now() < deadline) {
    const fields = kind === 'instagram' ? 'status_code,status' : 'status,error_message';
    const r = await fetch(`${url}?fields=${fields}&access_token=${encodeURIComponent(token)}`, { signal: AbortSignal.timeout(30000) });
    const d: any = await r.json().catch(() => ({}));
    if (!r.ok) throw new PipelineError(`${kind}_publish`, d?.error?.message || `Could not check ${kind} media processing status (HTTP ${r.status}).`);
    const status = String(d.status_code || d.status || '').toUpperCase();
    if (status === 'FINISHED' || status === 'PUBLISHED' || status === 'READY') return;
    if (status === 'ERROR' || status === 'EXPIRED' || status === 'FAILED') throw new PipelineError(`${kind}_publish`, d?.error_message || `${kind} media processing failed (${status}).`);
    last = status || last;
    await new Promise((z) => setTimeout(z, 7000));
  }
  throw new PipelineError(`${kind}_publish`, `${kind} media processing timed out${last ? ` (last status: ${last})` : ''}.`);
}

async function publishInstagramVideo(meta: { title: string; description: string; videoUrl: string }): Promise<string> {
  const c = SOCIAL.instagram;
  const base = `https://graph.instagram.com/${encodeURIComponent(c.userId)}`;
  const create = await fetch(`${base}/media`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ media_type: 'REELS', video_url: meta.videoUrl, caption: socialCaption(meta.title, meta.description), access_token: c.accessToken }), signal: AbortSignal.timeout(120000) });
  const d: any = await create.json().catch(() => ({}));
  if (!create.ok || !d.id) throw new PipelineError('instagram_publish', d?.error?.message || `Instagram container creation failed (HTTP ${create.status}).`);
  await waitForContainer(`https://graph.instagram.com/${encodeURIComponent(d.id)}`, c.accessToken, 'instagram');
  const pub = await fetch(`${base}/media_publish`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ creation_id: String(d.id), access_token: c.accessToken }), signal: AbortSignal.timeout(120000) });
  const pd: any = await pub.json().catch(() => ({}));
  if (!pub.ok || !pd.id) throw new PipelineError('instagram_publish', pd?.error?.message || `Instagram publish failed (HTTP ${pub.status}).`);
  try {
    const pr = await fetch(`https://graph.instagram.com/${encodeURIComponent(pd.id)}?fields=permalink&access_token=${encodeURIComponent(c.accessToken)}`, { signal: AbortSignal.timeout(30000) });
    const rd: any = await pr.json().catch(() => ({}));
    if (pr.ok && rd.permalink) return String(rd.permalink);
  } catch {}
  return `https://www.instagram.com/reel/${pd.id}/`;
}

async function publishThreadsVideo(meta: { title: string; description: string; videoUrl: string }): Promise<string> {
  const c = SOCIAL.threads;
  const base = `https://graph.threads.net/${encodeURIComponent(c.userId)}`;
  // If Threads still says the text is too long, shorten and try again (never fail on the caption).
  let create: Response = null as any; let d: any = {};
  for (const max of [480, 380, 260, 150, 80]) {
    create = await fetch(`${base}/threads`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ media_type: 'VIDEO', video_url: meta.videoUrl, text: threadsText(meta.title, meta.description, max), access_token: c.accessToken }), signal: AbortSignal.timeout(120000) });
    d = await create.json().catch(() => ({}));
    if (create.ok && d.id) break;
    if (!/at most \d+ characters|too long/i.test(String(d?.error?.message || d?.error?.error_user_msg || ''))) break;
    log(`Threads caption too long at ${max} bytes; shortening and retrying.`);
  }
  if (!create.ok || !d.id) throw new PipelineError('threads_publish', d?.error?.message || `Threads container creation failed (HTTP ${create.status}).`);
  await waitForContainer(`https://graph.threads.net/${encodeURIComponent(d.id)}`, c.accessToken, 'threads');
  const pub = await fetch(`${base}/threads_publish`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ creation_id: String(d.id), access_token: c.accessToken }), signal: AbortSignal.timeout(120000) });
  const pd: any = await pub.json().catch(() => ({}));
  if (!pub.ok || !pd.id) throw new PipelineError('threads_publish', pd?.error?.message || `Threads publish failed (HTTP ${pub.status}).`);
  try {
    const pr = await fetch(`https://graph.threads.net/${encodeURIComponent(pd.id)}?fields=permalink&access_token=${encodeURIComponent(c.accessToken)}`, { signal: AbortSignal.timeout(30000) });
    const rd: any = await pr.json().catch(() => ({}));
    if (pr.ok && rd.permalink) return String(rd.permalink);
  } catch {}
  return c.username ? `https://www.threads.net/@${encodeURIComponent(c.username)}/post/${pd.id}` : `https://www.threads.net/`;
}

async function reportStatus(status: 'running' | 'failed' | 'completed' | 'skipped', step: string, progress: number, logLine: string, extra: Record<string, any> = {}) {
  const res = await appRequest('POST', STUDIO ? `/api/animate/runner/${STUDIO.id}/status` : `${campaignPath()}/status`, {
    status, step, progress, log: logLine, partNumber: CFG.partNumber, runId: CFG.runId, runUrl: CFG.runUrl, ...extra
  });
  if (res && res.status === 401) console.warn('The app rejected the status update (runner key mismatch).');
}

// ---------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------
function run(cmd: string, args: string[], opts: { timeoutMs?: number; input?: Buffer } = {}): Promise<{ code: number; stdout: Buffer; stderr: string }> {
  return new Promise((resolve) => {
    let child: any;
    try {
      child = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    } catch (err: any) {
      resolve({ code: -1, stdout: Buffer.alloc(0), stderr: String(err?.message || err) });
      return;
    }
    const out: Buffer[] = [];
    let err = '';
    let timer: any = null;
    if (opts.timeoutMs) {
      timer = setTimeout(() => { err += `\n[timeout after ${opts.timeoutMs}ms]`; try { child.kill('SIGKILL'); } catch {} }, opts.timeoutMs);
    }
    child.stdout.on('data', (d: Buffer) => out.push(d));
    child.stderr.on('data', (d: Buffer) => { err += d.toString(); if (err.length > 200000) err = err.slice(-100000); });
    child.on('error', (e: any) => { if (timer) clearTimeout(timer); resolve({ code: -1, stdout: Buffer.concat(out), stderr: err + String(e?.message || e) }); });
    child.on('close', (code: number | null) => { if (timer) clearTimeout(timer); resolve({ code: code ?? -1, stdout: Buffer.concat(out), stderr: err }); });
    if (opts.input) child.stdin.end(opts.input); else child.stdin.end();
  });
}

async function probeDuration(file: string): Promise<number> {
  const r = await run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', file]);
  const d = parseFloat(r.stdout.toString().trim());
  return Number.isFinite(d) && d > 0 ? d : 0;
}

async function imageSize(file: string): Promise<[number, number]> {
  const r = await run('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file]);
  const [w, h] = r.stdout.toString().trim().split(',').map((n) => parseInt(n, 10));
  return [w || 0, h || 0];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clampNum = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------------------
// 1. Script — free Gemini models (Groq fallback), written as timed scenes
// ---------------------------------------------------------------------------
/** A performance cue placed before the `index`-th spoken word of a scene. */
interface Cue { index: number; tag: string }
interface Scene { narration: string; shot: 'scene' | 'panel' | 'full'; emotion: string; imagePrompt: string; searchQuery: string; cues: Cue[]; productShot?: boolean; imageCredit?: string }

// ---------------------------------------------------------------------------
// Performance tags: the script writer places [tags] inside the narration right
// before the word where a change should land. They are removed from the voice
// and captions and turned into cues timed to that exact word.
// ---------------------------------------------------------------------------
const EMOTION_TAGS = ['neutral', 'happy', 'excited', 'sad', 'crying', 'serious', 'worried', 'scared', 'surprised', 'angry', 'calm', 'curious', 'laugh'];
const GESTURE_TAGS = ['look_image', 'look_left', 'look_right', 'look_up', 'think', 'nod', 'shake_head', 'lean_in',
  'point', 'explain', 'count', 'wave', 'shrug', 'hands_up', 'hand_chest', 'fist'];
const TAG_ALIASES: Record<string, string> = {
  joy: 'happy', joyful: 'happy', smile: 'happy', smiling: 'happy', cheerful: 'happy', warm: 'happy', amused: 'happy', hopeful: 'happy', proud: 'happy', relieved: 'happy',
  excitement: 'excited', thrilled: 'excited', enthusiastic: 'excited', energetic: 'excited',
  sadness: 'sad', sorrow: 'sad', somber: 'sad', melancholy: 'sad', grief: 'sad', lonely: 'sad', disappointed: 'sad', tearful: 'crying', cry: 'crying',
  tense: 'worried', anxious: 'worried', nervous: 'worried', uneasy: 'worried', concerned: 'worried', suspense: 'worried',
  fear: 'scared', afraid: 'scared', terrified: 'scared', horrified: 'scared', frightened: 'scared', dread: 'scared',
  shock: 'surprised', shocked: 'surprised', amazed: 'surprised', astonished: 'surprised', wow: 'surprised',
  anger: 'angry', furious: 'angry', frustrated: 'angry', mad: 'angry',
  focused: 'serious', grave: 'serious', stern: 'serious', urgent: 'serious', dramatic: 'serious', mysterious: 'serious', determined: 'serious',
  relaxed: 'calm', gentle: 'calm', soft: 'calm', thoughtful: 'curious', intrigued: 'curious', wonder: 'curious',
  look_at_image: 'look_image', look_at_picture: 'look_image', look_picture: 'look_image', look_screen: 'look_image', show: 'look_image', look_side: 'look_image',
  pointing: 'point', point_image: 'point', point_at_image: 'point', point_screen: 'point', gesture: 'explain', explaining: 'explain', open_hands: 'explain', present: 'explain', presenting: 'explain',
  one: 'count', step: 'count', counting: 'count', number: 'count', hello: 'wave', hi: 'wave', goodbye: 'wave', bye: 'wave', waving: 'wave',
  shrugging: 'shrug', dunno: 'shrug', whoa: 'hands_up', hands_up: 'hands_up', raise_hands: 'hands_up', heart: 'hand_chest', hand_on_heart: 'hand_chest', chest: 'hand_chest',
  fist_pump: 'fist', clenched: 'fist', laughing: 'laugh', laughs: 'laugh', haha: 'laugh', giggle: 'laugh', chuckle: 'laugh', chuckles: 'laugh', grin: 'happy',
  glance_left: 'look_left', glance_right: 'look_right', look_away: 'look_left', thinking: 'think', ponder: 'think', hmm: 'think',
  shake: 'shake_head', head_shake: 'shake_head', no: 'shake_head', yes: 'nod', nodding: 'nod', lean: 'lean_in', lean_forward: 'lean_in', whisper: 'lean_in'
};

function normaliseTag(raw: string): string | null {
  const t = String(raw ?? '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const v = TAG_ALIASES[t] || t;
  return EMOTION_TAGS.includes(v) || GESTURE_TAGS.includes(v) ? v : null;
}

const cleanNarration = (s: string) => s.replace(/[*_#`]/g, '').replace(/\s+/g, ' ').trim();
const speechTokens = (s: string) => cleanForSpeech(s).split(/\s+/).filter(Boolean);

/** Split "[sad] She waited. [look_image] The door..." into clean text + cues at word indexes. */
function parseTaggedNarration(raw: string): { text: string; cues: Cue[] } {
  const cues: Cue[] = [];
  let text = '';
  let last = 0;
  for (const m of raw.matchAll(/\[([A-Za-z][A-Za-z _-]{1,28})\]|\(([A-Za-z][A-Za-z _-]{1,28})\)/g)) {
    const tag = normaliseTag(m[1] || m[2]);
    if (!tag && m[2]) continue; // an ordinary parenthesis stays in the text
    text += raw.slice(last, m.index) + ' ';
    last = (m.index || 0) + m[0].length;
    if (tag) cues.push({ index: speechTokens(cleanNarration(text)).length, tag });
  }
  text += raw.slice(last);
  const clean = cleanNarration(text.replace(/\[[^\]]{0,40}\]/g, ' '));
  const count = speechTokens(clean).length;
  return { text: clean, cues: cues.map((c) => ({ ...c, index: Math.min(c.index, count) })) };
}

/**
 * Fallback direction for scripts that came back with few tags: read each
 * sentence's mood from its words so the face still follows the story.
 */
const MOOD_WORDS: [RegExp, string][] = [
  [/\b(died|dead|death|funeral|grave|lost|lonely|cried|tears|goodbye|miss(ed)? (him|her|you)|tragic|heartbroken|sorry)\b/i, 'sad'],
  [/\b(scream|screamed|blood|shadow|footsteps|knock(ing|ed)?|whisper(ed)?|creak(ed|ing)?|behind (me|her|him)|something moved|dark(ness)?|terrif)/i, 'scared'],
  [/\b(suddenly|can't believe|unbelievable|shocking|out of nowhere|no way|wait,|what\?)/i, 'surprised'],
  [/\b(warning|danger(ous)?|crisis|killed|war|storm|flood|crash|urgent|arrested|attack|emergency|record low|collapsed?)\b/i, 'serious'],
  [/\b(worried|nervous|strange|wrong|weird|uneasy|missing|nobody|locked|alone)\b/i, 'worried'],
  [/\b(haha|hilarious|laughed|so funny|joked?)\b/i, 'laugh'],
  [/\b(amazing|incredible|delicious|perfect|love|finally|won|win|best|beautiful|great news|good news|sunny|celebrat|crispy|golden|easy)\b/i, 'happy'],
  [/\b(furious|outrage|angry|betrayed|unfair|lied)\b/i, 'angry']
];

function autoCues(scene: Scene): Cue[] {
  const cues = [...scene.cues];
  if (cues.filter((c) => EMOTION_TAGS.includes(c.tag)).length >= 2) return cues;
  const tokens = speechTokens(scene.narration);
  let sentence: string[] = [];
  let startIdx = 0;
  const flush = (endIdx: number) => {
    const text = sentence.join(' ');
    sentence = [];
    const hasEmotion = cues.some((c) => EMOTION_TAGS.includes(c.tag) && c.index >= startIdx && c.index < endIdx);
    if (!hasEmotion) {
      const mood = MOOD_WORDS.find(([re]) => re.test(text));
      if (mood) cues.push({ index: startIdx, tag: mood[1] });
    }
    if (/!$/.test(text) && !cues.some((c) => c.tag === 'nod' && c.index >= startIdx && c.index < endIdx)) cues.push({ index: Math.max(startIdx, endIdx - 1), tag: 'nod' });
    startIdx = endIdx;
  };
  tokens.forEach((tok, i) => {
    sentence.push(tok);
    if (/[.!?]["')\]]*$/.test(tok) || i === tokens.length - 1) flush(i + 1);
  });
  return cues.sort((a, b) => a.index - b.index);
}

/** Every cue as an absolute time, pinned to the word it was written before. */
function timeCues(scenes: Scene[], words: Word[], times: { start: number; end: number }[]): { t: number; tag: string }[] {
  const out: { t: number; tag: string }[] = [];
  let tokenStart = 0;
  scenes.forEach((s, i) => {
    const count = speechTokens(s.narration).length;
    const inScene = words.filter((w) => (w.token ?? -1) >= tokenStart && (w.token ?? -1) < tokenStart + count);
    const firstWordAt = inScene[0]?.start ?? times[i].start;
    const cues = autoCues(s);
    if (!cues.some((c) => c.index === 0 && EMOTION_TAGS.includes(c.tag))) {
      const e = emotionTag(s.emotion);
      if (e) cues.unshift({ index: 0, tag: e });
    }
    for (const c of cues) {
      let t: number;
      if (c.index >= count) t = inScene.length ? inScene[inScene.length - 1].end : times[i].end;
      else if (c.index === 0) t = firstWordAt;
      else t = (inScene.find((w) => (w.token ?? -1) >= tokenStart + c.index)?.start) ?? firstWordAt;
      out.push({ t: +Math.max(0, t).toFixed(3), tag: c.tag });
    }
    tokenStart += count;
  });
  return out.sort((a, b) => a.t - b.t);
}

/** Scene-level emotion (the JSON "emotion" field) as a tag. */
function emotionTag(e: string): string | null {
  const t = normaliseTag(String(e || ''));
  return t && EMOTION_TAGS.includes(t) ? t : null;
}
interface Script {
  title: string;
  description: string;
  hashtags: string[];
  tags: string[];
  visualStyle: string;
  characters: string;
  scenes: Scene[];
  usedFallbackTemplate: boolean;
  model?: string;
  aiError?: string;
  sources?: string[];
  sourceHeadline?: string;
  premise?: string;
  /** Tech / tutorials: the product's official website (its real images and a screenshot are used). */
  officialUrl?: string;
  /** News / tech: the headline the video is about (its articles' photos are used). */
  sourceStory?: { title: string; source: string; link: string };
  /** Where every real image came from (shown on screen and in the description). */
  imageCredits?: string[];
  /** Full license attribution for every CC BY / public-domain image used. */
  imageAttributions?: string[];
  /** The researched facts this script was written from (non-story videos). */
  factPack?: FactPack;
  /** True once the independent fact check passed. */
  factChecked?: boolean;
  /** Links to the articles the story was verified with. */
  sourceLinks?: string[];
}

// ---------------------------------------------------------------------------
// Script writer pool: Gemini (keys rotated per run, instant failover) → Groq.
// An exhausted model is never retried in the same run (see llm.ts).
// ---------------------------------------------------------------------------
const LLM = new LlmPool({
  localUrl: CFG.localLlmUrl,
  visionUrl: CFG.localVlmUrl || undefined,
  log: (m) => log(m),
  seed: parseInt(crypto.createHash('md5').update(`${CFG.campaignId}:${CFG.partNumber}:${CFG.runId}`).digest('hex').slice(0, 6), 16)
});

function stripTags(s: string): string {
  return s.replace(/<!\[CDATA\[|\]\]>/g, '').replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();
}

/** Loose key for "is this the same story?" comparisons. */
const storyKey = (t: string) => String(t || '').toLowerCase().replace(/\s+-\s+[^-]+$/, '').replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length > 3).slice(0, 8).sort().join(' ');
function sameStory(a: string, b: string): boolean {
  const A = new Set(storyKey(a).split(' ')), B = storyKey(b).split(' ');
  if (!A.size || !B.length) return false;
  const shared = B.filter((w) => A.has(w)).length;
  return shared >= Math.max(3, Math.ceil(Math.min(A.size, B.length) * 0.6));
}

/** Real, recent headlines for tech/news — freshest first, never one we already covered. */
async function recentHeadlines(pastTitles: string[], pastSources: string[]): Promise<{ title: string; source: string; date: string; link: string; desc?: string }[]> {
  if (CFG.offline) return [];
  const topicBySub: Record<string, string> = {
    'latest smartphone': 'smartphone launch', 'ai reasoning models': 'new AI model released', 'silicon & processors': 'new processor chip announced',
    gadgets: 'new gadget launch', 'ai tools': 'new AI tool launched', world: 'world news', 'business & money': 'business news', 'science & health': 'science discovery',
    entertainment: 'entertainment news', sports: 'sports news'
  };
  const base = CFG.topic || topicBySub[CFG.subGenre.toLowerCase()] || (CFG.category === 'tech' ? 'new AI tool launched' : 'breaking news');
  const seen = [...pastTitles, ...pastSources, ...CFG.usedHeadlines];
  const items: { title: string; source: string; date: string; link: string; desc?: string; ts: number }[] = [];
  // Freshest window first; widen only if everything recent was already covered.
  for (const window of CFG.category === 'news' ? ['when:1d', 'when:2d', 'when:4d'] : ['when:2d', 'when:5d', 'when:10d']) {
    const url = `${CFG.newsBase}?q=${encodeURIComponent(`${base} ${window}`)}&hl=en-US&gl=US&ceid=US:en`;
    try {
      const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 AnimatoAutoPoster/4.0' }, signal: AbortSignal.timeout(20000) });
      if (!res.ok) continue;
      const xml = await res.text();
      for (const m of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
        const block = m[1];
        const title = stripTags((block.match(/<title>([\s\S]*?)<\/title>/) || [])[1] || '');
        const source = stripTags((block.match(/<source[^>]*>([\s\S]*?)<\/source>/) || [])[1] || '');
        const date = stripTags((block.match(/<pubDate>([\s\S]*?)<\/pubDate>/) || [])[1] || '');
        const link = stripTags((block.match(/<link>([\s\S]*?)<\/link>/) || [])[1] || '');
        const desc = ((block.match(/<description>([\s\S]*?)<\/description>/) || [])[1] || '').slice(0, 4000);
        if (!title) continue;
        const clean = source && title.endsWith(` - ${source}`) ? title.slice(0, -(source.length + 3)) : title;
        if (seen.some((s) => sameStory(s, clean)) || items.some((x) => sameStory(x.title, clean))) continue;
        items.push({ title: clean, source, date, link, desc, ts: Date.parse(date) || 0 });
      }
    } catch (err: any) {
      log(`Headline lookup failed (${err?.message}).`);
    }
    if (items.length >= 6) break;
  }
  // Google News unreachable or thin: Bing News gives real headlines with the publishers' own links.
  if (items.length < 4) {
    try {
      for (const b of await bingNews(base)) {
        if (seen.some((x) => sameStory(x, b.title)) || items.some((x) => sameStory(x.title, b.title))) continue;
        items.push({ title: b.title, source: b.outlet, date: '', link: b.url, desc: '', ts: 0 });
      }
      log(`Bing News added headlines (now ${items.length}).`);
    } catch { /* keep what we have */ }
  }
  items.sort((a, b) => b.ts - a.ts);
  log(`Found ${items.length} fresh headlines for "${base}" (skipped anything already covered).`);
  return items.slice(0, 10).map(({ ts, ...h }) => h);
}

function lengthSpec() {
  return IS_SHORTS
    ? { words: '135-165', minWords: 110, maxWords: 190, scenes: '7-10', minScenes: 5, maxScenes: 12, seconds: 'about 55-60 seconds' }
    : { words: '360-460', minWords: 280, maxWords: 520, scenes: '16-24', minScenes: 12, maxScenes: 28, seconds: 'about 2.5-3 minutes' };
}

/** Every non-story video is written from a researched, verified fact sheet. */
const TRUTH_RULES = `
TRUTH RULES (the most important rules — viewers must be able to trust every word)
- Every factual statement (numbers, prices, specs, dates, names, places, quotes, features, outcomes, rankings, "first/only/biggest") must come from the FACT SHEET below, or be universally known stable background. Nothing else.
- Never invent or "round up" a detail to fill time. If the sheet doesn't say it, don't say it — explain why it matters, give context, or ask the viewer a question instead.
- Anything marked UNCONFIRMED may only be said as "reportedly" / "according to …", or left out.
- Don't present speculation, predictions or opinions as facts. Attribute claims to the outlet or company that made them.
- Every number you say (amounts, prices, dates, scores, specs, times) is matched against the sheet before publishing; a sentence with a number the sheet doesn't contain is cut.
- If the sheet says it is from a SINGLE SOURCE, credit that outlet in the narration ("according to …").`;

function categoryBrief(pastStory: string, headlines: { title: string; source: string; date: string }[], pack: FactPack | null = null): string {
  const topic = CFG.topic ? `\nCreator's direction: "${CFG.topic}".` : '';
  const sub = CFG.subGenre ? `\nSub-genre: ${CFG.subGenre}.` : '';
  const sheet = pack ? `${TRUTH_RULES}\n\nFACT SHEET (researched and verified on the live web for this video)\n"""\n${factSheet(pack)}\n"""` : '';
  const news = pack?.headline
    ? pack.roundup
      ? `\nTHIS VIDEO IS A QUICK ROUNDUP of ${pack.roundup} fresh headlines (listed in the fact sheet). Give each headline its own short beat (one or two scenes), credit its outlet by name, and say only what the headline and its details state. Open with a hook about the day's biggest one. Put "${pack.headline.title}" in "sourceHeadline".${sheet}`
      : `\nTHE STORY FOR THIS VIDEO (verified by ${pack.outlets.length} independent outlet(s)): "${pack.headline.title}"${pack.headline.source ? ` (${pack.headline.source})` : ''}. Build the whole video around it. Mention a source naturally once. Put this exact headline in "sourceHeadline".${sheet}`
    : headlines.length
    ? `\nFRESH HEADLINES (newest first; none of these has been covered on this channel before). Use ONLY facts stated here — do not invent numbers, prices, specs, quotes, names or dates:\n${headlines.map((h, i) => `${i + 1}. ${h.title}${h.source ? ` (${h.source}${h.date ? `, ${h.date.slice(0, 16)}` : ''})` : ''}`).join('\n')}\nPick the single most important/interesting story (prefer #1-#3, the newest) and build the whole video around it. Mention the source naturally once. Put the exact headline you used in "sourceHeadline".${TRUTH_RULES}`
    : '';
  switch (CFG.category) {
    case 'cooking':
      return `FORMAT: a narrated cooking tutorial${sub}${topic}${pack?.recipe ? `\n- THE DISH: ${pack.subject}. Teach exactly the VERIFIED RECIPE in the fact sheet (same amounts, times, temperatures and order).${sheet}` : `${TRUTH_RULES}\n- Pick ONE specific, genuinely good dish (different from the previous videos listed below) and use a standard, well-tested recipe for it with correct, food-safe times and temperatures.`}
- Scene 1 is the HOOK: a mouth-watering promise or a surprising tip, max 14 words ("The secret to crispy fried rice is day-old rice, and here's why.").
- Then: ingredients with exact amounts, then clear step-by-step instructions with times/temperatures, one pro tip, and a satisfying final plating moment.
- End with a one-line call to action (ask a question viewers will answer in the comments).
- Use shot "panel" for ingredient and step scenes (the presenter points at the photo), "scene" for the hook and the final dish.
- ONE thing per scene on screen: an ingredient scene shows that ingredient, a step scene shows that step, so the viewer is guided visually step by step.
- searchQuery (every scene): the ONE main thing visible at that moment in its plain common name, 1-3 words, as you'd type it into a photo search — an ingredient ("tomatoes", "red onions", "scotch bonnet peppers", "parboiled rice"), or the dish/step ("jollof rice", "frying plantain", "chopped tomatoes"). Never a sentence, never two ingredients.
- imagePrompt (every scene): a realistic overhead or 45-degree food photo of exactly that ingredient or step (e.g. "fresh ripe tomatoes on a wooden board, natural light"). Used only if no real photo is found.`;
    case 'tech':
      return `FORMAT: a tech / AI tool tutorial-review${sub}${topic}${news}
- Cover ONE real, newly released or trending AI tool, app, model or gadget${pack ? ' — the one in the verified story above' : headlines.length ? ' from the headlines above' : ''}.
- Scene 1 is the HOOK (max 14 words): the most useful or surprising thing it does for the viewer ("This free AI tool turns a photo into a 3D model in seconds.").
- Then, tutorial style: WHAT it is (one line) → the problem it solves / who it helps → WHERE to get it (official website, app store or platform by name — never invent a URL) → HOW to use it in 3-5 concrete steps ("Open…", "Upload…", "Type a prompt like…", "Export…") → one pro tip → one honest limitation → a clear verdict.
- Talk like a friendly expert showing a friend, not an ad. Never state a spec, price, date or feature that is not in the fact sheet.
- Teach, don't announce: the viewer should finish knowing exactly what it does for THEM, where to find it and what to click first. Say the steps out loud ("[count] Step one: open…"), and react to what impresses you.
- Use shot "panel" for most scenes: the presenter points at the image of the tool/step.
- The pictures are REAL images found on the web — the product's own website and screenshots, the news articles about it, press photos — never generated. So:
  - "officialUrl": the tool's official website or product page (e.g. "https://gemini.google.com"). Only a URL you are sure is real; otherwise leave it empty.
  - searchQuery (every scene): name the real product, company, person or device EXACTLY, with the full model name/number (e.g. "Samsung Galaxy S25 Ultra", "iPhone 16 Pro", "Nvidia Blackwell GPU", "Sam Altman"), 2-6 words, no generic words like "technology" or "AI concept". The picture must show that exact product.${pack || headlines.length ? '' : '\n- No headlines were available: pick a well-known, clearly real AI tool and stay factual.'}`;
    case 'ads': {
      const brief = CFG.adBrief
        ? `\nTHE PRODUCT (read from the advertiser's own PDF — use ONLY these facts, never invent a price, feature, claim or link):\n"""${CFG.adBrief.slice(0, 5000)}"""`
        : '\nNo product document was provided: write a clean, honest teaser for the product named in the creator\'s direction and invent nothing.';
      const adRules = CFG.adBrief ? TRUTH_RULES.replace(/the FACT SHEET below/g, 'THE PRODUCT document above').replace(/the sheet/g, 'the document') : '';
      const P = CFG.adProfile;
      const who = P.company
        ? `\nTHE ADVERTISER: "${P.company}". Name them in the hook or the first two scenes and again in the call to action, as the ones behind it (e.g. "${P.company} just built…", "${P.company} sets this up for you"). Put "${P.company}" in the title too.`
        : '';
      const service = P.service
        ? `\nTHIS IS A SERVICE, NOT SOMETHING TO DOWNLOAD: there is no app to install and no website to sign up on. Never say "download", "install", "get the app", "visit the website", "link in bio" or name an app store or URL. ${P.company || 'The company'} sets it up and customises it for each customer's own business or channel, so they don't have to do anything themselves. The call to action is to MESSAGE ${P.company || 'them'}${P.contact ? ` — say exactly: "${P.contact}"` : ' (say "send us a message" / "DM us")'} to have it built for their business. Frame features as what the customer gets ("you get…", "we set up…"), and leave "officialUrl" empty.`
        : '';
      const steps = P.service
        ? 'what it does in one line → who it is for → the 2-3 features that matter, each with the benefit in plain words → that ' + (P.company || 'the company') + ' sets it all up for them (done for you) → the offer or price ONLY if the document states it → the call to action: message them.'
        : 'what it is in one line → who it\'s for → the 2-3 features that matter, each with the benefit in plain words → how to get it (the exact site, app store or plan named in the document) → the offer or price ONLY if the document states it → a clear call to action.';
      const PR = CFG.adPromo;
      const promo = PR
        ? `\nPROMO RUNNING NOW (set by the advertiser — this overrides "price only if the document states it"): ${PR.discount ? `${PR.discount} off` : 'a special offer'}${PR.details ? ` — ${PR.details}` : ''}${PR.code ? `. Promo code: ${PR.code}` : ''}${PR.ends ? `. Ends: ${PR.ends}` : ''}. Mention the promo clearly in ONE scene near the end, just before the call to action (e.g. "Right now ${P.company || 'they'}'re running a promo — ${PR.discount || 'a special discount'} off${PR.code ? `, use code ${PR.code}` : ''}."), and put it in the description. Use exactly these terms — never invent a different amount, code or deadline.`
        : '\nNO PROMO is running: never mention a discount, sale, promo code or limited-time offer unless the document itself states one.';
      return `FORMAT: a short, honest ${P.service ? 'advert for a done-for-you service' : 'product advert'} that viewers actually enjoy${sub}${topic}${brief}${adRules}${who}${service}${promo}
- Scene 1 is the HOOK (max 14 words): the problem the viewer has, or the single best thing this ${P.service ? 'service' : 'product'} does ("Your meeting notes write themselves now — here's how.").
- Then: ${steps}
- The presenter genuinely likes it and speaks from experience: warm, specific, never shouty, no fake urgency, no invented testimonials.
- Use shot "panel" whenever the product is shown, and set "productShot": true on those scenes so the real product photo from the PDF is used.
- searchQuery (scenes without a product photo): the product or company name as written in the document, or the real everyday setting it is used in ("small business owner laptop"). Real photos only — nothing is generated.`;
    }
    case 'news':
      return `FORMAT: a 60-second news explainer${sub}${topic}${news}
- Scene 1 is the HOOK: what happened, in max 14 words, in plain language.
- Then: the key facts (who, what, where, when) exactly as the fact sheet states them, why it matters to the viewer, and what happens next. Neutral, accurate, no speculation, no opinions.
- It must be a story that is NOT in the list of previous video titles below.
- Use shot "panel" for fact scenes. The pictures are the REAL photos from the news articles about this story (never generated), so searchQuery must name the real place/person/organisation/event exactly as a news photo caption would (e.g. "Lagos flooding", "Bola Tinubu", "SpaceX Starship launch"), 2-6 words.
- The presenter reacts like someone who has followed the story: a beat of surprise at the number that matters, [lean_in] for the human detail, [serious] for the consequence. Viewers must feel this really happened, not that a page is being read.
- Close with what to watch for next — only what the sources say is scheduled or expected, never your own prediction.${pack || headlines.length ? '' : '\n- No headlines were available: explain one important, well-established recent development without inventing details.'}`;
    default: {
      const tone = /horror|suspense|scary/i.test(CFG.subGenre) ? 'village/small-town horror: slow dread, a real folk-evil or haunting, sensory detail (cold air, oil lamps, footsteps on sand), frightening but never gory'
        : /mystery/i.test(CFG.subGenre) ? 'a gripping mystery with clues the viewer can follow and a fair, surprising answer'
        : /twist/i.test(CFG.subGenre) ? 'a clean setup, quiet misdirection and a twist that recontextualises everything'
        : /love|romance/i.test(CFG.subGenre) ? 'warm, emotional, bittersweet and hopeful'
        : /comed|funny/i.test(CFG.subGenre) ? 'a comedy: a silly situation that keeps escalating, funny characters, a laugh-out-loud payoff'
        : /drama/i.test(CFG.subGenre) ? 'a human drama: real people, a hard choice, an honest and moving turn'
        : /sci-?fi|science/i.test(CFG.subGenre) ? 'science fiction: one surprising invention or future idea changes an ordinary life'
        : /thrill/i.test(CFG.subGenre) ? 'a thriller: a secret, a threat, rising tension and a sharp twist'
        : /fantasy/i.test(CFG.subGenre) ? 'fantasy adventure: magic in the everyday world, wonder and courage'
        : /folk/i.test(CFG.subGenre) ? 'a folk tale told around a fire: a clever hero, a lesson, a memorable ending'
        : 'gripping, emotional, cinematic';
      const part = CFG.partNumber, last = CFG.arcParts;
      const known = CFG.storyPremise
        ? `\nTHIS STORY (keep every name, place and fact exactly):\n${CFG.storyPremise}${CFG.storyCharacters ? `\nCharacters: ${CFG.storyCharacters}` : ''}${CFG.storyTitle ? `\nSeries title: ${CFG.storyTitle}` : ''}`
        : '';
      const cont = pastStory ? `\nWHAT HAPPENED IN THE EARLIER PARTS:\n${pastStory}` : '';
      const arcStep = part <= 1
        ? `PART 1 of ${last} — SET UP AND IGNITE.
- Open on the protagonist by name in a specific, vivid place ("Anna sold roasted corn at the junction in Umuoka, a village where nobody walked after 9 p.m.").
- Establish what they want, the ordinary rule of their world, and the ONE thing that breaks it.
- End on the first real shock, so Part 2 is unmissable.`
        : part >= last
          ? `PART ${part} of ${last} — CLIMAX AND FULL ENDING.
- Pay off everything: the truth is revealed, the protagonist acts, the wrongdoer faces the consequence, and the reader learns what happened to everyone.
- NO cliffhanger, NO "part ${part + 1}", NO unanswered question. Close the story completely with a final line that lands (justice, a cost, a lesson, or a chilling last image).
- Finish with one line inviting the viewer to the NEXT story on the channel.`
          : `PART ${part} of ${last} — RAISE THE STAKES AND TURN.
- Deepen the danger and reveal something that changes how the viewer reads Part 1 (who is really behind it, what the protagonist did, what is at stake).
- The protagonist must DO something, fail or half-succeed, and end the part in worse trouble than they started.`;
      return `FORMAT: a complete short story told by a NARRATOR in the THIRD PERSON — a real story with a plot, not a monologue${sub}${topic}${known}${cont}
- The presenter is the storyteller and NEVER a character in it. Use names and he/she/they, never "I" for the protagonist.
- The whole story runs for EXACTLY ${last} parts and this is part ${part}.
${arcStep}
- Tone: ${tone}.
- A story means: a named protagonist with a want, a specific place (village, compound, market, boarding school, church, city flat), other named people who do things, a wrongdoing or a threat, rising consequences, and a clear ending. Things must HAPPEN — dialogue-in-narration, actions, choices, consequences — never vague musing.
- Scene 1 is the HOOK (max 16 words): one concrete, impossible-to-scroll-past fact about this story.
- Every scene moves the plot: new information, a new action or a new consequence. No repetition, no filler, no summarising what was just said.
- Short spoken sentences, past tense, plain words. Keep the viewer feeling it: sounds, smells, small physical details.
${part >= last ? '- The title must NOT contain "(Part ...)" if the story ends here; instead make it the story\'s own title.' : `- Title must end with "(Part ${part})".`}
- Also return "premise": 2-3 sentences of what this story is about, who is in it and what has happened so far (the next part is written from this), and "characters": each named person's fixed look (age, build, hair, clothes) for the pictures.
- imagePrompt: describe the exact moment of that scene as a still from a high-end 3D animated family film (big-studio feature quality) — WHO (named character as an ORIGINAL stylised cartoon character + their fixed look; never an existing movie character), WHERE (a cartoon version of the place), WHAT is happening, the light and the camera angle. Every person is a cartoon character with big expressive eyes and soft rounded features — never a real or photorealistic human. FRAMING (important — the image model draws faces and torsos well but mangles hands, full-body action and crowds): frame every scene as a medium shot or close-up of ONE or TWO characters, waist-up or head-and-shoulders, in a calm simple pose (standing, sitting, talking, looking), hands low or out of frame; put the story's action and place in the background and props. Never full-body running / jumping / fighting poses, never crowds or groups, never close-ups of fingers.`;
    }
  }
}

function buildPrompt(pastStory: string, pastTitles: string[], headlines: any[], pack: FactPack | null = null): string {
  const L = lengthSpec();
  const avoid = pastTitles.length ? `\nPrevious video titles (do NOT repeat these topics): ${pastTitles.slice(-35).join(' | ')}` : '';
  return `You are writing a ${IS_SHORTS ? 'YouTube Short (vertical)' : 'YouTube video (16:9)'} of ${L.seconds}, narrated by an animated presenter${CFG.presenterName ? ` called ${CFG.presenterName}` : ''}.
${categoryBrief(pastStory, headlines, pack)}${avoid}${CFG.performanceNotes ? `\n\nWHAT IS WORKING ON THIS CHANNEL (learn from it):\n${CFG.performanceNotes}` : ''}${CFG.presenterName ? `\nPRESENTER NAME: ${CFG.presenterName}. Whenever the presenter introduces themself or signs off, use this name (e.g. "Hi, I'm ${CFG.presenterName}") — never any other name.` : ''}

RULES
- Total narration: ${L.words} words across ${L.scenes} scenes. Each scene is 1-3 spoken sentences (8-40 words).
- Write for the ear: short sentences, concrete words, no emojis, no hashtags, no stage directions, no "In this video".
- Suitable for a general YouTube audience (PG-13): tension and mystery are great; no gore, no graphic violence, no self-harm, nothing sexual.
- Every scene gets its own image that shows exactly what is being said at that moment.

PERFORMANCE TAGS (the presenter is an animated character with a face, head, arms and hands; it performs tags you write INSIDE "narration")
- Put a tag right before the word where the change should land. Tags are never spoken or shown as captions.
- Emotion tags (the face keeps it until the next emotion tag): [neutral] [calm] [happy] [excited] [curious] [serious] [worried] [scared] [surprised] [sad] [crying] [angry] [laugh]
- Head/eye tags: [look_image] turn and look at the picture, [look_left] [look_right] glance aside, [look_up], [think] ponder, [nod], [shake_head], [lean_in] for a secret or key point.
- Hand/body tags: [point] point at the picture on screen, [explain] open-palm explaining gesture, [count] hold up a finger for a step or item, [wave] wave hello/goodbye, [shrug] "who knows?", [hands_up] "whoa!", [hand_chest] heartfelt/sad, [fist] determined/emphasis.
- Start EVERY scene with an emotion tag and change emotion whenever the feeling of the words changes, exactly like a real presenter. Use [laugh] only for genuinely funny moments and [crying] only for truly heartbreaking ones.
- Example: "[serious] Heavy rain flooded the coast overnight. [point] This is Main Street this morning. [happy] But the good news? [nod] The weekend looks sunny." / tutorial: "[explain] First, open the app. [count] Step one: upload your photo."
- Use 2-4 tags per scene overall, including a hand/body tag in most scenes; [point] or [look_image] whenever the words refer to what is on screen; [wave] in the first or last scene. Never use a tag that contradicts the words.
- The presenter LIVES the story: react as a person telling it to a friend — [lean_in] for a secret, [hands_up] at a shock, [laugh] at something funny, [crying] at heartbreak, [shake_head] at something wrong, [fist] at injustice. A flat, unreacting delivery is a failure.

YOUTUBE PACKAGING
- "title": max 70 characters, curiosity + the main keyword, honest (no false clickbait), Title Case.
- "description": a DETAILED, well-written description of 120-250 words in plain text (no hashtags, no markdown, no emoji spam, no "In this video we will" filler). Paragraph 1: a strong hook that names the exact subject (the real names of the people, product, place or dish). Paragraph 2-3: ${descriptionBrief()}. Last line: one specific question about THIS video that invites comments. Every statement must be true and match the script.
- "hashtags": 5-8 lowercase hashtags without "#", each about THIS video's actual subject: the specific names in it (person, product, brand, place, dish), the precise niche and the topic people search for — e.g. for a jollof rice video "jollofrice", "nigerianfood", "westafricanfood", "ricerecipe", "cookingtutorial". NEVER generic filler ("viral", "fyp", "foryou", "trending", "explore", "reels", "shorts", "love", "instagood", "follow", "like", "subscribe", "video", "new") and never a tag unrelated to the video.
- "tags": 8-15 search phrases.

Return ONLY this JSON (no markdown):
{
  "title": "...",
  "description": "...",
  "hashtags": ["..."],
  "tags": ["..."],
  "visualStyle": "one consistent look for every image, e.g. 'dark cinematic film still, cold blue shadows, 35mm, moody practical lighting'",
  "characters": "fixed physical description of each recurring named person for consistent images (or empty)",
  "sourceHeadline": "the exact headline used (tech/news only, else empty)",
  "officialUrl": "tech/tutorials: the official website of the product (real URL only), else empty",
  "premise": "stories only: 2-3 sentences — who this story is about, where, and everything that has happened so far",
  "scenes": [
    { "narration": "[emotion] spoken words with [gesture] tags where they land", "shot": "scene|panel|full", "emotion": "main emotion of the scene", "imagePrompt": "stories only: the animated scene to draw", "searchQuery": "real things to find a real photo of (names of people, products, places, dishes)", "productShot": false }
  ]
}`;
}

/** Same request in softer words, for providers whose moderation flags horror/crime vocabulary. */
function saferScriptPrompt(p: string): string {
  const swaps: [RegExp, string][] = [
    [/\b(blood(y|ied)?|gore|gory|guts|bleeding)\b/gi, 'dark'],
    [/\b(corpse|dead body|cadaver)\b/gi, 'shadowy figure'],
    [/\b(murder(ed|er|ing|s)?|kill(ed|er|ing|s)?|slaughter(ed)?|stab(bed|bing)?|strangl(ed|ing))\b/gi, 'crime'],
    [/\b(suicide|self-harm)\b/gi, 'loss'],
    [/\b(knife|knives|gun|pistol|rifle|weapon|axe|machete)\b/gi, 'object'],
    [/\b(demon(ic)?|possessed|satanic)\b/gi, 'unexplained'],
    [/\b(naked|nude)\b/gi, 'alone'],
    [/\b(torture(d)?|gruesome|mutilat\w*)\b/gi, 'terrible']
  ];
  let out = p;
  for (const [re, to] of swaps) out = out.replace(re, to);
  return `${out}

SAFE MODE: write it for a general audience (PG-13). Suspense, mystery and emotion only — no gore, no graphic violence, no self-harm, nothing sexual, no real people.`;
}

const extractJson = extractJsonObject;

function cleanHashtag(h: any): string {
  return String(h || '').toLowerCase().replace(/^#/, '').replace(/[^a-z0-9]/g, '').slice(0, 30);
}

const DEFAULT_TAGS: Record<string, string[]> = {
  ads: ['productreview', 'tools', 'smallbusiness', 'tech'],
  stories: ['storytime', 'scarystories', 'horrorstory', 'creepy'],
  cooking: ['recipe', 'cooking', 'easyrecipe', 'foodie'],
  tech: ['tech', 'technews', 'gadgets', 'ai'],
  news: ['news', 'breakingnews', 'worldnews', 'explained'],
  musical: ['originalmusic', 'musicvideo', 'song', 'songs', 'singer', 'choir']
};

/** What the description's body must cover, per category. */
function descriptionBrief(): string {
  switch (CFG.category) {
    case 'news': return 'the key verified facts — what happened, who is involved, where and when — and why it matters to the viewer; say "reportedly" for anything not confirmed';
    case 'tech': return 'what the product/tool/update is, who makes it, what it actually does, how to get or use it (the real steps shown) and who it is for';
    case 'cooking': return 'the dish and where it comes from, the main ingredients, a short summary of the method with the key time/temperature, and the tip from the video';
    case 'ads': return 'what the product is, the real features and benefits from the brief (nothing invented), who it is for and how to get it';
    default: return 'a gripping spoiler-free teaser of the story — who it is about, where it happens and the mystery or danger they face — without revealing the ending or the twist';
  }
}

/** Hashtags nobody should ever use: generic reach-bait that says nothing about the video. */
const JUNK_HASHTAGS = new Set(('viral viralvideo viralvideos fyp fypage fypシ foryou foryoupage foryourpage trending trend trendingnow explore explorepage reels reel reelsinstagram reelitfeelit shorts short youtubeshorts shortvideo shortsvideo ytshorts tiktok tiktokviral instagood instagram insta instadaily photooftheday picoftheday love like likes likeforlike likeforlikes follow followme followforfollow follow4follow subscribe sub video videos new newvideo today daily best top amazing awesome cool fun funny wow omg lol goodvibes happy beautiful cute life lifestyle motivation inspiration content contentcreator creator facebook fb meta youtube youtuber watch share comment blowup blowthisup nofilter tbt the and for with this that you your part episode').split(' '));
const STOPWORDS = new Set('a an the and or but of to in on at for with from by as is are was were be been it its this that these those you your we our they their he she his her them i me my not no so than then there here what when where who why how all any can will just into over out up down about after before more most very also has have had do does did new video'.split(' '));

/** Hype words that make meaningless tags on their own ("#perfect", "#shocking"). */
const HYPE_WORDS = new Set('perfect best ultimate amazing incredible insane crazy shocking secret secrets truth finally really never ever always simple quick easy easiest fastest biggest huge must watch everyone nobody something anything nothing everything people thing things time times year years inside behind while could would should home made make makes making real still first last next part finale'.split(' '));
/** Words that make a tag clearly about the category even when the script never says them. */
const CATEGORY_ROOTS: Record<string, string[]> = {
  cooking: ['food', 'recipe', 'cook', 'kitchen', 'meal', 'dish', 'dinner', 'lunch', 'breakfast', 'bake', 'cuisine', 'foodie', 'snack', 'dessert'],
  tech: ['tech', 'gadget', 'software', 'app', 'phone', 'computer', 'coding', 'digital', 'tutorial', 'howto'],
  news: ['news', 'headline', 'breaking', 'update', 'report', 'explained', 'currentevents'],
  musical: ['music', 'song', 'songs', 'singer', 'vocal', 'vocals', 'choir', 'musicvideo', 'originalmusic', 'afrobeats', 'gospel', 'nasheed', 'rnb', 'pop', 'dance'],
  ads: ['review', 'product', 'shop', 'business', 'brand', 'deal'],
  stories: ['story', 'stories', 'tale', 'storytime', 'drama', 'romance', 'mystery']
};

const wordsOf = (t: string) => String(t || '').toLowerCase().replace(/[’']/g, '').split(/[^a-z0-9]+/).filter(Boolean);

/** Every 1-3 word run in the video's own words, joined — a hashtag must be one of these (or a known niche tag). */
function subjectVocabulary(sc: Script): Set<string> {
  const texts = [sc.title, sc.description, sc.premise || '', sc.sourceHeadline || '', sc.factPack?.subject || '', sc.factPack?.productName || '', CFG.subGenre, CFG.topic,
    ...sc.scenes.map((x) => `${x.narration} ${x.searchQuery || ''}`), ...(sc.tags || [])];
  const vocab = new Set<string>();
  for (const t of texts) {
    const w = wordsOf(t);
    for (let i = 0; i < w.length; i++) for (let n = 1; n <= 3 && i + n <= w.length; n++) {
      const run = w.slice(i, i + n);
      if (n === 1 && STOPWORDS.has(run[0])) continue;
      const j = run.join('');
      vocab.add(j);
      if (j.endsWith('s')) vocab.add(j.slice(0, -1)); else vocab.add(`${j}s`);
    }
  }
  return vocab;
}

/** Well-known niche tags that are always relevant to a category (people search them). */
const NICHE_TAGS: Record<string, string[]> = {
  stories: ['storytime', 'scarystories', 'horrorstory', 'horrorstories', 'creepypasta', 'mysterystory', 'shortstory', 'animatedstory', 'suspense', 'thriller', 'ghoststory', 'truescarystories', 'bedtimestory', 'lovestory', 'dramastory'],
  cooking: ['recipe', 'recipes', 'cooking', 'easyrecipe', 'easyrecipes', 'homecooking', 'cookingtutorial', 'foodie', 'dinnerideas', 'lunchideas', 'breakfastideas', 'mealprep', 'comfortfood', 'healthyrecipes'],
  tech: ['tech', 'technews', 'technology', 'gadgets', 'ai', 'artificialintelligence', 'techtips', 'tutorial', 'howto', 'software', 'apps', 'smartphone', 'productivity'],
  news: ['news', 'breakingnews', 'worldnews', 'newsupdate', 'currentevents', 'explained', 'headlines', 'politics', 'economy'],
  musical: ['originalmusic', 'musicvideo', 'song', 'songs', 'singer', 'choir', 'afrobeats', 'gospelmusic', 'worshipmusic', 'nasheed', 'devotionalmusic', 'rnbmusic', 'popmusic', 'dancemusic'],
  ads: ['productreview', 'smallbusiness', 'shopsmall', 'musthave', 'productdemo']
};

function cleanHashtagList(raw: any[], sc: Script): string[] {
  const vocab = subjectVocabulary(sc);
  const niche = new Set(NICHE_TAGS[CFG.category] || NICHE_TAGS.stories);
  // Words of the video itself (4+ letters) and the category's own vocabulary: a tag must contain one.
  const roots = [...vocab].filter((w) => w.length >= 4 && w.length <= 14 && !STOPWORDS.has(w) && !HYPE_WORDS.has(w));
  const catRoots = CATEGORY_ROOTS[CFG.category] || CATEGORY_ROOTS.stories;
  const scary = /horror|scary|creepy|ghost|haunt|suspense|thriller|mystery|spooky|paranormal/i.test(`${CFG.subGenre} ${CFG.topic} ${sc.title} ${sc.premise || ''}`);
  const relevant = (h: string) => vocab.has(h) || niche.has(h) || roots.some((r) => h.includes(r)) || catRoots.some((r) => h.includes(r));
  const ok = (h: string) => (h.length >= 3 || niche.has(h)) && h.length <= 28 && !/^\d+$/.test(h) && !JUNK_HASHTAGS.has(h) && !STOPWORDS.has(h) && !HYPE_WORDS.has(h)
    && relevant(h) && !(CFG.category === 'stories' && !scary && /horror|scary|creepy|ghost|haunt|creepypasta/.test(h) && !vocab.has(h));
  const out: string[] = [];
  const add = (h: string) => { if (ok(h) && !out.includes(h) && !out.some((o) => o === `${h}s` || `${o}s` === h)) out.push(h); };
  for (const h of raw.map(cleanHashtag)) add(h);
  if (out.length < 5) {
    // Derive from the video itself: the named subject first, then its key words.
    const subj = [sc.factPack?.productName, sc.factPack?.subject?.split(/[:\-–—|,(]/)[0]].filter(Boolean) as string[];
    for (const s of subj) { const w = wordsOf(s).filter((x) => !STOPWORDS.has(x)); if (w.length && w.length <= 3) add(w.join('')); }
    const titleWords = wordsOf(sc.title.replace(/\((part \d+|finale)\)/i, '')).filter((w) => w.length >= 5 && !STOPWORDS.has(w) && !HYPE_WORDS.has(w) && !/^\d+$/.test(w));
    for (const w of titleWords) if (out.length < 3) add(w);
    const defaults = CFG.category === 'stories'
      ? (scary ? ['storytime', 'scarystories', 'horrorstory', 'creepypasta'] : ['storytime', 'shortstory', 'animatedstory', 'dramastory'])
      : [...(DEFAULT_TAGS[CFG.category] || []), ...(NICHE_TAGS[CFG.category] || [])];
    for (const d of defaults) if (out.length < 5) add(d);
  }
  return out.slice(0, 8);
}

/** Plain, readable description text: no hashtags, markdown, links-as-markdown, emoji spam or JSON residue. */
function cleanDescriptionText(t: string): string {
  let d = String(t || '')
    .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '$1 $2')
    .replace(/(^|\s)#[\p{L}\p{N}_]+/gu, '$1')
    .replace(/\*\*|__|`+|^#+\s*/gm, '')
    .replace(/^\s*(description|summary)\s*:\s*/i, '')
    .replace(/\[(\w+)\]/g, '')
    .replace(/(\p{Extended_Pictographic}️?){2,}/gu, (m) => [...m][0]);
  const lines = d.split(/\r?\n/).map((l) => l.replace(/[ \t]+/g, ' ').trim());
  d = lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  if (/^\.{3}$|^(n\/a|none|tbd|description)$/i.test(d)) return '';
  return d;
}

const firstSentence = (t: string) => { const m = String(t ?? '').match(/^.{20,180}?[.!?](\s|$)/); return (m ? m[0] : String(t).slice(0, 160)).trim(); };

/** A readable, detailed body built from the (fact-checked) script when the writer's description is thin. */
function descriptionFromScript(sc: Script): string {
  const scenes = sc.scenes.map((x) => x.narration.trim()).filter(Boolean);
  if (CFG.category === 'stories') {
    // Only the set-up — never the ending or the twist.
    const setup = scenes.slice(0, Math.max(2, Math.ceil(scenes.length * 0.3))).join(' ');
    return [sc.premise && sc.premise.length > 60 ? sc.premise : setup.slice(0, 600)].join('\n\n');
  }
  return scenes.slice(0, 3).join(' ').slice(0, 700);
}

const COMMENT_QUESTION: Record<string, string> = {
  news: 'What do you think happens next? Tell us in the comments.',
  tech: 'Would you use this? Tell us in the comments.',
  cooking: 'Would you try this recipe? Tell us how yours turned out in the comments.',
  ads: 'Have questions about it? Ask in the comments.',
  stories: 'What would you have done? Tell us in the comments.'
};

/** Clean and complete the title, description and hashtags of a finished script. */
function polishMetadata(sc: Script): void {
  let d = cleanDescriptionText(sc.description);
  const words = d.split(/\s+/).filter(Boolean).length;
  if (words < 60) {
    const body = descriptionFromScript(sc);
    d = d ? `${d}\n\n${body}` : body;
  }
  if (!/\?\s*$/.test(d.split('\n').filter(Boolean).pop() || '')) d = `${d}\n\n${COMMENT_QUESTION[CFG.category] || COMMENT_QUESTION.stories}`;
  sc.description = d.slice(0, 2200).trim();
  sc.hashtags = cleanHashtagList(Array.isArray(sc.hashtags) ? sc.hashtags : [], sc);
  sc.tags = Array.from(new Set([...(sc.tags || []).map((t) => String(t).replace(/[<>#]/g, '').trim()).filter((t) => t && !JUNK_HASHTAGS.has(t.toLowerCase().replace(/\s+/g, ''))), ...sc.hashtags])).slice(0, 18);
}

/** The category section of the final description (only verified material). */
function descriptionDetails(sc: Script): string {
  const recipe = sc.factPack?.recipe || '';
  if (CFG.category === 'cooking' && recipe) {
    const lines = recipe.split('\n');
    const ing = lines.filter((l) => l.startsWith('- ')).slice(0, 20).map((l) => `• ${l.slice(2)}`);
    const steps = lines.filter((l) => /^\d+\.\s/.test(l)).slice(0, 12).map((l) => l.length > 170 ? `${l.slice(0, 167).trimEnd()}…` : l);
    const tip = lines.find((l) => l.startsWith('TIP: '));
    const safety = lines.find((l) => l.startsWith('SAFETY: '));
    return [ing.length ? `🛒 INGREDIENTS\n${ing.join('\n')}` : '', steps.length ? `👩‍🍳 METHOD\n${steps.join('\n')}` : '', tip ? `💡 ${tip}` : '', safety ? `⚠️ ${safety}` : ''].filter(Boolean).join('\n\n');
  }
  if (CFG.category === 'stories' || sc.usedFallbackTemplate) return '';
  // News / tech / ads: the key points exactly as narrated (the narration passed the fact check).
  const seen = new Set<string>();
  const points = sc.scenes.map((x) => firstSentence(x.narration)).filter((p) => {
    const k = String(p ?? '').toLowerCase().slice(0, 40);
    if (p.split(' ').length < 5 || seen.has(k) || /\b(subscribe|follow|comment|like this video)\b/i.test(p)) return false;
    seen.add(k); return true;
  }).slice(0, 6);
  const head = CFG.category === 'news' ? '📌 KEY POINTS' : CFG.category === 'tech' ? '📌 WHAT YOU WILL LEARN' : '📌 HIGHLIGHTS';
  return points.length >= 2 ? `${head}\n${points.map((p) => `• ${p}`).join('\n')}` : '';
}

/** Only a plain https homepage/product URL is kept (never a search, news aggregator or social page). */
function cleanOfficialUrl(v: any): string {
  const raw = String(v || '').trim();
  if (!raw) return '';
  try {
    const u = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`);
    if (!/^https?:$/.test(u.protocol) || !u.hostname.includes('.')) return '';
    if (/(^|\.)(google|bing|news\.google|youtube|youtu|facebook|twitter|x|instagram|tiktok|reddit|t)\.(com|co|be)$/i.test(u.hostname)) return '';
    return u.toString().slice(0, 300);
  } catch { return ''; }
}

function normaliseScript(parsed: any, model: string, relaxed = false): Script {
  const L = { ...lengthSpec() };
  if (relaxed) { L.minScenes = Math.ceil(L.minScenes * 0.6); L.minWords = Math.ceil(L.minWords * 0.65); }
  const scenes: Scene[] = (Array.isArray(parsed.scenes) ? parsed.scenes : [])
    .map((s: any) => {
      const tagged = parseTaggedNarration(String(s?.narration || s?.text || ''));
      return {
        narration: tagged.text,
        cues: tagged.cues,
        shot: (['scene', 'panel', 'full'].includes(String(s?.shot)) ? s.shot : 'scene') as Scene['shot'],
        emotion: String(s?.emotion || 'neutral').toLowerCase().slice(0, 20),
        imagePrompt: String(s?.imagePrompt || s?.visual || '').trim().slice(0, 400),
        searchQuery: String(s?.searchQuery || '').replace(/[^\w\s'-]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80),
        productShot: s?.productShot === true || s?.product === true
      };
    })
    .filter((s: Scene) => s.narration.split(' ').length >= 3);
  const words = scenes.reduce((n, s) => n + s.narration.split(' ').length, 0);
  if (scenes.length < L.minScenes) throw new Error(`only ${scenes.length} scenes`);
  if (words < L.minWords) throw new Error(`script too short (${words} words)`);
  // Trim an over-long script at a scene boundary (keeps timing within the format).
  while (scenes.length > L.minScenes && scenes.reduce((n, s) => n + s.narration.split(' ').length, 0) > L.maxWords) scenes.splice(scenes.length - 2, 1);
  if (scenes.length > L.maxScenes) scenes.splice(L.maxScenes - 1, scenes.length - L.maxScenes);
  let title = String(parsed.title || '').replace(/[#"]/g, '').replace(/\s+/g, ' ').trim();
  if (!title) title = scenes[0].narration.slice(0, 60);
  if (CFG.category === 'stories') {
    title = title.replace(/\s*\(part \d+\)\s*$/i, '');
    if (CFG.partNumber < CFG.arcParts) title = `${title.slice(0, 58)} (Part ${CFG.partNumber})`;
    else if (CFG.arcParts > 1) title = `${title.slice(0, 52)} (Finale)`;
  }
  const hashtags: string[] = Array.from(new Set<string>((Array.isArray(parsed.hashtags) ? parsed.hashtags : []).map(cleanHashtag).filter((h: string) => h.length >= 3 && !JUNK_HASHTAGS.has(h)))).slice(0, 10);
  const tags = Array.from(new Set([...(Array.isArray(parsed.tags) ? parsed.tags : []).map((t: any) => String(t).replace(/[<>#]/g, '').trim()).filter(Boolean), ...hashtags])).slice(0, 18);
  return {
    title: title.slice(0, 95),
    description: cleanDescriptionText(String(parsed.description || '')).slice(0, 2200),
    hashtags,
    tags,
    visualStyle: String(parsed.visualStyle || '').slice(0, 200),
    characters: String(parsed.characters || '').slice(0, 300),
    scenes,
    usedFallbackTemplate: false,
    model,
    sourceHeadline: String(parsed.sourceHeadline || '').slice(0, 300),
    premise: String(parsed.premise || '').replace(/\s+/g, ' ').slice(0, 900),
    // A service ad has no website to send people to.
    officialUrl: CFG.category === 'ads' && CFG.adProfile.service ? '' : cleanOfficialUrl(parsed.officialUrl)
  };
}

/** Research context shared by the fact research, the fact check and the image check. */
let RESEARCH_CTX: ResearchCtx | null = null;
const researchCtx = (pastTitles: string[] = []): ResearchCtx => (RESEARCH_CTX ||= {
  llm: LLM, log, offline: CFG.offline, category: CFG.category, subGenre: CFG.subGenre, topic: CFG.topic, pastTitles, adBrief: CFG.adBrief
});

/** Research BEFORE writing: every non-story video starts from verified facts. */
async function researchFor(pastTitles: string[], headlines: any[]): Promise<FactPack | null> {
  const ctx = researchCtx(pastTitles);
  if (CFG.category === 'news' || CFG.category === 'tech') {
    const pack = await researchNews(ctx, headlines);
    if (!pack && !CFG.offline) {
      throw new PipelineError('research_failed', `None of the fresh ${CFG.category === 'tech' ? 'tech' : 'news'} articles could be read right now (nothing made-up is ever posted). The next attempt runs automatically.`);
    }
    return pack;
  }
  if (CFG.category === 'cooking') return researchRecipe(ctx);
  if (CFG.category === 'ads' && CFG.adBrief) return { subject: 'the advertised product', facts: [], excerpts: [{ source: 'advertiser PDF', url: '', text: CFG.adBrief.slice(0, 6000) }], outlets: ['advertiser document'], links: [], confirmed: true };
  return null;
}

/** "[serious] Heavy rain…" — the scene's narration with its performance tags put back in. */
function taggedOf(s: Scene): string {
  const toks = s.narration.split(/\s+/).filter(Boolean);
  const out: string[] = [];
  toks.forEach((t, i) => { for (const c of s.cues) if (c.index === i) out.push(`[${c.tag}]`); out.push(t); });
  for (const c of s.cues) if (c.index >= toks.length) out.push(`[${c.tag}]`);
  return out.join(' ');
}

/**
 * Keeps the script true to its sources WITHOUT another AI call (so it can never
 * fail because a model is busy): every number the script states — amounts,
 * prices, dates, scores, specs, times, temperatures — must appear in the
 * researched source material. A sentence with a number the sources don't
 * contain is cut. Returns false when too little of the script is left.
 */
const SERVICE_WRONG_CTA = /\b(download|install|app store|play store|google play|get the app|visit (our|the) (web)?site|link in (the )?(bio|description)|sign up (at|on)|www\.|https?:\/\/|\.com\b)/i;
function groundScript(script: Script, pack: FactPack | null): boolean {
  if (CFG.category === 'stories' || script.usedFallbackTemplate || !pack) return true;
  const source = [factSheet(pack), pack.recipe || '', ...pack.excerpts.map((e) => e.text)].join('\n');
  const strict = CFG.category === 'cooking';
  let cut = 0;
  const kept: Scene[] = [];
  for (const sc of script.scenes) {
    const sentences = taggedOf(sc).split(/(?<=[.!?])\s+(?=\S)/);
    const good = sentences.filter((t) => {
      // A service ad never tells people to download an app or visit a website.
      if (CFG.category === 'ads' && CFG.adProfile.service && SERVICE_WRONG_CTA.test(CFG.adProfile.contact ? t.split(CFG.adProfile.contact).join(' ') : t)) { cut++; log(`Service ad: cut "${t.replace(/\[[a-z_]+\]\s*/g, '').slice(0, 90)}" — there is no app or website to get; people message the company.`); return false; }
      const bad = unsupportedNumbers(source, t.replace(/\[[a-z_]+\]/g, ''), strict);
      if (bad.length) { cut++; log(`Grounding: cut "${t.replace(/\[[a-z_]+\]\s*/g, '').slice(0, 90)}" — ${bad.join(', ')} is not in the sources.`); }
      return !bad.length;
    });
    const t = parseTaggedNarration(good.join(' '));
    if (t.text.split(' ').length >= 3) kept.push({ ...sc, narration: t.text, cues: t.cues });
  }
  const L = lengthSpec();
  if (kept.length < Math.ceil(L.minScenes * 0.6)) {
    log(`Grounding: ${cut} sentence(s) had numbers the sources don't contain and too little was left — asking for a new script.`);
    return false;
  }
  // Title/description: drop claims with unsupported numbers too.
  if (unsupportedNumbers(source, script.title, strict).length) {
    // A title with a number the sources don't state: keep the clean parts, else use the verified headline.
    const parts = script.title.split(/\s+[-–:|]\s+/);
    const clean = parts.filter((x) => !unsupportedNumbers(source, x, strict).length);
    const was = script.title;
    script.title = (clean.length && clean[0] === parts[0] ? clean.join(' - ') : String(pack.subject || '').replace(/\s+[-–|]\s+[^-–|]+$/, '')).slice(0, 90) || was;
    log(`Grounding: title "${was}" had a number the sources don't contain — now "${script.title}".`);
  }
  script.description = script.description.split(/(?<=[.!?])\s+/).filter((t) => !unsupportedNumbers(source, t, strict).length).join(' ');
  script.scenes = kept;
  script.factChecked = true;
  log(`Grounding: every number in the script matches the sources ✔${cut ? ` (${cut} unsupported sentence(s) removed)` : ''}${pack.singleSource ? ` — single source: ${pack.singleSource}` : ''}.`);
  return true;
}

async function generateScript(pastStory: string, pastTitles: string[], pastSources: string[]): Promise<Script> {
  const headlines = (CFG.category === 'tech' || CFG.category === 'news') ? await recentHeadlines(pastTitles, pastSources) : [];
  const pack = CFG.category === 'stories' ? null : await researchFor(pastTitles, headlines);
  const prompt = buildPrompt(pastStory, pastTitles, pack?.headline ? [pack.headline] : headlines, pack);
  const system = 'You are an award-winning short-form video writer and director. Your videos open with an irresistible hook, make complete sense, stay engaging every single second and end with a reason to follow. You answer with one valid JSON object and nothing else.';
  let nearMiss: { script: Script; words: number } | null = null;
  let lastError = '';
  const withSources = (sc: Script) => {
    sc.factPack = pack || undefined;
    if (pack?.headline) {
      // The verified story is what the video is about; cite every outlet that confirmed it.
      sc.sourceHeadline = pack.headline.title;
      sc.sources = [`${pack.headline.title} (${pack.outlets.slice(0, 4).join(', ') || pack.headline.source})`];
      sc.sourceStory = { title: pack.headline.title, source: pack.headline.source, link: pack.links[0]?.url || pack.headline.link };
      if (!sc.officialUrl && pack.officialUrl) sc.officialUrl = cleanOfficialUrl(pack.officialUrl);
      sc.sourceLinks = pack.links.map((l) => l.url).filter(Boolean).slice(0, 4);
      return sc;
    }
    if (pack?.recipe) { sc.sources = [`Recipe checked against: ${pack.outlets.slice(0, 4).join(', ') || 'published recipes'}`]; return sc; }
    const used = headlines.find((h) => sc.sourceHeadline && sameStory(h.title, sc.sourceHeadline)) || (headlines.length ? headlines.find((h) => sc.scenes.some((x) => sameStory(h.title, x.narration))) : null);
    // Cite only what the video is really about — never a neighbouring headline.
    sc.sources = used ? [`${used.title} (${used.source})`] : sc.sourceHeadline ? [sc.sourceHeadline] : [];
    // The photos must be of the story the script is actually about: the matched
    // headline, else the headline the writer named — never an unrelated one.
    if (used) sc.sourceStory = { title: used.title, source: used.source, link: used.link };
    else if (sc.sourceHeadline) sc.sourceStory = { title: sc.sourceHeadline, source: '', link: '' };
    return sc;
  };
  if (!CFG.offline && !LLM.hasKeys) throw new PipelineError('script_failed', 'The AI model on the runner did not start.');
  if (!CFG.offline) {
    log(`Script writer: the self-hosted model on this runner (${CFG.localLlmUrl}).`);
    const t0 = Date.now();
    for await (const a of LLM.attempts({
      system,
      user: prompt,
      saferUser: saferScriptPrompt(prompt),
      temperature: CFG.category === 'stories' ? 0.95 : 0.7,
      maxTokens: IS_SHORTS ? 6000 : 10000,
      task: 'script',
      json: true,
      timeoutMs: 100000
    })) {
      const label = `${a.provider}/${a.model}`;
      try {
        let parsed = extractJson(a.text);
        let script: Script;
        // A small local model often writes too little: hand it its own script back and ask it
        // to extend it to the target length (up to two rounds) before giving up on the answer.
        for (let round = 0; round < 2; round++) {
          try { normaliseScript(parsed, label); break; } catch (v: any) {
            if (!/too short|only \d+ scenes/.test(String(v?.message))) break;
            const L = lengthSpec();
            const have = (Array.isArray(parsed?.scenes) ? parsed.scenes : []).reduce((n: number, sc: any) => n + String(sc?.narration || '').split(/\s+/).filter(Boolean).length, 0);
            log(`${label}: the script has ${have} words — asking it to extend it to ${L.words} words (${L.scenes} scenes).`);
            const more = await expandScript(system, prompt, parsed, have);
            if (!more) break;
            parsed = more;
          }
        }
        try {
          script = normaliseScript(parsed, label);
        } catch (validation: any) {
          try {
            const near = normaliseScript(parsed, label, true);
            const words = near.scenes.reduce((n, x) => n + x.narration.split(' ').length, 0);
            if (!nearMiss || words > nearMiss.words) nearMiss = { script: near, words };
          } catch {}
          throw validation;
        }
        withSources(script);
        if (!groundScript(script, pack)) throw new Error('too much of the script was not backed by the sources');
        log(`Script by ${label} in ${((Date.now() - t0) / 1000).toFixed(1)}s: "${script.title}" — ${script.scenes.length} scenes, ${script.scenes.reduce((n, x) => n + x.narration.split(' ').length, 0)} words, ${script.scenes.reduce((n, x) => n + x.cues.length, 0)} performance cues.`);
        return script;
      } catch (err: any) {
        lastError = `${label}: ${err?.message}`;
        log(`${label} answered but the script was unusable (${err?.message}) — asking the next model.`);
      }
    }
    if (!lastError) lastError = LLM.lastErrors.slice(-3).join(' | ') || 'no model answered';
  } else {
    lastError = 'offline test mode';
  }
  if (nearMiss) {
    log(`Using the best AI script (${nearMiss.words} words — a little shorter than asked) from ${nearMiss.script.model}.`);
    const sc = withSources(nearMiss.script);
    if (groundScript(sc, pack)) return sc;
  }
  log(`⚠️ AI script generation failed (${lastError}).`);
  return { ...templateScript(), aiError: lastError };
}

/** Ask the local model to lengthen a script it wrote (same JSON shape, more scenes and narration). */
async function expandScript(system: string, prompt: string, parsed: any, have: number): Promise<any | null> {
  const L = lengthSpec();
  const user = `${prompt}\n\nYOUR DRAFT (JSON) IS TOO SHORT: it has ${have} words of narration, but the video needs ${L.words} words in ${L.scenes} scenes (${L.seconds}).\nRewrite it LONGER: keep the title, style and every good line, deepen each beat, and add new scenes (more detail, examples, context, a stronger ending). Every scene's "narration" should be 1-3 full sentences.\nReturn the COMPLETE script as ONE JSON object with exactly the same fields as the draft.\n\nDRAFT:\n${JSON.stringify(parsed).slice(0, 12000)}`;
  for await (const a of LLM.attempts({ system, user, temperature: 0.7, maxTokens: IS_SHORTS ? 6000 : 10000, json: true, timeoutMs: 100000, task: 'script_expand' })) {
    try {
      const j = extractJson(a.text);
      const words = (Array.isArray(j?.scenes) ? j.scenes : []).reduce((n: number, sc: any) => n + String(sc?.narration || '').split(/\s+/).filter(Boolean).length, 0);
      if (words > have) return j;
    } catch {}
  }
  return null;
}

function templateScript(): Script {
  const s = (narration: string, shot: Scene['shot'], emotion: string, imagePrompt: string, searchQuery: string): Scene => {
    const tagged = parseTaggedNarration(narration);
    return { narration: tagged.text, cues: tagged.cues, shot, emotion, imagePrompt, searchQuery };
  };
  const nora = 'Nora, a woman in her thirties with short dark hair and a yellow raincoat';
  return {
    title: `The Lighthouse Signal (Part ${CFG.partNumber})`,
    description: 'An episodic mystery told in parts. What would you do next?',
    hashtags: ['scarystories', 'mystery', 'storytime'],
    tags: ['scary story', 'mystery story', 'lighthouse'],
    visualStyle: 'dark cinematic film still, cold blue shadows, 35mm, moody lighting',
    characters: nora,
    usedFallbackTemplate: true,
    scenes: [
      s('[serious] For seventy years, nobody had kept the lighthouse on Blackwood Point. [surprised] Then one night, its light came on.', 'scene', 'tense', 'an old stone lighthouse on a cliff at night, its lamp glowing blue through thick fog', 'lighthouse fog night'),
      s('[worried] This is the story of Nora, [look_left] the only person in town who went up to look.', 'scene', 'tense', `${nora} walking up a foggy cliff path at night with a flashlight`, 'foggy cliff path night'),
      s('[serious] The rusted door was already open. [lean_in] Inside, the air smelled of salt and old stone.', 'scene', 'scared', 'a rusted iron door hanging open at the base of a lighthouse, darkness inside', 'old rusted door dark'),
      s('[worried] On the spiral stairs, [point] Nora found footprints. Fresh. Still wet. [scared] Going up.', 'panel', 'scared', 'wet footprints on old stone spiral stairs lit by a flashlight beam', 'spiral staircase stone'),
      s('[scared] Every step she climbed echoed twice, [look_up] as if someone above her was climbing too.', 'scene', 'scared', `${nora} looking up a narrow spiral staircase into darkness, flashlight beam`, 'spiral staircase looking up'),
      s('[surprised] At the top, the great glass lens was turning on its own, [hands_up] humming like it was alive.', 'full', 'shocked', 'a huge glowing lighthouse lens turning in a dark lantern room', 'lighthouse lens'),
      s('[serious] Scratched into the glass, in fresh sharp letters, was that night\'s date. [scared] And Nora\'s name.', 'full', 'shocked', 'letters scratched into glass, close up, eerie blue light', 'scratched glass close up'),
      s('[worried] Someone knew she would come. [calm] Part two is next. [curious] Would you have climbed those stairs? [wave]', 'scene', 'tense', `${nora} frozen in a dark lantern room, blue light on her face`, 'woman dark room blue light')
    ]
  };
}

// ---------------------------------------------------------------------------
// 2. Narration with word timings
// ---------------------------------------------------------------------------
interface Word { text: string; start: number; end: number; token?: number; line?: number }
interface Narration { audioPath: string; duration: number; words: Word[]; wordsReliable: boolean; engine: string; neural: boolean }

const VOICES: Record<string, Record<string, string[]>> = {
  female: {
    stories: ['en-US-AvaMultilingualNeural', 'en-US-EmmaMultilingualNeural', 'en-US-AriaNeural'],
    news: ['en-US-EmmaMultilingualNeural', 'en-US-AriaNeural', 'en-US-JennyNeural'],
    default: ['en-US-AvaMultilingualNeural', 'en-US-JennyNeural', 'en-US-AriaNeural']
  },
  male: {
    stories: ['en-US-AndrewMultilingualNeural', 'en-US-ChristopherNeural', 'en-US-GuyNeural'],
    news: ['en-US-BrianMultilingualNeural', 'en-US-GuyNeural', 'en-US-ChristopherNeural'],
    default: ['en-US-AndrewMultilingualNeural', 'en-US-BrianMultilingualNeural', 'en-US-GuyNeural']
  }
};
const RATE: Record<string, string> = { stories: '-3%', cooking: '+4%', tech: '+5%', news: '+5%', ads: '+4%', musical: '0%' };

function cleanForSpeech(text: string): string {
  return text.replace(/[*_#`>~]/g, ' ').replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, ' ').replace(/\s+/g, ' ').trim();
}

/** Attach the script's own tokens (with punctuation) to the TTS word timings. */
function alignWords(boundaries: Word[], script: string): Word[] {
  const tokens = script.split(/\s+/).filter(Boolean);
  // Letters of every alphabet count (Korean, Arabic, Cyrillic…), not just a–z.
  const norm = (s: string) => String(s ?? '').toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]/gu, '');
  const out: Word[] = [];
  let ti = 0;
  for (const b of boundaries) {
    const nb = norm(b.text);
    if (!nb) continue;
    let found = -1;
    for (let k = ti; k < Math.min(tokens.length, ti + 6); k++) {
      const nt = norm(tokens[k]);
      if (nt && (nt === nb || nt.startsWith(nb) || nb.startsWith(nt))) { found = k; break; }
    }
    if (found >= 0) {
      // A token already consumed (e.g. "1/2" spoken as several words): extend the last word instead.
      out.push({ text: tokens[found], start: b.start, end: b.end, token: found });
      ti = found + 1;
    } else if (out.length && out[out.length - 1].token === ti - 1) {
      out[out.length - 1].end = b.end;
    } else {
      out.push({ text: b.text, start: b.start, end: b.end, token: -1 });
    }
  }
  return out;
}

let LAST_TTS_ERROR = '';

async function synthesizeNarration(script: string): Promise<Narration> {
  const text = cleanForSpeech(script);
  const textFile = path.join(WORK_DIR, 'script.txt');
  fs.writeFileSync(textFile, text);
  const python = ENV.PYTHON || 'python3';
  const voices = (VOICES[CFG.gender][CFG.category] || VOICES[CFG.gender].default);

  let lastTtsError = '';
  if (!CFG.offline) {
    // Each voice at the category's pace, then at normal pace; short pause between
    // attempts so a transient Edge TTS hiccup does not cost the run.
    const attempts: { voice: string; rate: string }[] = [];
    for (const voice of voices) attempts.push({ voice, rate: RATE[CFG.category] || '+0%' });
    attempts.push({ voice: voices[0], rate: '+0%' }, { voice: voices[1] || voices[0], rate: '+0%' });
    for (let a = 0; a < attempts.length; a++) {
      const { voice, rate } = attempts[a];
      if (a > 0) await sleep(Math.min(8000, 2000 * a));
      const mp3 = path.join(WORK_DIR, 'narration.mp3');
      const wordsFile = path.join(WORK_DIR, 'words.json');
      for (const f of [mp3, wordsFile]) { try { fs.unlinkSync(f); } catch {} }
      // "--rate=-3%" (with "="): a value starting with "-" would otherwise be read as a new option.
      const r = await run(python, [path.join(HERE, 'tts.py'), `--text-file=${textFile}`, `--voice=${voice}`, `--rate=${rate}`, `--out-audio=${mp3}`, `--out-words=${wordsFile}`], { timeoutMs: 180000 });
      const duration = fs.existsSync(mp3) ? await probeDuration(mp3) : 0;
      if (r.code === 0 && duration > 2) {
        let words: Word[] = [];
        let reliable = false;
        try {
          const data = JSON.parse(fs.readFileSync(wordsFile, 'utf8'));
          if (Array.isArray(data.words) && data.words.length > 5) {
            words = alignWords(data.words, text);
            reliable = true;
          }
        } catch {}
        if (!reliable) words = estimateWordTimes(text, duration);
        log(`Narration: ${voice}, ${duration.toFixed(1)}s, ${words.length} timed words${reliable ? '' : ' (estimated)'}.`);
        return { audioPath: mp3, duration, words, wordsReliable: reliable, engine: `edge-tts:${voice}`, neural: true };
      }
      lastTtsError = r.stderr.trim().split('\n').filter(Boolean).slice(-2).join(' | ') || `exit ${r.code}`;
      log(`edge-tts with ${voice} (${rate}) failed (exit ${r.code}): ${lastTtsError}`);
    }
  }
  LAST_TTS_ERROR = lastTtsError;

  // Fallback that runs entirely on this runner: Piper neural voices (open source, no service).
  try {
    const piper = await piperNarration(textFile);
    if (piper) {
      log(`Narration: Piper ${piper.voice} on the runner, ${piper.duration.toFixed(1)}s (word times estimated).`);
      return { audioPath: piper.file, duration: piper.duration, words: estimateWordTimes(text, piper.duration), wordsReliable: false, engine: `piper:${piper.voice}`, neural: true };
    }
  } catch (e: any) { log(`Piper voice unavailable (${String(e?.message || e).slice(0, 200)}).`); }

  // Last resort: ffmpeg's built-in flite voice.
  const wav = path.join(WORK_DIR, 'narration_flite.wav');
  const voice = CFG.gender === 'male' ? 'kal16' : 'slt';
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `flite=textfile=${textFile}:voice=${voice}`, '-ar', '44100', '-ac', '1', wav], { timeoutMs: 120000 });
  const duration = fs.existsSync(wav) ? await probeDuration(wav) : 0;
  if (r.code !== 0 || duration < 1) throw new PipelineError('tts_failed', `Voice synthesis failed with every engine. ${r.stderr.slice(-300)}`);
  log(`⚠️ Neural voice unavailable — used the offline flite voice (${duration.toFixed(1)}s).`);
  return { audioPath: wav, duration, words: estimateWordTimes(text, duration), wordsReliable: false, engine: `flite:${voice}`, neural: false };
}

/** Piper (rhasspy) neural text-to-speech, installed on the runner on first use. */
async function piperNarration(textFile: string): Promise<{ file: string; duration: number; voice: string } | null> {
  const root = path.join(os.homedir(), 'animato-piper'), venv = path.join(root, 'venv'), py = path.join(venv, 'bin', 'python');
  const voice = CFG.gender === 'male' ? 'en_US-ryan-medium' : 'en_US-hfc_female-medium';
  const vpath = CFG.gender === 'male' ? 'ryan/medium' : 'hfc_female/medium';
  const model = path.join(root, `${voice}.onnx`);
  fs.mkdirSync(root, { recursive: true });
  if (!fs.existsSync(py) || (await run(py, ['-c', 'import piper'], { timeoutMs: 60000 })).code !== 0) {
    if ((await run('python3', ['-m', 'venv', venv], { timeoutMs: 120000 })).code !== 0) return null;
    const r = await run(py, ['-m', 'pip', 'install', '-q', '--disable-pip-version-check', 'piper-tts'], { timeoutMs: 600000 });
    if (r.code !== 0) throw new Error(`piper install failed: ${r.stderr.slice(-200)}`);
  }
  for (const ext of ['', '.json']) {
    const f = model + ext;
    if (!fs.existsSync(f) || fs.statSync(f).size < 100) {
      const ok = await download(`https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${vpath}/${voice}.onnx${ext}`, f, 180000);
      if (!ok) throw new Error(`could not download the ${voice} voice`);
    }
  }
  const wav = path.join(WORK_DIR, 'narration_piper.wav');
  const r = await new Promise<{ code: number; err: string }>((resolve) => {
    const p = spawn(py, ['-m', 'piper', '-m', model, '-f', wav], { stdio: ['pipe', 'ignore', 'pipe'] });
    let err = '';
    p.stderr.on('data', (d) => { err = (err + d).slice(-2000); });
    p.on('close', (code) => resolve({ code: code ?? -1, err }));
    p.on('error', (e) => resolve({ code: -1, err: String(e) }));
    p.stdin.end(fs.readFileSync(textFile, 'utf8'));
  });
  const duration = fs.existsSync(wav) ? await probeDuration(wav) : 0;
  if (r.code !== 0 || duration < 1) throw new Error(`piper failed: ${r.err.slice(-200)}`);
  return { file: wav, duration, voice };
}

function estimateWordTimes(text: string, duration: number): Word[] {
  const tokens = text.split(/\s+/).filter(Boolean);
  const weight = (t: string) => Math.max(1, t.replace(/[^a-z]/gi, '').length) + (/[.!?]$/.test(t) ? 5 : /[,;:]$/.test(t) ? 2 : 0);
  const total = tokens.reduce((n, t) => n + weight(t), 0) || 1;
  const span = Math.max(0.5, duration - 0.3);
  let t = 0.15;
  return tokens.map((tok, i) => {
    const d = (weight(tok) / total) * span;
    const w = { text: tok, start: t, end: t + d * 0.85, token: i };
    t += d;
    return w;
  });
}

/** Scene start/end times from where each scene's first word is actually spoken. */
function timeScenes(scenes: Scene[], words: Word[], duration: number): { start: number; end: number }[] {
  const counts = scenes.map((s) => cleanForSpeech(s.narration).split(/\s+/).filter(Boolean).length);
  const starts: number[] = [];
  let tokenStart = 0;
  for (let i = 0; i < scenes.length; i++) {
    const w = words.find((x) => (x.token ?? -1) >= tokenStart);
    starts.push(i === 0 ? 0 : Math.max(0, (w ? w.start : duration * (tokenStart / Math.max(1, counts.reduce((a, b) => a + b, 0)))) - 0.12));
    tokenStart += counts[i];
  }
  const total = duration;
  return scenes.map((_, i) => ({ start: starts[i], end: i + 1 < scenes.length ? starts[i + 1] : total }));
}

// ---------------------------------------------------------------------------
// 3. Images — one per scene
// ---------------------------------------------------------------------------
async function download(url: string, file: string, timeoutMs = 45000, headers: Record<string, string> = {}): Promise<boolean> {
  try {
    if (url.startsWith('data:')) {
      const comma = url.indexOf(',');
      fs.writeFileSync(file, url.slice(0, comma).includes('base64') ? Buffer.from(url.slice(comma + 1), 'base64') : Buffer.from(decodeURIComponent(url.slice(comma + 1))));
    } else {
      const res = await fetch(url, { headers: { 'User-Agent': 'AnimatoAutoPoster/3.0 (+https://github.com)', ...headers }, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) return false;
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 4000) return false;
      fs.writeFileSync(file, buf);
    }
    const [w, h] = await imageSize(file);
    return w >= 320 && h >= 320;
  } catch {
    return false;
  }
}

/** Normalise any downloaded image to a JPEG the stage can decode quickly. */
async function toJpeg(src: string, dst: string, cropBottom = 0): Promise<boolean> {
  const vf = [cropBottom > 0 ? `crop=iw:ih*${(1 - cropBottom).toFixed(3)}:0:0` : '', "scale='min(1920,iw)':-2"].filter(Boolean).join(',');
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-frames:v', '1', '-vf', vf, '-q:v', '3', dst], { timeoutMs: 60000 });
  return r.code === 0 && fs.existsSync(dst);
}

// Ads on a Shorts turn still render their creative at 16:9 (see adLandscapeShort in
// the main pipeline) and only get framed into the vertical Short afterwards, so the
// images fetched/generated for them should be landscape too, not portrait.
const orientation = (CFG.category === 'ads' && IS_SHORTS) ? 'landscape' : W > H * 1.2 ? 'landscape' : H > W * 1.2 ? 'portrait' : 'square';
// ---------------------------------------------------------------------------
// AI pictures (story scenes, plus cooking / news fallbacks) are drawn by a local
// CPU image generator that runs on this runner: animato-cloud/imagegen.py
// (1-step SDXS model + detail upscaler). No API keys, no rate limits, no web
// service. The model is loaded ONCE per run and every scene reuses it.
// ---------------------------------------------------------------------------
const IMAGEGEN_PORT = 8012;
const IMAGEGEN_BASE = `http://127.0.0.1:${IMAGEGEN_PORT}`;
let imagegenReady: Promise<boolean> | null = null;
let imagegenDisabled = '';
let imagegenCount = 0;
let imagegenStreak = 0;
let imagegenProc: ReturnType<typeof spawn> | null = null;

/** Short style prefixes. The text encoder reads only ~77 tokens, so style stays tiny and the scene gets the rest. */
const FOOD_STYLE = 'professional food photography, appetizing, natural window light, sharp focus';
const EDITORIAL_STYLE = 'editorial illustration, realistic, cinematic lighting, wide view';

/** Drop style/boilerplate words the style prefix already says, so the scene keeps its token budget. */
function slimPrompt(p: string, maxChars: number): string {
  return p
    .replace(/\b(high[- ]end|3d animated|animated|feature[- ]film|family[- ]film|still from|render(ed)?|cinematic|highly detailed|ultra[- ]detailed|8k|4k|masterpiece|wholesome|painterly|saturated|no text|no watermark|no captions|no logos?)\b/gi, ' ')
    .replace(/[\r\n]+/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .replace(/([,.;:!?])(?:\s*[,.;:!?])+/g, '$1')
    .replace(/^[\s,.;:!?]+|[\s,.;:!?]+$/g, '')
    .slice(0, maxChars);
}

/** Starts the local generator (once) and resolves true when the model is loaded and ready. */
function startImageGenerator(): Promise<boolean> {
  if (imagegenReady) return imagegenReady;
  imagegenReady = (async () => {
    if (CFG.offline) return false;
    const script = path.join(HERE, 'imagegen.py');
    if (!fs.existsSync(script)) { imagegenDisabled = 'imagegen.py is missing'; return false; }
    const py = ENV.IMAGEGEN_PYTHON || 'python3';
    const t0 = Date.now();
    log('🖼️ Starting the local image generator (model loads while the script is written)…');
    const proc = spawn(py, [script, 'serve', `--port=${IMAGEGEN_PORT}`], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUNBUFFERED: '1' } });
    imagegenProc = proc;
    let tail = '';
    let exited = false;
    const onData = (d: Buffer) => {
      const text = d.toString();
      tail = (tail + text).slice(-3000);
      for (const line of text.split('\n')) if (line.startsWith('[imagegen]')) log(`🖼️ ${line.slice(10).trim()}`);
    };
    proc.stdout?.on('data', onData);
    proc.stderr?.on('data', onData);
    proc.on('error', (e) => { exited = true; imagegenDisabled = `could not start (${e.message})`; });
    proc.on('close', () => { exited = true; if (!imagegenDisabled) imagegenDisabled = 'the generator stopped'; });
    process.on('exit', () => { try { proc.kill('SIGKILL'); } catch {} });
    const until = Date.now() + 10 * 60000; // the very first run also downloads the model
    while (Date.now() < until) {
      if (exited) {
        log(`⚠️ The image generator is not available (${imagegenDisabled || 'stopped'}): ${tail.split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 300)}`);
        return false;
      }
      try {
        const h = await fetch(`${IMAGEGEN_BASE}/health`, { signal: AbortSignal.timeout(4000) });
        const j: any = await h.json();
        if (j?.error) { imagegenDisabled = String(j.error).slice(0, 200); log(`⚠️ The image generator failed to load: ${imagegenDisabled}`); try { proc.kill('SIGKILL'); } catch {} return false; }
        if (j?.ready) { log(`🖼️ Image generator ready in ${((Date.now() - t0) / 1000).toFixed(0)}s (upscaler: ${j.upscaler || 'resize'}).`); return true; }
      } catch {}
      await sleep(1500);
    }
    imagegenDisabled = 'the model did not load within 10 minutes';
    log(`⚠️ ${imagegenDisabled}.`);
    try { proc.kill('SIGKILL'); } catch {}
    return false;
  })();
  return imagegenReady;
}

/** Draws one picture with the local generator. Returns 'ai-clean' (never watermarked) or null. */
async function aiImage(prompt: string, seed: number, file: string, style = '', mode = ''): Promise<'ai' | 'ai-clean' | null> {
  if (CFG.offline || !prompt || imagegenDisabled) return null;
  if (!(await startImageGenerator())) return null;
  const size = orientation === 'portrait' ? { w: 864, h: 1536 } : orientation === 'landscape' ? { w: 1536, h: 864 } : { w: 1152, h: 1152 };
  for (let attempt = 0; attempt < 2; attempt++) {
    if (imagegenDisabled) return null;
    try {
      const res = await fetch(`${IMAGEGEN_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.slice(0, 1200), style, mode, seed: seed % 4294967295, width: size.w, height: size.h, out: file }),
        signal: AbortSignal.timeout(240000)
      });
      const data: any = await res.json().catch(() => ({}));
      if (res.ok && data?.ok && fs.existsSync(file)) { imagegenCount++; imagegenStreak = 0; return 'ai-clean'; }
      log(`Image generator: HTTP ${res.status} ${String(data?.error || '').slice(0, 160)}`);
      if (res.status === 503) await sleep(2000);
    } catch (err: any) {
      log(`Image generator request failed (${err?.message}).`);
    }
  }
  if (++imagegenStreak >= 3) { imagegenDisabled = 'it failed 3 times in a row'; log('⚠️ Image generator disabled for the rest of this run (it failed 3 times in a row).'); }
  return null;
}

// ---------------------------------------------------------------------------
// Real images from the web — news, tech, tutorials, cooking, ads.
// Nothing here is generated: every picture is found on the web (the story's own
// news photos, the product's official site and a real screenshot of it,
// Wikipedia / Wikimedia, free photo libraries) and credited on screen.
// Only STORIES are illustrated by an image model (they are fiction).
// ---------------------------------------------------------------------------
interface WebImage {
  url: string;
  /** Page to send as Referer when downloading (Wikimedia rate-limits image requests without one). */
  referer?: string;
  /** Short on-screen credit, e.g. "Photo: Jane Doe · CC BY 4.0". */
  credit: string;
  kind: 'screenshot' | 'wiki' | 'library' | 'product' | 'advertiser';
  /** Full attribution for the description (title, author, license, source page). */
  attribution?: string;
  /** What the image is of, from its source (file title, caption, tags) — used to check it matches the scene. */
  meta?: string;
}
const BROWSER_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';
/** Logos, icons, avatars, tracking pixels, placeholders — never used as a scene picture. */
const BAD_IMAGE = /(logo|favicon|sprite|icon|avatar|placeholder|default[-_]?(image|og|share|thumb)|blank\.|spacer|pixel|1x1|badge|button|banner-ad|advert|doubleclick|gravatar|emoji|\.svg(\?|$)|\.gif(\?|$)|data:image)/i;
const MIN_REAL_W = 600;

const hostOf = (u: string) => { try { return new URL(u).hostname.replace(/^www\./, ''); } catch { return ''; } };
const htmlDecode = (v: string) => v.replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#x2F;/gi, '/').trim();
const absUrl = (u: string, base: string) => { try { return new URL(htmlDecode(u), base).toString(); } catch { return ''; } };
const plain = (html: any) => htmlDecode(String(html || '').replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ').trim();

/**
 * COPYRIGHT: only images whose license allows reuse in a monetised video with
 * cropping/zooming and no share-alike obligation are accepted — public domain,
 * CC0 and CC BY (credited). Share-alike, non-commercial, no-derivatives,
 * fair-use / non-free and trademarked files are refused.
 */
function reusableLicense(short: string, restrictions = '', forAds = false): string | null {
  const l = plain(short);
  if (!l) return null;
  if (/\b(sa|nc|nd)\b|share[- ]?alike|non-?commercial|no[- ]?deriv|fair use|non-?free|all rights reserved|copyrighted/i.test(l)) return null;
  if (/trademark/i.test(restrictions)) return null;
  if (forAds && /personality/i.test(restrictions)) return null; // a person's likeness never endorses a product
  if (/^(cc0|pdm|public domain|pd\b|pd-|no restrictions|no known copyright)/i.test(l)) return l.replace(/^pd-.*/i, 'Public domain');
  if (/^cc[- ]?by([- ][\d.]+)?$/i.test(l) || /^cc[- ]?by [\d.]+/i.test(l)) return l.toUpperCase().replace('CC-BY', 'CC BY');
  if (/^attribution$/i.test(l)) return 'CC BY';
  return null;
}
const shortCredit = (author: string, license: string) => {
  const who = plain(author).replace(/^(photo(graph)? by|by)\s+/i, '').slice(0, 26) || 'Unknown author';
  return /public domain|cc0|pdm/i.test(license) ? `Photo: ${who} · Public domain` : `Photo: ${who} · ${license}`;
};

async function fetchText(url: string, timeoutMs = 15000, maxBytes = 2_000_000): Promise<{ text: string; finalUrl: string } | null> {
  if (CFG.offline || !url) return null;
  try {
    const res = await fetch(url, { redirect: 'follow', headers: { 'User-Agent': BROWSER_UA, Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.9' }, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    return { text: buf.subarray(0, maxBytes).toString('utf8'), finalUrl: res.url || url };
  } catch { return null; }
}
async function fetchJson(url: string, headers: Record<string, string> = {}, timeoutMs = 15000): Promise<any> {
  if (CFG.offline) return null;
  try {
    const res = await fetch(url, { headers: { 'User-Agent': 'AnimatoAutoPoster/5.0 (+https://github.com; credits every image it uses)', ...headers }, signal: AbortSignal.timeout(timeoutMs) });
    return res.ok ? await res.json() : null;
  } catch { return null; }
}

/** The share / lead images of a page — used ONLY for the advertiser's own website. */
function pageImages(html: string, pageUrl: string): string[] {
  const out: string[] = [];
  const add = (u: string) => { const a = absUrl(u, pageUrl); if (a && /^https?:/i.test(a) && !BAD_IMAGE.test(a) && !out.includes(a)) out.push(a); };
  for (const m of html.slice(0, 400_000).matchAll(/<meta\b[^>]*>/gi)) {
    const tag = m[0];
    const key = (tag.match(/\b(?:property|name|itemprop)\s*=\s*["']([^"']+)["']/i) || [])[1]?.toLowerCase() || '';
    const content = (tag.match(/\bcontent\s*=\s*["']([^"']+)["']/i) || [])[1];
    if (content && /^(og:image(:secure_url|:url)?|twitter:image(:src)?|image)$/.test(key)) add(content);
  }
  return out.slice(0, 6);
}
async function advertiserImages(url: string): Promise<WebImage[]> {
  const r = await fetchText(url);
  if (!r) return [];
  const who = hostOf(r.finalUrl) || hostOf(url);
  return pageImages(r.text, r.finalUrl).slice(0, 3).map((u) => ({ url: u, credit: `Image: ${who}`, kind: 'advertiser' as const, attribution: `Product images: ${who} (the advertiser)` }));
}

/** Wikimedia Commons files with their license, author and file page (only reusable ones are kept). */
async function commonsFiles(params: string, forAds = false): Promise<WebImage[]> {
  const d = await fetchJson(`${CFG.commonsApiBase}?action=query&format=json&origin=*&${params}&prop=imageinfo&iiprop=url|size|mime|extmetadata&iiurlwidth=1280`);
  const pages = (Object.values(d?.query?.pages || {}) as any[]).sort((a, b) => (a.index || 0) - (b.index || 0));
  const out: WebImage[] = [];
  for (const p of pages) {
    const ii = p?.imageinfo?.[0];
    if (!ii || p.missing !== undefined || (ii.width || 0) < 800 || !/jpe?g|png|webp/i.test(ii.mime || ii.url || '')) continue;
    const md = ii.extmetadata || {};
    const lic = reusableLicense(md.LicenseShortName?.value || md.License?.value || '', md.Restrictions?.value || '', forAds);
    if (!lic) continue;
    const url = ii.thumburl || ii.url;
    if (!url || BAD_IMAGE.test(url)) continue;
    const author = plain(md.Artist?.value || md.Credit?.value || '');
    const title = plain(md.ObjectName?.value || String(p.title || '').replace(/^File:/, '').replace(/\.[a-z]+$/i, ''));
    const page = ii.descriptionurl || `https://commons.wikimedia.org/wiki/${encodeURIComponent(String(p.title || ''))}`;
    const licUrl = plain(md.LicenseUrl?.value || '');
    const desc = plain(md.ImageDescription?.value || '').slice(0, 300);
    out.push({ url, referer: page, credit: shortCredit(author, lic), kind: 'library', meta: `${title} ${desc} ${plain(md.Categories?.value || '').replace(/\|/g, ' ')}`,
      attribution: `"${title}" by ${author || 'unknown author'} — ${lic}${licUrl ? ` (${licUrl})` : ''} — ${page}` });
  }
  return out;
}

/** Lead image of the best-matching Wikipedia article — only when it is a freely licensed Commons file. */
async function wikiImages(query: string, forAds = false): Promise<WebImage[]> {
  if (!query) return [];
  const d = await fetchJson(`${CFG.wikiApiBase}?action=query&format=json&origin=*&generator=search&gsrsearch=${encodeURIComponent(query)}&gsrlimit=2&prop=pageimages&piprop=name`);
  const pages = (Object.values(d?.query?.pages || {}) as any[]).sort((a, b) => (a.index || 0) - (b.index || 0)).filter((p) => p?.pageimage);
  if (!pages.length) return [];
  // Non-free files (fair-use logos, posters) live on Wikipedia itself, not on Commons → they come back "missing" and are skipped.
  // The article's title counts as the image's subject (the lead image of "iPhone 16 Pro" shows the iPhone 16 Pro).
  const out: WebImage[] = [];
  for (const p of pages) {
    for (const x of await commonsFiles(`titles=${encodeURIComponent(`File:${p.pageimage}`)}`, forAds)) out.push({ ...x, kind: 'wiki', meta: `${p.title} ${x.meta || ''}` });
  }
  return out;
}

/** Freely licensed photo libraries: Wikimedia Commons, Openverse (CC0/PDM/CC BY), Pexels, Pixabay. */
async function libraryImages(query: string, forAds = false): Promise<WebImage[]> {
  if (!query) return [];
  const q = encodeURIComponent(query);
  const out: WebImage[] = [];
  if (CFG.pexelsKey) {
    const d = await fetchJson(`https://api.pexels.com/v1/search?query=${q}&per_page=4&orientation=${orientation}`, { Authorization: CFG.pexelsKey });
    for (const p of d?.photos || []) out.push({ url: orientation === 'portrait' ? p.src?.portrait || p.src?.large2x : p.src?.large2x || p.src?.large, meta: String(p.alt || ''), credit: `Photo: ${String(p.photographer || 'Pexels').slice(0, 24)} · Pexels`, kind: 'library', attribution: `Photo by ${p.photographer || 'unknown'} on Pexels (Pexels License) — ${p.url || 'https://www.pexels.com'}` });
  }
  if (CFG.pixabayKey) {
    const d = await fetchJson(`https://pixabay.com/api/?key=${CFG.pixabayKey}&q=${q}&image_type=photo&safesearch=true&per_page=4&orientation=${orientation === 'landscape' ? 'horizontal' : 'vertical'}`);
    for (const h of d?.hits || []) out.push({ url: h.largeImageURL, meta: String(h.tags || ''), credit: `Photo: ${String(h.user || 'Pixabay').slice(0, 24)} · Pixabay`, kind: 'library', attribution: `Image by ${h.user || 'unknown'} on Pixabay (Pixabay Content License) — ${h.pageURL || 'https://pixabay.com'}` });
  }
  out.push(...await commonsFiles(`generator=search&gsrnamespace=6&gsrlimit=12&gsrsearch=${q}%20filetype:bitmap`, forAds));
  // Openverse (millions of CC0 / CC BY photos, e.g. Flickr food photography) is always asked:
  // Commons alone often has nothing for everyday ingredients and cooking steps.
  {
    const o = await fetchJson(`${CFG.openverseBase}?q=${q}&page_size=12&mature=false&license=cc0,pdm,by`);
    for (const r of o?.results || []) {
      if ((r.width || 0) < 800) continue;
      const lic = reusableLicense(`${r.license === 'by' ? 'CC BY' : String(r.license || '').toUpperCase()} ${r.license_version || ''}`.trim());
      if (!lic) continue;
      out.push({ url: r.url, meta: `${plain(r.title)} ${(Array.isArray(r.tags) ? r.tags : []).map((t: any) => t?.name || '').join(' ')}`, credit: shortCredit(r.creator || '', lic), kind: 'library', attribution: `"${plain(r.title) || 'Untitled'}" by ${plain(r.creator) || 'unknown author'} — ${lic}${r.license_url ? ` (${r.license_url})` : ''} — ${r.foreign_landing_url || r.url}` });
    }
  }
  return out.filter((x) => x.url && !BAD_IMAGE.test(x.url));
}

/**
 * Error / block pages that must NEVER end up in a video ("This site can't be
 * reached", HTTP errors, bot checks, captchas, parked domains).
 */
const BROKEN_PAGE = /this site can.?t be reached|site can.?t be reached|\berr_[a-z_]{4,}|dns_probe|server.?s ip address could not be found|took too long to respond|refused to connect|just a moment\.\.\.|checking (if the site connection is secure|your browser)|attention required|verify you are (a )?human|are you a robot|enable javascript and cookies to continue|unusual traffic from your|domain (is )?for sale|buy this domain|account (has been )?suspended|welcome to nginx|default web page|apache2 (ubuntu|debian) default page/i;
/** Only trusted in the page TITLE (or on a nearly empty page): these words also appear on healthy pages. */
const BROKEN_TITLE = /access denied|forbidden|\b(400|401|403|404|410|429|500|502|503|504)\b|not found|bad gateway|service unavailable|coming soon|under construction|parked|it works!|error|captcha/i;

/** Pixel variety of an image (0 = a flat colour). Blank / half-loaded screenshots score very low. */
async function imageVariety(file: string): Promise<number> {
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-vf', 'scale=96:60,format=gray', '-f', 'rawvideo', '-'], { timeoutMs: 20000 });
  if (r.code !== 0 || !r.stdout.length) return 0;
  const px = r.stdout;
  let mean = 0; for (const v of px) mean += v; mean /= px.length;
  let varc = 0; for (const v of px) varc += (v - mean) ** 2;
  const hist = new Set<number>(); for (const v of px) hist.add(v >> 3);
  return Math.sqrt(varc / px.length) * Math.min(1, hist.size / 8);
}

/** A page that loaded, but only shows a wall instead of the article / product. */
const WALLED_PAGE = /subscribe to (continue|read|keep reading)|sign in to (continue|read|keep reading)|log ?in to (continue|read)|create (a )?(free )?account to (continue|read)|register to continue|become a (member|subscriber) to|this (article|content|story) is (for|available to) subscribers|you have reached your (article|free) limit|enable cookies to continue|turn off your ad ?blocker|please disable your ad ?blocker/i;

interface Shot {
  finalUrl: string;
  title: string;
  /** One or two verified PNG captures of the page (top of page first). */
  files: string[];
}

/**
 * A VERIFIED screenshot of a live website, driven through the Chrome DevTools
 * protocol (not a blind `--screenshot`). A capture is only returned when every
 * one of these is true, so a wrong or broken picture can never reach a video:
 *
 *   1. the main document answered 2xx/3xx and Chrome did not show an error page
 *   2. the page is not a bot-check / parked / paywall / login wall
 *   3. it really is the page we asked for (same site, or the story's own words
 *      appear on it) — never a redirect to some unrelated homepage
 *   4. cookie, consent, newsletter and app-install overlays are dismissed
 *   5. lazy-loaded images, web fonts and late network work have finished
 *   6. the capture is retina (2×) and is not blank / flat
 *
 * Anything else returns null: the scene then uses a licensed photo instead.
 */
async function siteScreenshot(url: string, file: string, opts: { expect?: string[]; second?: string } = {}): Promise<Shot | null> {
  const chrome = findChrome();
  const WS = (globalThis as any).WebSocket;
  if (!chrome || CFG.offline || !url || !WS) return null;
  const wantHost = hostOf(url);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'animato-shot-'));
  const proc = spawn(chrome, ['--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--hide-scrollbars', '--mute-audio', '--no-first-run',
    '--no-default-browser-check', '--disable-extensions', '--disable-features=IsolateOrigins,site-per-process,TranslateUI', '--autoplay-policy=user-gesture-required',
    `--user-data-dir=${profile}`, '--remote-debugging-port=0', '--window-size=1440,900', 'about:blank'], { stdio: 'ignore' });
  let ws: any = null;
  const cleanup = () => { try { ws?.close(); } catch {} try { proc.kill('SIGKILL'); } catch {} try { fs.rmSync(profile, { recursive: true, force: true }); } catch {} };
  const fail = (why: string) => { log(`Screenshot of ${wantHost || url} rejected: ${why}.`); return null; };
  try {
    // 1. Connect to Chrome.
    let port = '';
    for (let i = 0; i < 100 && !port; i++) {
      const f = path.join(profile, 'DevToolsActivePort');
      if (fs.existsSync(f)) port = fs.readFileSync(f, 'utf8').split('\n')[0].trim();
      if (!port) await sleep(100);
    }
    if (!port) return fail('Chrome did not start');
    const targets: any[] = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page');
    if (!page) return fail('no browser tab');
    ws = new WS(page.webSocketDebuggerUrl);
    await new Promise<void>((res, rej) => { ws.onopen = () => res(); ws.onerror = () => rej(new Error('devtools connection failed')); });
    let id = 0;
    const pending = new Map<number, (v: any) => void>();
    const listeners: ((m: any) => void)[] = [];
    ws.onmessage = (ev: any) => {
      const m = JSON.parse(String(ev.data));
      if (m.id && pending.has(m.id)) { pending.get(m.id)!(m); pending.delete(m.id); } else listeners.forEach((l) => l(m));
    };
    const cmd = (method: string, params: any = {}, timeoutMs = 20000) => new Promise<any>((res) => {
      const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
      setTimeout(() => { if (pending.has(i)) { pending.delete(i); res({ error: { message: `${method} timed out` } }); } }, timeoutMs);
    });
    const evaluate = async (expression: string, timeoutMs = 20000) => {
      const r = await cmd('Runtime.evaluate', { returnByValue: true, awaitPromise: true, expression }, timeoutMs);
      return r.result?.result?.value;
    };

    // 2. Navigate like a real desktop visitor and watch the main document's status.
    await cmd('Page.enable'); await cmd('Network.enable'); await cmd('Runtime.enable');
    await cmd('Network.setUserAgentOverride', { userAgent: BROWSER_UA, acceptLanguage: 'en-US,en;q=0.9' });
    // Retina metrics: text in the framed screenshot stays sharp at 1080p and 4K.
    await cmd('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 2, mobile: false });
    await cmd('Emulation.setScriptExecutionDisabled', { value: false });
    let docStatus = 0, loaded = false, inflight = 0, lastActivity = Date.now();
    listeners.push((m) => {
      if (m.method === 'Network.responseReceived' && m.params?.type === 'Document' && !docStatus) docStatus = m.params.response?.status || 0;
      if (m.method === 'Page.loadEventFired') loaded = true;
      if (m.method === 'Network.requestWillBeSent') { inflight++; lastActivity = Date.now(); }
      if (m.method === 'Network.loadingFinished' || m.method === 'Network.loadingFailed') { inflight = Math.max(0, inflight - 1); lastActivity = Date.now(); }
    });
    const nav = await cmd('Page.navigate', { url }, 30000);
    if (nav.error) return fail(nav.error.message);
    if (nav.result?.errorText) return fail(nav.result.errorText);
    for (let i = 0; i < 200 && !loaded; i++) await sleep(100);        // up to 20 s for the load event
    // Wait for the network to go quiet (lazy images, fonts, hero videos…).
    for (let i = 0; i < 120; i++) {
      if (inflight === 0 && Date.now() - lastActivity > 700) break;
      await sleep(100);
    }
    if (docStatus && (docStatus < 200 || docStatus >= 400)) return fail(`HTTP ${docStatus}`);

    // 3. Dismiss cookie / consent / newsletter / app-install overlays, then make
    //    every lazy image load by walking down the page and back to the top.
    await evaluate(`(async () => {
      const YES = /^(accept|accept all|accept all cookies|allow all|i agree|agree|got it|ok|okay|continue|understood|allow|yes, i agree|i accept|save and (accept|close)|reject all|decline|no thanks|not now|maybe later|close|dismiss|skip)$/i;
      const BAD = /cookie|consent|gdpr|onetrust|cmp|truste|privacy|banner|cc-window|qc-cmp|didomi|usercentrics|newsletter|subscribe|signup|sign-up|paywall|modal|overlay|popup|interstitial|app-?(banner|install)|promo|notification|sp_message|piano|tp-modal/i;
      const click = () => {
        const els = Array.from(document.querySelectorAll('button, a[role=button], [role=button], input[type=button], input[type=submit]')).slice(0, 400);
        for (const el of els) {
          const t = ((el.innerText || el.value || el.getAttribute('aria-label') || '') + '').trim();
          if (!t || t.length > 28 || !YES.test(t)) continue;
          const box = el.getBoundingClientRect();
          if (!box.width || !box.height) continue;
          const holder = el.closest('div,section,aside,dialog,form') || el;
          const tag = (holder.id || '') + ' ' + (typeof holder.className === 'string' ? holder.className : '');
          const fixed = ['fixed', 'sticky'].includes(getComputedStyle(holder).position);
          if (BAD.test(tag) || fixed || /cookie|consent/i.test((holder.innerText || '').slice(0, 300))) { try { el.click(); } catch (e) {} return true; }
        }
        return false;
      };
      click(); await new Promise((r) => setTimeout(r, 500)); click();
      // Anything still floating over the page and looking like an overlay: hide it.
      for (const el of Array.from(document.querySelectorAll('body *')).slice(0, 4000)) {
        const s = getComputedStyle(el);
        if (s.position !== 'fixed' && s.position !== 'sticky') continue;
        const box = el.getBoundingClientRect();
        const tag = (el.id || '') + ' ' + (typeof el.className === 'string' ? el.className : '') + ' ' + (el.getAttribute('aria-label') || '');
        const txt = (el.innerText || '').slice(0, 400);
        const covers = box.height > innerHeight * 0.55 && box.width > innerWidth * 0.55;
        if (BAD.test(tag) || covers || /\\b(cookies?|consent|subscribe|newsletter|sign up)\\b/i.test(txt)) el.style.setProperty('display', 'none', 'important');
      }
      for (const el of Array.from(document.querySelectorAll('[class*=paywall], [id*=paywall], [class*=backdrop], [class*=overlay], .modal, dialog[open]')).slice(0, 200)) {
        el.style && el.style.setProperty('display', 'none', 'important');
      }
      document.documentElement.style.setProperty('overflow', 'auto', 'important');
      document.body.style.setProperty('overflow', 'auto', 'important');
      document.body.style.removeProperty('position');
      // Wake up lazy images: walk down a few screens, then come back.
      const h = Math.min(document.body.scrollHeight, innerHeight * 5);
      for (let y = 0; y <= h; y += Math.round(innerHeight * 0.75)) { window.scrollTo(0, y); window.dispatchEvent(new Event('scroll')); await new Promise((r) => setTimeout(r, 220)); }
      window.scrollTo(0, 0); window.dispatchEvent(new Event('scroll'));
      for (const img of Array.from(document.images)) { img.loading = 'eager'; if (img.dataset && img.dataset.src && !img.src) img.src = img.dataset.src; }
      try { await document.fonts.ready; } catch (e) {}
      for (let i = 0; i < 40; i++) {
        const shown = Array.from(document.images).filter((im) => im.getBoundingClientRect().top < innerHeight * 1.2 && im.naturalWidth === 0 && im.currentSrc);
        if (!shown.length) break;
        await new Promise((r) => setTimeout(r, 100));
      }
      await new Promise((r) => setTimeout(r, 700));
      return true;
    })()`, 45000);

    // 4. Read the page that is actually on screen now and check it is the right one.
    const probe = await evaluate(`(() => {
      const text = (document.body && document.body.innerText || '').replace(/\\s+/g, ' ').trim();
      const imgs = Array.from(document.images).filter((i) => i.naturalWidth > 120 && i.getBoundingClientRect().top < innerHeight * 1.5).length;
      return { title: document.title || '', head: text.slice(0, 900), length: text.length, imgs, url: location.href,
        height: document.body ? document.body.scrollHeight : 0,
        errorPage: !!document.querySelector('#main-frame-error, .neterror, #sub-frame-error') };
    })()`);
    const info = probe;
    if (!info) return fail('page could not be read');
    if (info.errorPage || /^(chrome-error|about:)/.test(info.url)) return fail('browser error page');
    if (BROKEN_PAGE.test(`${info.title} ${info.head}`) || BROKEN_TITLE.test(info.title) || (info.length < 600 && BROKEN_TITLE.test(info.head)))
      return fail(`error/blocked page ("${String(info.title || info.head).slice(0, 60)}")`);
    if (WALLED_PAGE.test(info.head) || (WALLED_PAGE.test(`${info.title} ${info.head}`) && info.length < 1800))
      return fail('paywall / sign-in wall');
    if (info.length < 200 && info.imgs < 2) return fail('page is nearly empty');
    // Right page? Same site is enough; otherwise the story's own words must be on it.
    const gotHost = hostOf(info.url);
    const sameSite = !!gotHost && !!wantHost && (gotHost === wantHost || gotHost.endsWith(`.${wantHost}`) || wantHost.endsWith(`.${gotHost}`));
    const words = (opts.expect || []).join(' ').toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/)
      .filter((w) => w.length > 3 && !['this', 'that', 'with', 'from', 'what', 'when', 'your', 'about', 'after', 'into', 'their', 'says', 'will', 'more', 'than', 'have', 'been'].includes(w));
    const haystack = `${info.title} ${info.head} ${info.url}`.toLowerCase();
    const hits = Array.from(new Set(words)).filter((w) => haystack.includes(w));
    if (!sameSite && words.length && !hits.length)
      return fail(`redirected to ${gotHost || 'another site'}, which is not about this story`);
    if (!sameSite && !words.length) return fail(`redirected to ${gotHost || 'another site'}`);

    // 5. Capture, and check the picture itself. One retry if it comes out flat.
    const files: string[] = [];
    const grab = async (target: string): Promise<boolean> => {
      for (let attempt = 0; attempt < 2; attempt++) {
        const shot = await cmd('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false }, 30000);
        const b64 = shot.result?.data;
        if (!b64) { await sleep(1500); continue; }
        fs.writeFileSync(target, Buffer.from(b64, 'base64'));
        const variety = await imageVariety(target);
        if (variety >= 6) return true;
        try { fs.unlinkSync(target); } catch {}
        log(`Screenshot attempt ${attempt + 1} of ${gotHost} was flat (variety ${variety.toFixed(1)}) — waiting and retrying.`);
        await sleep(2500);
      }
      return false;
    };
    if (!(await grab(file))) return fail('blank / flat capture');
    files.push(file);
    // A second view further down the page, so two scenes never show the same frame.
    if (opts.second && info.height > 1400) {
      await evaluate(`(async () => {
        window.scrollTo(0, Math.min(document.body.scrollHeight - innerHeight, Math.round(innerHeight * 1.15)));
        window.dispatchEvent(new Event('scroll'));
        for (const el of Array.from(document.querySelectorAll('body *')).slice(0, 3000)) {
          const s = getComputedStyle(el);
          if (s.position === 'fixed' || s.position === 'sticky') el.style.setProperty('display', 'none', 'important');
        }
        await new Promise((r) => setTimeout(r, 900));
        return true;
      })()`, 20000);
      if (await grab(opts.second)) {
        const a = crypto.createHash('md5').update(fs.readFileSync(file)).digest('hex');
        const b = crypto.createHash('md5').update(fs.readFileSync(opts.second)).digest('hex');
        if (a === b) { try { fs.unlinkSync(opts.second); } catch {} } else files.push(opts.second);
      }
    }
    log(`Screenshot of ${gotHost} verified: HTTP ${docStatus || 'ok'}, "${String(info.title).slice(0, 60)}", ${info.length} chars, ${info.imgs} image(s)${hits.length ? `, matched "${hits.slice(0, 4).join(', ')}"` : ''}, ${files.length} view(s) at 2×.`);
    return { finalUrl: info.url, title: info.title, files };
  } catch (err: any) {
    return fail(err?.message || String(err));
  } finally {
    cleanup();
  }
}

/** The product's official website as recorded on Wikidata (property P856) — used when the script's URL doesn't work. */
async function wikidataOfficialSite(name: string): Promise<string> {
  if (!name) return '';
  const d = await fetchJson(`${CFG.wikidataApiBase}?action=wbsearchentities&format=json&origin=*&language=en&type=item&limit=3&search=${encodeURIComponent(name)}`);
  const ids = (d?.search || []).map((x: any) => x.id).filter(Boolean).slice(0, 3);
  if (!ids.length) return '';
  const e = await fetchJson(`${CFG.wikidataApiBase}?action=wbgetentities&format=json&origin=*&props=claims&ids=${ids.join('|')}`);
  for (const qid of ids) {
    const claims = e?.entities?.[qid]?.claims?.P856 || [];
    const best = claims.find((c: any) => c.rank === 'preferred') || claims[0];
    const url = cleanOfficialUrl(best?.mainsnak?.datavalue?.value);
    if (url) return url;
  }
  return '';
}

/**
 * Present a website screenshot inside a browser window with the real address in
 * the bar. Viewers see it is the actual site, and the margin survives the
 * panel's slow zoom so no text is cut off.
 */
async function framedScreenshot(png: string, pageUrl: string, out: string): Promise<boolean> {
  const addr = (() => { try { const u = new URL(pageUrl); return `${u.hostname.replace(/^www\./, '')}${u.pathname === '/' ? '' : u.pathname}`.slice(0, 60); } catch { return ''; } })();
  const txt = path.join(WORK_DIR, 'site_addr.txt');
  fs.writeFileSync(txt, addr || 'official website');
  const font = path.join(HERE, 'assets/fonts/Poppins-Bold.ttf');
  const vf = [
    'scale=1180:-2',
    'pad=1440:990:130:170:0x14161b',
    'drawbox=x=130:y=104:w=1180:h=66:color=0x2a2d35:t=fill',
    'drawbox=x=150:y=128:w=18:h=18:color=0xff5f57:t=fill', 'drawbox=x=178:y=128:w=18:h=18:color=0xfebc2e:t=fill', 'drawbox=x=206:y=128:w=18:h=18:color=0x28c840:t=fill',
    'drawbox=x=250:y=116:w=900:h=42:color=0x3a3e48:t=fill',
    `drawtext=fontfile=${font}:textfile=${txt}:fontsize=26:fontcolor=0xe8eaf0:x=272:y=123`
  ].join(',');
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', png, '-frames:v', '1', '-vf', vf, '-q:v', '3', out], { timeoutMs: 60000 });
  return r.code === 0 && fs.existsSync(out);
}

/** Download a real image, reject tiny / duplicate ones, normalise to JPEG. */
const usedImageHashes = new Set<string>();
const usedImageUrls = new Set<string>();
const WIKI_UA = 'AnimatoAutoPoster/5.1 (https://github.com/animato-auto-poster; image credits shown in every video) Node.js';
async function takeImage(img: WebImage, raw: string, out: string): Promise<boolean> {
  if (usedImageUrls.has(img.url)) return false;
  usedImageUrls.add(img.url);
  // Wikimedia (since late 2025): images must be fetched with a descriptive bot User-Agent, a Referer and a
  // standard thumbnail width (1280) — a browser-looking UA without a referer gets HTTP 429 (every picture failed).
  const wm = /(^|\.)wikimedia\.org\//i.test(img.url.replace(/^https?:\/\//, ''));
  const headers: Record<string, string> = wm
    ? { 'User-Agent': WIKI_UA, Referer: img.referer || 'https://commons.wikimedia.org/', Accept: 'image/webp,image/jpeg,image/*;q=0.8' }
    : { 'User-Agent': BROWSER_UA, Accept: 'image/avif,image/webp,image/*,*/*;q=0.8' };
  let ok = await download(img.url, raw, 30000, headers);
  // Rate-limited thumbnail → try the next standard size down once.
  if (!ok && wm && /\/1280px-/.test(img.url)) { await sleep(1500); ok = await download(img.url.replace('/1280px-', '/960px-'), raw, 30000, headers); }
  if (!ok) return false;
  const [w, h] = await imageSize(raw);
  if (w < MIN_REAL_W || h < 300 || w / h > 4 || h / w > 4) return false;
  const hash = crypto.createHash('md5').update(fs.readFileSync(raw)).digest('hex');
  if (usedImageHashes.has(hash)) return false;
  usedImageHashes.add(hash);
  return toJpeg(raw, out);
}

/** Scenes that talk about using the tool / its website get the real screenshot. */
const WEBSITE_WORDS = /\b(website|site|open|go to|visit|sign ?up|log ?in|download|install|app store|play store|click|tap|type|upload|paste|dashboard|interface|homepage|free plan|pricing)\b/i;
/** News: the scene that names where the story comes from gets the article's own page. */
const SOURCE_WORDS = /\b(according to|reported|reports|report|announced|confirmed|statement|published|sources?|story|article|headline|per )\b/i;

/** The fixed look of every named character who appears in this scene (keeps people consistent across images). */
function castFor(script: Script, scene: Scene): string {
  if (!script.characters) script.characters = CFG.storyCharacters;
  if (!script.characters) return '';
  const parts = script.characters.split(/[;\n]+|\.\s+(?=[A-Z][a-z]+[:,( ])/).map((x) => x.trim()).filter(Boolean);
  const text = `${scene.imagePrompt} ${scene.narration}`.toLowerCase();
  const hits = parts.filter((p) => {
    const name = (p.match(/^([A-Z][a-zA-Z'-]+)/) || [])[1];
    return name ? text.includes(name.toLowerCase()) : false;
  });
  if (hits.length) return `Characters: ${hits.join('; ')}`;
  return CFG.category === 'stories' && parts.length === 1 && /\b(she|he|her|his|they)\b/.test(text) ? `Character: ${parts[0]}` : '';
}

/** Stories are illustrated as an animated family film — never photoreal, never real people. */
const ANIMATED_STYLE = 'high-end 3D animated feature-film still, every person is an ORIGINAL stylised 3D cartoon character with big expressive eyes and soft rounded features (not any existing movie, TV or game character, no logos, no brand mascots), cartoon environment, warm cinematic lighting, rich saturated colours, detailed painterly background, wholesome family-film render, no real people, no photographic humans, no realistic faces, no photorealism';
const PHOTO_WORDS = /\b(photo(graph(y|ic)?)?|photoreal(istic)?|realistic|real[- ]life|dslr|35 ?mm|50 ?mm|bokeh|film still|cinematic still|hyper ?real(istic)?|raw photo|8k photo|portrait photo|headshot)\b/gi;
/** Real-person words become cartoon characters (a story picture never shows a real human). */
const PERSON_WORDS: [RegExp, string][] = [
  [/\b(man|woman|men|women|person|people|lady|ladies|gentleman|pastor|priest|girl|boy|child|children|kid|kids|villager|villagers|crowd|mother|father|old man|old woman)\b/gi, 'cartoon $1']
];
function styleFor(script: Script): string {
  return CFG.category === 'stories' ? ANIMATED_STYLE : script.visualStyle || '';
}
/** For stories: strip photo wording and make every person a cartoon character. */
const animatedPrompt = (p: string) => {
  if (CFG.category !== 'stories') return p;
  let out = p.replace(PHOTO_WORDS, 'animated');
  for (const [re, to] of PERSON_WORDS) out = out.replace(re, to);
  return out.replace(/\bcartoon cartoon\b/gi, 'cartoon').replace(/\s+/g, ' ').trim();
};

/**
 * Does the downloaded picture really show what the presenter is talking about?
 * A vision model looks at it (Gemini, then Groq's vision model). null = no
 * vision model available (the metadata check alone decides).
 */
let visionBudget = 40;
async function looksRight(file: string, subject: string): Promise<boolean | null> {
  if (visionBudget <= 0 || CFG.offline || !LLM.hasKeys || !subject) return null;
  visionBudget--;
  const small = `${file}.vis.jpg`;
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', file, '-vf', 'scale=512:-2', '-q:v', '5', small], { timeoutMs: 20000 });
  if (r.code !== 0 || !fs.existsSync(small)) return null;
  const v = await visionMatches(researchCtx(), fs.readFileSync(small).toString('base64'), subject, CFG.category);
  try { fs.unlinkSync(small); } catch {}
  if (!v) { visionBudget = 0; log('Image check: no vision model answered — relying on the images\' own titles/tags from now on.'); return null; }
  if (!v.match) log(`Image check: rejected a picture for "${subject}" (it shows ${v.shows || 'something else'}).`);
  return v.match;
}

/**
 * Story frames: a vision model checks that the characters are drawn correctly (no melted faces,
 * extra or fused limbs). false = redraw with a new seed; null = no vision model answered.
 */
let anatomyBudget = 36;
let anatomyRedraws = 0;
async function anatomyOk(file: string): Promise<boolean | null> {
  if (anatomyBudget <= 0 || CFG.offline || !LLM.hasKeys || ENV.STORY_ANATOMY_CHECK === 'off') return null;
  anatomyBudget--;
  const small = `${file}.anat.jpg`;
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', file, '-vf', 'scale=512:-2', '-q:v', '5', small], { timeoutMs: 20000 });
  if (r.code !== 0 || !fs.existsSync(small)) return null;
  const v = await visionAnatomy(researchCtx(), fs.readFileSync(small).toString('base64'));
  try { fs.unlinkSync(small); } catch {}
  if (!v) { anatomyBudget = 0; log('Anatomy check: no vision model answered — drawn scenes are used as they are.'); return null; }
  if (!v.ok) log(`Anatomy check: redrawing a scene (${v.problem || 'drawing error'}).`);
  return v.ok;
}

/** Realistic food photo when no real photo of that ingredient / step exists (cooking only). */
async function foodImage(s: Scene, subject: string, seed: number, raw: string, out: string): Promise<boolean> {
  const what = (s.imagePrompt && !/\b(person|people|man|woman|chef|hand|hands|face)\b/i.test(s.imagePrompt) ? s.imagePrompt : `${subject}, fresh, on a kitchen counter`).slice(0, 300);
  const prompt = slimPrompt(`${subject}, ${what}`, 220);
  for (let attempt = 0; attempt < 2; attempt++) {
    const got = await aiImage(prompt, seed + attempt * 101, raw, FOOD_STYLE);
    if (!got || !(await toJpeg(raw, out, got === 'ai' ? 0.04 : 0))) continue;
    if ((await looksRight(out, subject)) === false) continue;
    return true;
  }
  return false;
}

async function gatherImages(script: Script): Promise<{ files: (string | null)[]; aiCount: number; credits: (string | null)[] }> {
  const isStory = CFG.category === 'stories';
  const n = script.scenes.length;
  const files: (string | null)[] = new Array(n).fill(null);
  const credits: (string | null)[] = new Array(n).fill(null);
  let aiCount = 0;

  // Test hook (never set in production): take scene images from a local folder.
  const testDir = ENV.ANIMATO_TEST_IMAGES_DIR;
  const testImages = testDir && fs.existsSync(testDir) ? fs.readdirSync(testDir).filter((x) => /\.(jpe?g|png|webp)$/i.test(x)).sort() : [];
  if (testImages.length) {
    for (let i = 0; i < n; i++) { const out = path.join(WORK_DIR, `scene_${i}.jpg`); if (await toJpeg(path.join(testDir!, testImages[i % testImages.length]), out)) files[i] = out; }
    return { files, aiCount, credits };
  }

  if (isStory) {
    // ---- STORIES: fiction, so every picture is drawn — original 3D animated cartoon scenes only.
    const seedBase = parseInt(crypto.createHash('md5').update(`${CFG.campaignId}:${CFG.partNumber}`).digest('hex').slice(0, 6), 16);
    const ready = await startImageGenerator(); // already loading since the run started
    const deadline = Date.now() + 9 * 60 * 1000;
    const drawOne = async (i: number) => {
      if (!ready || imagegenDisabled || Date.now() > deadline) return;
      const s = script.scenes[i];
      const raw = path.join(WORK_DIR, `scene_${i}.raw`), out = path.join(WORK_DIR, `scene_${i}.jpg`);
      // Scene first, then the fixed look of the characters in it. The generator cleans the prompt
      // (calm medium shots, no crowds / fingers), adds the story style and trims the tail to fit.
      const prompt = [animatedPrompt(s.imagePrompt || s.narration).slice(0, 320), animatedPrompt(castFor(script, s)).slice(0, 160)].filter(Boolean).join('. ');
      const tries: string[] = [];
      let chosen = '';
      for (let attempt = 0; attempt < 3; attempt++) {
        if (attempt > 0 && (imagegenDisabled || Date.now() > deadline - 20000)) break;
        const got = await aiImage(prompt, seedBase + i * 7 + attempt * 977, raw, '', 'story');
        if (!got) break;
        const cand = path.join(WORK_DIR, `scene_${i}_try${attempt}.jpg`);
        if (!(await toJpeg(raw, cand, 0))) continue;
        tries.push(cand);
        const ok = await anatomyOk(cand); // false = melted face / extra limbs → draw it again
        if (ok !== false) { chosen = cand; break; }
        anatomyRedraws++;
      }
      if (!chosen && tries.length) chosen = tries[0]; // every try was flagged: keep the first one
      if (chosen) { fs.copyFileSync(chosen, out); files[i] = out; aiCount++; }
    };
    // One picture is drawn at a time (the CPU is the limit); the 2nd worker only overlaps file conversion.
    const queue = script.scenes.map((_, i) => i);
    await Promise.all(Array.from({ length: 2 }, async () => { while (queue.length) await drawOne(queue.shift()!); }));
  } else {
    // ---- EVERYTHING ELSE: real images found on the web. Nothing is generated.
    const deadline = Date.now() + 5 * 60 * 1000;
    const cat = CFG.category;
    const subject = (script.sourceStory?.title || script.sourceHeadline || script.title).replace(/\s*\(part \d+\)\s*$/i, '').slice(0, 160);

    // Ads: the advertiser's own product photos (imported from their PDF).
    const productFiles: string[] = [];
    if (cat === 'ads' && CFG.adImages.length && CFG.appUrl && !CFG.offline) {
      let k = 0;
      for (const id of CFG.adImages.slice(0, 12)) {
        const raw = path.join(WORK_DIR, `product_${k}.raw`), out = path.join(WORK_DIR, `product_${k}.jpg`);
        if (await download(`${CFG.appUrl}/api/automation/assets/${encodeURIComponent(id)}`, raw, 45000) && await toJpeg(raw, out)) { productFiles.push(out); k++; }
      }
      log(`Ad: ${productFiles.length}/${CFG.adImages.length} product images downloaded from the PDF.`);
    }
    let productCursor = 0;

    // COPYRIGHT-SAFE SOURCES ONLY. News/publisher photos and brands' marketing images
    // are never used: only public-domain, CC0 and CC BY images (credited), the
    // advertiser's own images (ads), and a screenshot of the tool's own interface
    // on tutorial steps (showing how to use it — review / instruction use).
    const forAds = cat === 'ads';
    const pool: WebImage[] = [];
    const briefUrl = forAds && !CFG.adProfile.service ? cleanOfficialUrl((CFG.adBrief.match(/\bhttps?:\/\/[^\s)"'<>]+|\bwww\.[a-z0-9-]+\.[a-z.]{2,}[^\s)"'<>]*/i) || [])[0]) : '';
    // A service has no website to show: never screenshot one for it.
    const official = forAds && CFG.adProfile.service ? '' : (script.officialUrl || briefUrl);
    /** News / tech: the page the story itself was published on. */
    const storyLink = cleanOfficialUrl(script.sourceStory?.link || '');
    // ---- REAL SCREENSHOTS (everything except stories)
    // News: the article's own page on the publisher's site. Tech / tutorials /
    // ads: the product's official site. Every capture is verified live — right
    // site, no cookie wall, no paywall, no error page, nothing blank — and shown
    // in a browser window with the real address, so what viewers see is exactly
    // what the site shows. If nothing passes, no screenshot is used at all.
    const shots: { file: string; credit: string; uses: number }[] = [];
    let shotUses = 0;
    const shotCap = cat === 'news' ? 2 : 3;
    if (official || storyLink || cat === 'tech') {
      const imgsP = forAds && official ? advertiserImages(official) : Promise.resolve([] as WebImage[]); // other brands' images are not used
      // Candidates, best first: the story's own article page, the URL the script
      // named, those sites' front pages, then the product's official site on
      // Wikidata. A candidate is only used if it passes every check in
      // siteScreenshot() — the first one that does wins.
      const tried = new Set<string>();
      const names = Array.from(new Set(script.scenes.map((x) => x.searchQuery).filter(Boolean))).slice(0, 2);
      const homeOf = (u: string) => { try { const x = new URL(u); return x.pathname === '/' ? '' : `${x.origin}/`; } catch { return ''; } };
      const expect = [script.sourceStory?.title || '', script.sourceHeadline || '', subject, ...names].filter(Boolean);
      const candidates: (() => Promise<string>)[] = [
        async () => storyLink,
        async () => official,
        async () => homeOf(storyLink),
        async () => homeOf(official),
        ...names.map((nm) => async () => wikidataOfficialSite(nm))
      ];
      const pngA = path.join(WORK_DIR, 'site_shot.png'), pngB = path.join(WORK_DIR, 'site_shot_b.png');
      let shot: Shot | null = null;
      const started = Date.now();
      for (const next of candidates) {
        if (Date.now() - started > 150_000) break;
        const url = await next();
        if (!url || tried.has(url)) continue;
        tried.add(url);
        shot = await siteScreenshot(url, pngA, { expect, second: pngB });
        if (shot) break;
      }
      const imgs = await imgsP;
      pool.push(...imgs);
      if (shot) {
        let k = 0;
        for (const png of shot.files) {
          const jpg = path.join(WORK_DIR, `site_shot_${k}.jpg`);
          // Always presented as a framed browser window with the real address.
          if ((await framedScreenshot(png, shot.finalUrl, jpg)) || (await toJpeg(png, jpg))) {
            shots.push({ file: jpg, credit: `Screenshot: ${hostOf(shot.finalUrl)}`, uses: 0 });
          }
          k++;
        }
        // Cite the page that actually loaded (tech / tutorials / ads only — a news
        // video cites its source story separately).
        if (cat !== 'news' && (!script.officialUrl || hostOf(script.officialUrl) !== hostOf(shot.finalUrl))) script.officialUrl = shot.finalUrl;
      }
      log(shots.length
        ? `Screenshots: ${shots.length} verified live view(s) of ${hostOf(shot!.finalUrl)}${cat === 'news' ? ' (the story\u2019s own page)' : ''}.`
        : `Screenshots: none — ${tried.size ? `${tried.size} candidate(s) failed verification (${Array.from(tried).map(hostOf).join(', ')})` : 'no page to shoot'}; licensed photos are used instead.`);
    }
    /** The least-used verified screenshot that is still within its budget. */
    const nextShot = () => {
      if (shotUses >= shotCap) return null;
      const free = shots.filter((x) => x.uses < 2).sort((a, b) => a.uses - b.uses);
      return free[0] || null;
    };

    /**
     * The first image that really shows `subject`:
     *   1. its own title / caption / tags name it — for a specific product every
     *      model identifier must be there ("Galaxy S25 Ultra" ≠ "Galaxy S23"), and
     *   2. a vision model looking at it agrees.
     */
    const tryList = async (list: WebImage[], raw: string, out: string, subject: string): Promise<WebImage | null> => {
      const strict = identifierTokens(subject).length > 0;
      let checked = 0;
      for (const img of list) {
        if (Date.now() > deadline || checked >= 5) return null;
        const needsCheck = img.kind === 'library' || img.kind === 'wiki';
        if (needsCheck && subject && !metadataMatches(subject, img.meta || '', strict)) continue;
        if (!(await takeImage(img, raw, out))) continue;
        checked++;
        if (needsCheck && (await looksRight(out, subject)) === false) continue;
        return img;
      }
      return null;
    };
    const fromPool = () => pool.filter((x) => !usedImageUrls.has(x.url));
    const attributions: string[] = [];
    let generatedFood = 0;

    const fetchOne = async (i: number) => {
      const s = script.scenes[i];
      const raw = path.join(WORK_DIR, `scene_${i}.raw`), out = path.join(WORK_DIR, `scene_${i}.jpg`);
      const set = (file: string, credit: string) => { files[i] = file; credits[i] = credit; };
      if (productFiles.length && (s.productShot || (forAds && s.shot === 'panel'))) { set(productFiles[productCursor++ % productFiles.length], 'Product image'); return; }
      // A website screenshot is always shown as a framed screen (panel), never stretched full-screen.
      if (WEBSITE_WORDS.test(s.narration) || (cat === 'news' && SOURCE_WORDS.test(s.narration))) {
        const sh = nextShot();
        if (sh) { sh.uses++; shotUses++; s.shot = 'panel'; set(sh.file, sh.credit); return; }
      }
      const q = s.searchQuery || subject.split(/\s+/).slice(0, 6).join(' ');
      // Best first: the named person / place / organisation's free Wikipedia image,
      // then freely licensed photo libraries for the scene, then for the whole topic.
      const sources: (() => Promise<WebImage[]>)[] = [];
      if (forAds) sources.push(async () => fromPool());
      sources.push(async () => wikiImages(q, forAds));
      sources.push(async () => libraryImages(q, forAds));
      if (q !== subject) sources.push(async () => wikiImages(subject.split(/\s+/).slice(0, 6).join(' '), forAds));
      if (q !== subject) sources.push(async () => libraryImages(subject.split(/\s+/).slice(0, 5).join(' '), forAds));
      for (const [k, src] of sources.entries()) {
        if (Date.now() > deadline) break;
        // Scene-level sources must show the scene's subject; topic-level ones the topic.
        const subj = k < (forAds ? 3 : 2) ? q : (script.factPack?.productName || subject.split(/\s+/).slice(0, 6).join(' '));
        const got = await tryList(await src(), raw, out, cat === 'ads' && k === 0 ? '' : subj);
        if (got) { set(out, got.credit); if (got.attribution && !attributions.includes(got.attribution)) attributions.push(got.attribution); return; }
        // Cooking: a picture of something else never beats a picture of THIS ingredient/step.
        if (cat === 'cooking' && k >= 1) break;
      }
      // Cooking: no real photo of this ingredient / step → a realistic food photo of exactly it.
      if (cat === 'cooking' && Date.now() < deadline + 90_000) {
        if (await foodImage(s, q, 9000 + i * 13, raw, out)) { set(out, 'AI-generated image'); generatedFood++; return; }
      }
      // Tech: the product's own verified website shows the exact product — better than a wrong one.
      const sh = nextShot() || (cat === 'tech' ? shots.sort((a, b) => a.uses - b.uses)[0] || null : null);
      if (sh) { sh.uses++; shotUses++; s.shot = 'panel'; set(sh.file, sh.credit); }
    };
    const queue = script.scenes.map((_, i) => i);
    await Promise.all(Array.from({ length: 4 }, async () => { while (queue.length) await fetchOne(queue.shift()!); }));
    // Second chance for scenes still without a picture: the whole topic, looser matching (no vision veto),
    // then an AI-made editorial illustration (credited as such). A blank or placeholder picture is never shown.
    const missing = script.scenes.map((_, i) => i).filter((i) => !files[i]);
    if (missing.length && !forAds) {
      log(`Images: ${missing.length}/${n} scene(s) had no matching photo — trying the whole topic, then an illustration.`);
      const topic = subject.split(/\s+/).slice(0, 6).join(' ');
      let looseList: WebImage[] | null = null;
      let aiMade = 0;
      for (const i of missing) {
        if (Date.now() > deadline + 120_000) break;
        const s = script.scenes[i];
        const raw = path.join(WORK_DIR, `scene_${i}.raw`), out = path.join(WORK_DIR, `scene_${i}.jpg`);
        if (!looseList) looseList = [...await wikiImages(topic, false), ...await libraryImages(topic, false), ...await libraryImages(topic.split(/\s+/).slice(0, 3).join(' '), false)];
        let got: WebImage | null = null;
        for (const img of looseList) {
          if (usedImageUrls.has(img.url)) continue;
          if (img.meta && !metadataMatches(topic.split(/\s+/).slice(0, 3).join(' '), img.meta, false)) continue;
          if (await takeImage(img, raw, out)) { got = img; break; }
        }
        if (got) { files[i] = out; credits[i] = got.credit; if (got.attribution && !attributions.includes(got.attribution)) attributions.push(got.attribution); continue; }
        if (aiMade < 4 && cat !== 'cooking') {
          const what = (s.imagePrompt || s.narration || topic).slice(0, 260);
          const prompt = slimPrompt(`${what}. ${topic}`, 230);
          const ai = await aiImage(prompt, 4000 + i * 17, raw, EDITORIAL_STYLE);
          if (ai && await toJpeg(raw, out, ai === 'ai' ? 0.04 : 0)) { files[i] = out; credits[i] = 'AI-generated illustration'; aiMade++; aiCount++; }
        }
      }
      log(`Images: second chance filled ${missing.filter((i) => files[i]).length}/${missing.length} (${aiMade} illustration(s)).`);
    }
    // Leftover product photos still beat a repeated image for ads.
    for (let i = 0; i < n && productFiles.length; i++) if (!files[i]) { files[i] = productFiles[productCursor++ % productFiles.length]; credits[i] = 'Product image'; }
    script.imageAttributions = attributions;
    aiCount += generatedFood;
    if (generatedFood) log(`Cooking: ${generatedFood} ingredient/step picture(s) had no real photo and were generated as realistic food photos.`);
  }

  // Scenes without an image reuse the nearest one so nothing is ever blank (never an invented picture).
  for (let i = 0; i < n; i++) {
    if (files[i]) continue;
    for (let d = 1; d < n; d++) {
      const j = files[i - d] ? i - d : files[i + d] ? i + d : -1;
      if (j >= 0) { files[i] = files[j]; credits[i] = credits[j]; break; }
    }
  }
  if (!files.some(Boolean)) {
    const grad = path.join(WORK_DIR, 'gradient.png');
    const colors = CFG.category === 'cooking' ? ['0x3b1d0f', '0x9a4a12'] : CFG.category === 'tech' ? ['0x061a2b', '0x0f4c75'] : ['0x0b0b1a', '0x3a1c4a'];
    await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', `gradients=s=${W}x${H}:c0=${colors[0]}:c1=${colors[1]}:x0=0:y0=0:x1=${W}:y1=${H}:nb_colors=2`, '-frames:v', '1', grad]);
    files.fill(grad);
  }
  script.imageCredits = Array.from(new Set(credits.filter((c): c is string => !!c && c !== 'Product image')));
  if (isStory) log(`Images: ${files.filter(Boolean).length}/${n} scenes, ${aiCount} drawn as original 3D animated scenes by the local image generator${anatomyRedraws ? ` (${anatomyRedraws} redrawn after the anatomy check)` : ''}${imagegenDisabled ? `; generator unavailable: ${imagegenDisabled}` : ''}.`);
  else log(`Images: ${new Set(files.filter(Boolean)).size} real image(s) for ${n} scenes — none generated. Sources: ${script.imageCredits.join(', ') || 'none found'}.`);
  return { files, aiCount, credits };
}

// ---------------------------------------------------------------------------
// 4. Character rig from the app (the character designed in the editor)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 5. Render in headless Chrome with the app's engine (falls back to FFmpeg)
// ---------------------------------------------------------------------------
function findChrome(): string | null {
  const candidates = [ENV.CHROME_PATH, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/opt/google/chrome/chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean) as string[];
  for (const dir of [path.join(os.homedir(), '.cache/ms-playwright'), '/opt/pw-browsers', path.join(os.homedir(), '.cache/puppeteer/chrome')]) {
    try {
      for (const d of fs.readdirSync(dir)) {
        for (const sub of ['chrome-linux/chrome', 'chrome-linux64/chrome', `${d}/chrome-linux64/chrome`]) candidates.push(path.join(dir, d, sub));
      }
    } catch {}
  }
  return candidates.find((c) => { try { return fs.statSync(c).isFile(); } catch { return false; } }) || null;
}

/**
 * Where the presenter is actually speaking, and how loud, measured from the
 * narration audio itself (50 ms windows) — works even when word timings are estimates.
 */
async function analyseVoice(file: string): Promise<{ levelDb: number; speech: { start: number; end: number }[] } | null> {
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-ac', '1', '-ar', '16000', '-f', 's16le', '-'], { timeoutMs: 60000 });
  if (r.code !== 0 || r.stdout.length < 32000) return null;
  const pcm = new Int16Array(r.stdout.buffer, r.stdout.byteOffset, Math.floor(r.stdout.length / 2));
  const WIN = 800, sec = WIN / 16000;
  const dbs: number[] = [];
  for (let i = 0; i + WIN <= pcm.length; i += WIN) {
    let sum = 0;
    for (let k = i; k < i + WIN; k++) sum += (pcm[k] / 32768) ** 2;
    dbs.push(10 * Math.log10(sum / WIN + 1e-12));
  }
  const peak = Math.max(...dbs);
  const gate = Math.max(-45, peak - 32);
  let power = 0, count = 0;
  const speech: { start: number; end: number }[] = [];
  dbs.forEach((d, i) => {
    if (d <= gate) return;
    power += 10 ** (d / 10); count++;
    const t = i * sec, last = speech[speech.length - 1];
    if (last && t - last.end < 0.6) last.end = t + sec;          // short gaps are part of the phrase
    else speech.push({ start: t, end: t + sec });
  });
  if (!count) return null;
  return { levelDb: 10 * Math.log10(power / count), speech: speech.filter((x) => x.end - x.start >= 0.12) };
}

/** Integrated loudness (LUFS, EBU R128 — how loud it SOUNDS, bass weighted down) of an audio file. */
async function loudnessLufs(file: string): Promise<number | null> {
  const r = await run('ffmpeg', ['-hide_banner', '-nostats', '-i', file, '-af', 'ebur128=framelog=quiet', '-f', 'null', '-'], { timeoutMs: 60000 });
  const m = r.stderr.match(/Integrated loudness:[\s\S]*?I:\s*(-?[\d.]+)\s*LUFS/);
  const v = m ? Number(m[1]) : NaN;
  return Number.isFinite(v) && v > -70 ? v : null;
}

/**
 * Background music is ORIGINAL — composed and synthesised for this video (see
 * music.ts). No third-party track is ever used, so there is nothing to license,
 * credit or get a Content ID claim for.
 *
 * It is mixed to be clearly HEARD but never fight the voice:
 *   - EQ for phone speakers (no sub-bass) with a dip where speech lives;
 *   - 13 dB under the measured voice level while the presenter speaks (the
 *     usual broadcast "music bed" level), 7 dB under in pauses, intro and outro;
 *   - smooth swells from the real speech timing instead of a pumping compressor.
 */
const MUSIC_UNDER_VOICE_DB = 13;
const MUSIC_PAUSE_LIFT_DB = 6;
async function findMusic(seconds: number, narration: Narration): Promise<string | null> {
  try {
    const mood = moodFor(CFG.category, CFG.subGenre);
    const file = path.join(WORK_DIR, `music_${mood}.wav`);
    const t0 = Date.now();
    const { L, R, sampleRate } = composeBuffers(mood, Math.max(10, seconds), `${CFG.campaignId}:${CFG.partNumber}`);
    eqForVoice(L, R, sampleRate);
    const voice = await analyseVoice(narration.audioPath);
    const speech = voice?.speech?.length ? voice.speech : narration.words.map((w) => ({ start: w.start, end: w.end }));
    // Match PERCEIVED loudness (LUFS): a bass-heavy bed measures loud but sounds quiet,
    // so raw RMS would leave it inaudible on phones. RMS is only the fallback.
    fs.writeFileSync(file, encodeWav(L, R, sampleRate));
    const [voiceLufs, musicLufs] = await Promise.all([loudnessLufs(narration.audioPath), loudnessLufs(file)]);
    const perceived = voiceLufs !== null && musicLufs !== null;
    const voiceDb = perceived ? voiceLufs! : voice?.levelDb ?? -18;
    const musicDb = perceived ? musicLufs! : levelDb(L, R);
    const speechGain = 10 ** ((voiceDb - MUSIC_UNDER_VOICE_DB - musicDb) / 20);
    automateLevel(L, R, sampleRate, { speech, speechGain, pauseGain: speechGain * 10 ** (MUSIC_PAUSE_LIFT_DB / 20) });
    fs.writeFileSync(file, encodeWav(L, R, sampleRate));
    log(`Music: original ${mood} soundtrack composed for this video (${Math.round(seconds)}s) — ${MUSIC_UNDER_VOICE_DB} dB under the voice (${voiceDb.toFixed(1)} ${perceived ? 'LUFS' : 'dBFS'}) while speaking, +${MUSIC_PAUSE_LIFT_DB} dB between ${speech.length} spoken phrase(s), intro and outro; ${((Date.now() - t0) / 1000).toFixed(1)}s.`);
    return file;
  } catch (err: any) {
    log(`⚠️ Music could not be composed (${err?.message || err}) — the video has voice only.`);
    return null;
  }
}

/** Audio: narration + the music bed (already levelled under the voice). */
function audioArgs(narration: string, music: string | null, firstInput: number): { inputs: string[]; filter: string } {
  const inputs = ['-i', narration];
  if (music) inputs.push('-stream_loop', '-1', '-i', music);
  const v = firstInput, m = firstInput + 1;
  const filter = music
    // The music file is already levelled and shaped around the voice (findMusic), so it is mixed as-is.
    ? `[${v}:a]aresample=48000,apad[vo];[${m}:a]aresample=48000[mus];[vo][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[aout]`
    : `[${v}:a]aresample=48000,apad[aout]`;
  return { inputs, filter };
}

async function renderWithStage(opts: {
  narration: Narration; scenes: Scene[]; times: { start: number; end: number }[]; cues: { t: number; tag: string }[]; images: (string | null)[];
  title: string; badge: string; endCard: string; music: string | null; duration: number; credits?: (string | null)[]; musicalStage?: boolean; chorusIntervals?: { start: number; end: number }[];
  /** Override the canvas the stage renders at (used for ads that render landscape then get framed into a Short — see adLandscapeShort). Defaults to the job's actual frame. */
  size?: { w: number; h: number };
  musical?: { stageId?: number; stageAutoSeed?: string; style?: string; sections?: { start: number; end: number; tag: string; emotion?: string }[]; vocalSpans?: { start: number; end: number }[]; drumLevel?: number; keysLevel?: number };
  stems?: Stems;
  lyrics?: { lines: { text: string; en: string }[]; noSpaces: boolean };
}): Promise<{ ok: boolean; character: string; reason?: string }> {
  const audioExt = path.extname(opts.narration.audioPath) || '.mp3';
  const accent = CFG.category === 'cooking' ? '#FFB020' : CFG.category === 'tech' ? '#22D3EE' : CFG.category === 'news' ? '#FF4D4D' : '#FFD23F';
  // Musicals (a full concert, every frame) render at 75% size and are upscaled — about twice as fast.
  const fast = !!opts.musicalStage && !opts.size && W * H > 1280 * 720;
  const drawSize = fast ? { w: Math.round(W * 0.75 / 2) * 2, h: Math.round(H * 0.75 / 2) * 2 } : opts.size;
  const RW = drawSize?.w || W, RH = drawSize?.h || H;
  const job = {
    width: RW, height: RH, fps: FPS, duration: opts.duration, category: CFG.category,
    title: opts.title, badge: opts.badge, endCard: opts.endCard, accent,
    audio: `/audio/narration${audioExt}`,
    words: opts.narration.words.map((w) => ({ text: w.text, start: +w.start.toFixed(3), end: +w.end.toFixed(3), ...(typeof w.line === 'number' ? { line: w.line } : {}) })),
    ...(opts.lyrics ? { lyrics: opts.lyrics } : {}),
    wordsReliable: opts.narration.wordsReliable,
    segments: opts.scenes.map((s, i) => ({ start: opts.times[i].start, end: opts.times[i].end, text: s.narration, image: opts.images[i] ? `/img/${path.basename(opts.images[i]!)}` : null, shot: s.shot, emotion: s.emotion, credit: opts.credits?.[i] || null })),
    cues: opts.cues,
    musicalStage: !!opts.musicalStage,
    chorusIntervals: opts.chorusIntervals || [],
    // The presenter: the CSS character spec designed in the app (the stage draws it).
    characterSpec: CFG.characterSpec,
    // A different outfit (same person) in every video of this automation.
    wardrobeSeed: `${CFG.campaignId || CFG.campaignName || 'local'}:${CFG.partNumber}`,
    ...(opts.musical ? { musical: opts.musical, musicalStyle: opts.musical.style || '' } : {}),
    // Isolated stems (when separation worked): vocals drive the lip sync, drums the drummer, other the pianist.
    ...(opts.stems ? { lipAudio: '/audio/vocals.wav', ...(opts.stems.drums ? { drumsAudio: '/audio/drums.wav' } : {}), ...(opts.stems.other ? { keysAudio: '/audio/other.wav' } : {}) } : {}),
    gender: CFG.gender,
    format: CFG.format,
    fontUrl: '/font/Poppins-Bold.ttf'
  };

  const audioExt2 = path.extname(opts.narration.audioPath) || '.mp3';
  const r = await runStage({
    stageFile: 'stage.js', job, duration: opts.duration,
    audio: audioArgs(opts.narration.audioPath, opts.music, 1),
    files: { [`/audio/narration${audioExt2}`]: opts.narration.audioPath, ...(opts.stems ? { '/audio/vocals.wav': opts.stems.vocals, ...(opts.stems.drums ? { '/audio/drums.wav': opts.stems.drums } : {}), ...(opts.stems.other ? { '/audio/other.wav': opts.stems.other } : {}) } : {}) },
    size: drawSize, scaleTo: fast ? { w: W, h: H } : undefined
  });
  return { ok: r.ok, character: r.character || 'none', reason: r.reason };
}

/**
 * Runs a stage bundle (stage.js: the presenter; anim.js: podcasts, films,
 * stickman) in headless Chrome and encodes its frames with the given audio.
 */
async function runStage(opts: { stageFile: string; job: any; duration: number; audio: { inputs: string[]; filter: string } | null; files: Record<string, string>; size?: { w: number; h: number }; imageOut?: string; scaleTo?: { w: number; h: number } }): Promise<{ ok: boolean; character?: string; reason?: string }> {
  const W = opts.size?.w || FRAME_W, H = opts.size?.h || FRAME_H;
  const chrome = findChrome();
  if (!chrome) return { ok: false, character: 'none', reason: 'Chrome not found on the runner' };
  const stageJs = path.join(HERE, opts.stageFile);
  if (!fs.existsSync(stageJs)) return { ok: false, character: 'none', reason: `${opts.stageFile} missing` };
  const job = opts.job;
  try { fs.writeFileSync(path.join(WORK_DIR, `job_${opts.stageFile.replace(/\W+/g, '_')}.json`), JSON.stringify(job)); } catch {}
  const totalFrames = Math.ceil(opts.duration * FPS);
  const frameBytes = W * H * 4;
  const aud = opts.audio;
  const ffArgs = opts.imageOut || !aud
    ? ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-r', String(FPS), '-i', 'pipe:0', '-frames:v', '1', '-q:v', '2', opts.imageOut || path.join(WORK_DIR, 'frame.jpg')]
    : ['-hide_banner', '-loglevel', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-r', String(FPS), '-i', 'pipe:0',
    ...aud.inputs, '-filter_complex', aud.filter, '-map', '0:v', '-map', '[aout]',
    // Rendered smaller for speed (musicals) → upscaled to the final frame here.
    ...(opts.scaleTo ? ['-vf', `scale=${opts.scaleTo.w}:${opts.scaleTo.h}:flags=lanczos`] : []),
    '-t', opts.duration.toFixed(3), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', OUTPUT_VIDEO];
  const ff = spawn('ffmpeg', ffArgs, { stdio: ['pipe', 'ignore', 'pipe'] });
  let ffErr = '';
  ff.stderr.on('data', (d: Buffer) => { ffErr += d.toString(); if (ffErr.length > 20000) ffErr = ffErr.slice(-10000); });
  const ffDone = new Promise<number>((resolve) => ff.on('close', (code: number | null) => resolve(code ?? -1)));
  ff.stdin.on('error', () => {});

  const types: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.svg': 'image/svg+xml', '.gif': 'image/gif', '.ttf': 'font/ttf', '.mp3': 'audio/mpeg', '.wav': 'audio/wav' };
  const serveFile = (res: any, file: string) => {
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    fs.createReadStream(file).pipe(res);
  };

  let framesWritten = 0;
  let finished: ((r: { ok: boolean; reason?: string; character?: string }) => void) | null = null;
  const result = new Promise<{ ok: boolean; reason?: string; character?: string }>((resolve) => { finished = resolve; });
  let lastProgressReport = 0;
  let pageErrors = 0;

  const server = http.createServer((req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    const p = decodeURIComponent(url.pathname);
    if (req.method === 'GET') {
      if (p === '/' || p === '/index.html') return serveFile(res, path.join(HERE, 'stage.html'));
      if (p === '/stage.js') return serveFile(res, stageJs);
      if (p === '/job.json') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(job)); return; }
      if (p === '/font/Poppins-Bold.ttf') return serveFile(res, path.join(HERE, 'assets/fonts/Poppins-Bold.ttf'));
      if (p.startsWith('/img/')) return serveFile(res, path.join(WORK_DIR, path.basename(p)));
      if (opts.files[p]) return serveFile(res, opts.files[p]);
      res.writeHead(404); res.end(); return;
    }
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', async () => {
      const body = Buffer.concat(chunks);
      if (p === '/frame') {
        if (body.length !== frameBytes) { res.writeHead(400); res.end(); return; }
        const ok = ff.stdin.write(body);
        framesWritten++;
        if (!ok) await new Promise((r) => ff.stdin.once('drain', r));
        res.writeHead(204); res.end();
        return;
      }
      let msg: any = {};
      try { msg = JSON.parse(body.toString() || '{}'); } catch {}
      res.writeHead(204); res.end();
      if (p === '/log') {
        console.log(`[stage${msg.level && msg.level !== 'info' ? ` ${msg.level}` : ''}] ${msg.msg}`);
        if (msg.level === 'error') pageErrors++;
      } else if (p === '/progress') {
        const pct = Math.round((msg.frame / Math.max(1, msg.total)) * 100);
        log(`Rendering ${pct}% (${msg.frame}/${msg.total} frames, ${msg.fps} fps)`);
        if (Date.now() - lastProgressReport > 20000) {
          lastProgressReport = Date.now();
          reportStatus('running', `4/5 Rendering the video (${pct}%)`, Math.round(60 + pct * 0.25), '');
        }
      } else if (p === '/done') {
        finished?.({ ok: true, character: msg.character });
      } else if (p === '/fail') {
        finished?.({ ok: false, reason: msg.error || 'stage failed' });
      }
    });
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
  const port = (server.address() as any).port;

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'animato-chrome-'));
  const chromeArgs = ['--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--no-first-run', '--no-default-browser-check',
    '--disable-extensions', '--mute-audio', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows', '--autoplay-policy=no-user-gesture-required', `--user-data-dir=${profile}`,
    '--remote-debugging-port=0', '--window-size=1280,800', '--js-flags=--max-old-space-size=4096', `http://127.0.0.1:${port}/`];
  const browser = spawn(chrome, chromeArgs, { stdio: ['ignore', 'ignore', 'pipe'] });
  let chromeErr = '';
  browser.stderr.on('data', (d: Buffer) => { chromeErr += d.toString(); if (chromeErr.length > 20000) chromeErr = chromeErr.slice(-10000); });
  browser.on('close', (code: number | null) => finished?.({ ok: false, reason: `Chrome exited (${code}): ${chromeErr.split('\n').filter((l) => /error|fatal/i.test(l)).slice(-3).join(' | ')}` }));
  log(`Rendering ${totalFrames} frames at ${W}x${H} in headless Chrome (${path.basename(chrome)})…`);

  // Watchdog: only a STALLED render fails (no new frame for 2 min). A slow but moving render keeps
  // going until the job's own time budget is nearly used up (then it stops with a clear message).
  const startedAt = Date.now();
  let lastCount = 0, lastMoveAt = Date.now();
  const timer = setInterval(() => {
    if (framesWritten !== lastCount) { lastCount = framesWritten; lastMoveAt = Date.now(); }
    const mins = ((Date.now() - startedAt) / 60000).toFixed(1);
    if (Date.now() - lastMoveAt > 120_000) finished?.({ ok: false, reason: `stage stalled (no frame for 2 min) after ${mins} min (${framesWritten}/${totalFrames} frames)` });
    else if (jobMinutesLeft() < 4) finished?.({ ok: false, reason: `out of job time after ${mins} min of rendering (${framesWritten}/${totalFrames} frames)` });
  }, 10_000);
  const outcome = await result;
  clearInterval(timer);
  try { browser.kill('SIGKILL'); } catch {}
  server.close();
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}

  if (!outcome.ok || framesWritten < totalFrames * 0.98) {
    try { ff.kill('SIGKILL'); } catch {}
    await ffDone;
    return { ok: false, character: 'none', reason: outcome.reason || `only ${framesWritten}/${totalFrames} frames rendered` };
  }
  ff.stdin.end();
  const code = await ffDone;
  if (code !== 0) return { ok: false, character: 'none', reason: `FFmpeg failed: ${ffErr.trim().split('\n').slice(-3).join(' | ')}` };
  if (pageErrors) log(`⚠️ The stage reported ${pageErrors} error(s) but finished.`);
  return { ok: true, character: outcome.character || 'unknown' };
}

/** Fallback when Chrome is unavailable: scene images + captions, no character. */
async function renderFallback(opts: { narration: Narration; times: { start: number; end: number }[]; images: (string | null)[]; title: string; badge: string; music: string | null; duration: number; size?: { w: number; h: number } }) {
  const W = opts.size?.w || FRAME_W, H = opts.size?.h || FRAME_H;
  const list = path.join(WORK_DIR, 'scenes.ffconcat');
  const lines = ['ffconcat version 1.0'];
  opts.images.forEach((img, i) => {
    lines.push(`file '${img}'`);
    lines.push(`duration ${Math.max(0.5, opts.times[i].end - opts.times[i].start).toFixed(3)}`);
  });
  lines.push(`file '${opts.images[opts.images.length - 1]}'`);
  fs.writeFileSync(list, lines.join('\n') + '\n');
  const ass = path.join(WORK_DIR, 'captions.ass');
  const size = Math.round(Math.min(W, H) * 0.075);
  const esc = (t: string) => t.replace(/[{}\\]/g, '');
  const ts = (t: number) => { const cs = Math.max(0, Math.round(t * 100)); return `${Math.floor(cs / 360000)}:${String(Math.floor(cs / 6000) % 60).padStart(2, '0')}:${String(Math.floor(cs / 100) % 60).padStart(2, '0')}.${String(cs % 100).padStart(2, '0')}`; };
  const groups: { s: number; e: number; t: string }[] = [];
  let cur: Word[] = [];
  const flush = () => { if (cur.length) groups.push({ s: cur[0].start, e: cur[cur.length - 1].end + 0.1, t: cur.map((w) => w.text).join(' ') }); cur = []; };
  for (const w of opts.narration.words) { cur.push(w); if (cur.length >= 3 || /[.!?,]$/.test(w.text)) flush(); }
  flush();
  fs.writeFileSync(ass, `[Script Info]\nScriptType: v4.00+\nPlayResX: ${W}\nPlayResY: ${H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: C,Poppins,${size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,${Math.round(size * 0.12)},3,5,60,60,0,1\nStyle: B,Poppins,${Math.round(size * 0.45)},&H00111111,&H00FFFFFF,&H003FD2FF,&H003FD2FF,-1,0,0,0,100,100,1,0,3,${Math.round(size * 0.25)},0,8,40,40,${Math.round(H * 0.04)},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 1,${ts(0)},${ts(opts.duration)},B,,0,0,0,,${esc(opts.badge)}\n${groups.map((g) => `Dialogue: 0,${ts(g.s)},${ts(g.e)},C,,0,0,0,,${esc(g.t.toUpperCase())}`).join('\n')}\n`);
  const aud = audioArgs(opts.narration.audioPath, opts.music, 1);
  const vf = `[0:v]scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H},setsar=1,fps=${FPS},subtitles=filename='${ass}':fontsdir='${path.join(HERE, 'assets/fonts')}',format=yuv420p[vout]`;
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, ...aud.inputs,
    '-filter_complex', `${vf};${aud.filter}`, '-map', '[vout]', '-map', '[aout]', '-t', opts.duration.toFixed(3),
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', OUTPUT_VIDEO], { timeoutMs: 20 * 60 * 1000 });
  if (r.code !== 0) throw new PipelineError('render_failed', `FFmpeg failed: ${r.stderr.trim().split('\n').slice(-4).join(' | ')}`);
}

// ---------------------------------------------------------------------------
// 6. YouTube
// ---------------------------------------------------------------------------
let cachedYouTubeToken = '';
let verifiedYouTubeChannelId = '';

async function youtubeAccessToken(): Promise<string> {
  if (cachedYouTubeToken && (!CFG.ytChannelId || verifiedYouTubeChannelId === CFG.ytChannelId)) return cachedYouTubeToken;
  if (!CFG.ytRefreshToken) {
    // The app keeps the YouTube connection; it hands this run a short-lived access token.
    const r = await appRequest('POST', `${campaignPath()}/youtube-token`, {});
    if (r?.status === 200 && r.data?.access_token) {
      cachedYouTubeToken = String(r.data.access_token);
      console.log(`::add-mask::${cachedYouTubeToken}`);
    } else if (r?.data?.code === 'youtube_reauth') {
      throw new PipelineError('youtube_reauth', 'YouTube access expired or was revoked. Reconnect YouTube in this automation\'s dashboard.');
    } else {
      throw new PipelineError(r?.data?.code === 'youtube_not_connected' ? 'youtube_not_connected' : 'youtube_auth', `Could not get YouTube access from the app (${r?.data?.error || `HTTP ${r?.status ?? 'no answer'}`}).`);
    }
  }
  if (!cachedYouTubeToken && !CFG.ytClientSecret) throw new PipelineError('youtube_config', 'The YouTube connection is not available to this run.');
  const res = cachedYouTubeToken ? null : await fetch(CFG.googleTokenUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ client_id: CFG.ytClientId, client_secret: CFG.ytClientSecret, refresh_token: CFG.ytRefreshToken, grant_type: 'refresh_token' }).toString(),
    signal: AbortSignal.timeout(30000)
  });
  const data: any = res ? await res.json().catch(() => ({})) : { access_token: cachedYouTubeToken };
  if ((res && !res.ok) || !data.access_token) {
    if (data?.error === 'invalid_grant') {
      throw new PipelineError('youtube_reauth', 'YouTube access expired or was revoked. Reconnect YouTube in this automation\'s dashboard. (If your Google OAuth consent screen is still in "Testing", Google expires the connection every 7 days — set it to "In production".)');
    }
    throw new PipelineError('youtube_auth', `Google token refresh failed (HTTP ${res?.status}): ${data?.error_description || data?.error || 'unknown error'}`);
  }
  cachedYouTubeToken = data.access_token;

  // Final destination guard: resolve the YouTube channel for this OAuth
  // credential and refuse to upload if it is not the channel assigned to this
  // automation. A routing mistake therefore becomes a failed run, never a
  // cross-post to another automation.
  if (CFG.ytChannelId) {
    const ch = await fetch('https://www.googleapis.com/youtube/v3/channels?part=id,snippet&mine=true', {
      headers: { Authorization: `Bearer ${cachedYouTubeToken}` },
      signal: AbortSignal.timeout(30000)
    });
    const chData: any = await ch.json().catch(() => ({}));
    const actualId = String(chData?.items?.[0]?.id || '');
    const actualTitle = String(chData?.items?.[0]?.snippet?.title || '');
    if (!ch.ok || !actualId) {
      throw new PipelineError('youtube_routing', `Could not verify the YouTube destination for automation ${CFG.campaignId || '(unknown)'}.`);
    }
    if (actualId !== CFG.ytChannelId) {
      throw new PipelineError(
        'youtube_routing',
        `YouTube destination mismatch for automation ${CFG.campaignId || '(unknown)'}: this automation is assigned to channel ${CFG.ytChannelId}, but the OAuth credential resolves to ${actualId}${actualTitle ? ` (${actualTitle})` : ''}. Nothing was uploaded.`
      );
    }
    verifiedYouTubeChannelId = actualId;
    log(`YouTube destination verified: ${actualTitle || actualId} (${actualId}).`);
  }

  return cachedYouTubeToken;
}

const YT_CATEGORY: Record<string, string> = { cooking: '26', tech: '28', stories: '24', news: '25', ads: '22', musical: '10' };

/** Build a YouTube-safe description. YouTube's limit is 5,000 UTF-8 bytes,
 * not 5,000 JavaScript characters. Strip controls and invalid URL-like text,
 * normalize whitespace, and leave headroom so emoji/non-ASCII text cannot
 * accidentally cross the API limit.
 */
function sanitizeYouTubeDescription(input: unknown): string {
  let text = String(input || '')
    .normalize('NFC')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .replace(/[<>]/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{4,}/g, '\n\n\n')
    .trim();
  if (!text) text = 'Created automatically with Animato AutoPoster Studio.';
  while (Buffer.byteLength(text, 'utf8') > 4900) {
    text = text.slice(0, Math.max(0, text.length - 64)).trimEnd();
  }
  return text;
}

function youtubeHashtagLine(values: unknown[]): string {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const h = String(value || '').trim().replace(/^#+/, '').toLowerCase().replace(/[^a-z0-9_]/g, '').slice(0, 30);
    if (h.length >= 3 && !seen.has(h)) {
      seen.add(h);
      out.push(`#${h}`);
    }
    if (out.length >= 7) break;
  }
  // The first three hashtags show above the title: the two most specific ones, then #shorts.
  const i = out.indexOf('#shorts');
  if (i > 2) { out.splice(i, 1); out.splice(2, 0, '#shorts'); }
  return out.slice(0, 6).join(' ');
}

async function uploadToYouTube(meta: { title: string; description: string; tags: string[]; synthetic: boolean }): Promise<{ videoId: string; url: string; privacy: string }> {
  const token = await youtubeAccessToken();
  const bytes = fs.readFileSync(OUTPUT_VIDEO);
  let title = meta.title.replace(/[<>]/g, '').trim() || 'New video';
  // No "#Shorts" in the title: YouTube classifies Shorts by length + vertical frame, and
  // the tag is already in the description's hashtag line — the title stays clean.
  title = title.slice(0, 100);
  let tagChars = 0;
  const tags = meta.tags.filter((t) => { tagChars += t.length + 3; return tagChars < 480; });
  const body = {
    snippet: { title, description: sanitizeYouTubeDescription(meta.description), tags, categoryId: YT_CATEGORY[CFG.category] || '24', defaultLanguage: 'en', defaultAudioLanguage: 'en' },
    status: { privacyStatus: CFG.privacy, selfDeclaredMadeForKids: false, containsSyntheticMedia: meta.synthetic }
  };
  const init = await fetch(`${CFG.youtubeUploadBase}/videos?uploadType=resumable&part=snippet,status`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json; charset=UTF-8', 'X-Upload-Content-Length': String(bytes.length), 'X-Upload-Content-Type': 'video/mp4' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(60000)
  });
  if (!init.ok) {
    const text = await init.text();
    const reason = /quotaExceeded/.test(text) ? 'YouTube API daily quota reached — uploads resume after midnight Pacific time.'
      : /uploadLimitExceeded/.test(text) ? 'This YouTube channel has hit its daily upload limit.'
      : /youtubeSignupRequired/.test(text) ? 'The connected Google account has no YouTube channel yet.'
      : `YouTube rejected the upload (HTTP ${init.status}): ${text.slice(0, 300)}`;
    throw new PipelineError('youtube_upload', reason);
  }
  const location = init.headers.get('location');
  if (!location) throw new PipelineError('youtube_upload', 'YouTube did not return an upload URL.');
  let lastErr = '';
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const put = await fetch(location, { method: 'PUT', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'video/mp4' }, body: bytes, signal: AbortSignal.timeout(10 * 60 * 1000) });
      const text = await put.text();
      if (put.ok) {
        const data = JSON.parse(text || '{}');
        if (!data.id) throw new Error('upload finished without a video id');
        const url = IS_SHORTS ? `https://youtube.com/shorts/${data.id}` : `https://www.youtube.com/watch?v=${data.id}`;
        return { videoId: data.id, url, privacy: data?.status?.privacyStatus || CFG.privacy };
      }
      lastErr = `HTTP ${put.status}: ${text.slice(0, 300)}`;
      if (put.status < 500) break;
    } catch (err: any) {
      lastErr = err?.message || String(err);
    }
    await sleep(attempt * 5000);
  }
  throw new PipelineError('youtube_upload', `Video upload to YouTube failed: ${lastErr}`);
}

// ---------------------------------------------------------------------------
// 2D animation & podcasts (anim/runner.ts) → publish → report back
// ---------------------------------------------------------------------------
async function produceAnimated(t0: number, pastTitles: string[]): Promise<number> {
  // Podcasts are always filmed in landscape. On a Short (vertical frame) the
  // landscape picture is placed in the middle of the vertical video.
  const landscapePodcast = CFG.category === 'podcast' && H >= W;
  const RW = landscapePodcast ? 1920 : W, RH = landscapePodcast ? 1080 : H;
  // A podcast hosted by the automation's own presenter keeps the presenter's usual voice.
  let hostVoice = '';
  if (CFG.podcastGuests || CFG.podcastSinger) {
    try { const cs = JSON.parse(String(CFG.castSpecs ? JSON.stringify(CFG.castSpecs) : '[]')); const g = cs[0]?.gender === 'male' ? 'male' : 'female'; hostVoice = (VOICES[g][CFG.hostCategory] || VOICES[g].default)[0]; } catch {}
  }
  const kit = {
    hostVoice,
    CFG, W: RW, H: RH, FPS, IS_SHORTS, WORK_DIR, HERE, LLM, extractJsonObject, log, reportStatus,
    run: (cmd: string, args: string[], o: { timeoutMs?: number } = {}) => run(cmd, args, o),
    probeDuration, pastTitles, PipelineError,
    composeMusic: (mood: string, seconds: number, seed: string) => composeBuffers(mood as any, Math.max(10, seconds), seed),
    automateLevel, eqForVoice,
    runStage: (o: { stageFile: string; job: any; audioFinal: string; files: Record<string, string>; duration: number }) => runStage({
      stageFile: o.stageFile, job: { ...o.job, wardrobeSeed: `${CFG.campaignId || CFG.campaignName || 'local'}:${CFG.partNumber}` }, duration: o.duration, files: o.files, size: { w: RW, h: RH },
      audio: { inputs: ['-i', o.audioFinal], filter: '[1:a]aresample=48000,apad[aout]' }
    })
  };
  const out = await runAnimated(kit as any);
  if (landscapePodcast) await landscapeIntoFrame(out.showName || '');
  const outDur = await probeDuration(OUTPUT_VIDEO);
  if (outDur < 3) throw new PipelineError('render_failed', `Rendered video is only ${outDur.toFixed(1)}s long.`);
  log(`Video: ${(fs.statSync(OUTPUT_VIDEO).size / 1e6).toFixed(1)} MB, ${outDur.toFixed(1)}s (${out.character}).`);
  const hashtagLine = youtubeHashtagLine([...out.hashtags, ...(IS_SHORTS ? ['shorts'] : [])]);
  const description = sanitizeYouTubeDescription([
    out.description,
    '🎬 Original animation, voices and music, made for this channel.',
    hashtagLine
  ].filter(Boolean).join('\n\n').trim());
  fs.writeFileSync(OUTPUT_META, JSON.stringify({ campaignId: CFG.campaignId, partNumber: CFG.partNumber, category: CFG.category, title: out.title, description, durationSec: outDur, model: out.model, createdAt: new Date().toISOString() }, null, 2));
  const libId = await saveToLibrary({ title: out.title, topic: CFG.topic || CFG.podcastAbout || CFG.subGenre, kind: CFG.category, durationSec: Math.round(outDur), width: W, height: H });
  const picked = await pickBestFrame(OUTPUT_VIDEO, outDur, out.highlights || []);
  const thumb = picked?.file || null;
  await saveLibraryThumbnail(libId, thumb);
  let published: { videoId: string; url: string; privacy: string } | null = null;
  const socialPublished: Record<string, string> = {};
  const publishErrors: Record<string, string> = {};
  if (!PUBLISH) {
    log('Publishing is OFF for this automation — the video is saved as a run artifact only.');
  } else if (CFG.dryRun) {
    log('Dry run — skipping publishing.');
  } else {
    let done = 0;
    const totalTargets = Number(WANT.youtube) + Number(WANT.facebook) + Number(WANT.instagram) + Number(WANT.threads);
    const publishStep = (platform: string) => Math.min(98, 82 + Math.round((done / Math.max(1, totalTargets)) * 16));
    const attemptPublish = async (platform: string, fn: () => Promise<string>, step: string, message: string) => {
      await reportStatus('running', step, publishStep(platform), message);
      let last: any = null;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const url = await fn();
          done++;
          return url;
        } catch (e: any) {
          last = e;
          if (attempt < 3) await new Promise((z) => setTimeout(z, 4000 * attempt));
        }
      }
      const msg = String(last?.message || last || `${platform} publishing failed`);
      publishErrors[platform] = msg.slice(0, 500);
      log(`❌ ${platform} publish failed after 3 attempts: ${msg}`);
      return '';
    };
    if (WANT.youtube) {
      const url = await attemptPublish('YouTube', async () => {
        const p = await uploadToYouTube({ title: out.title, description, tags: out.tags, synthetic: false });
        log(`Published on YouTube: ${p.url} (privacy: ${p.privacy})`);
        await setYouTubeThumbnail(p.videoId, thumb);
        await linkLibraryVideo(libId, p.url);
        published = p;
        return p.url;
      }, '5/5 Publishing to YouTube', `Uploading the ${IS_SHORTS ? 'Short' : 'video'} to YouTube…`);
      if (!url) log('YouTube did not publish; other selected platforms will still be attempted.');
    }
    if (WANT.facebook || WANT.instagram || WANT.threads) {
      const videoUrl = await publicVideoUrl(libId);
      const meta = { title: out.title, description, videoUrl };
      if (WANT.facebook) { socialPublished.facebook = await attemptPublish('Facebook', () => publishFacebookVideo(meta), '5/5 Publishing to Facebook', 'Uploading the finished video to Facebook…') || ''; if (!socialPublished.facebook) delete socialPublished.facebook; }
      if (WANT.instagram) { socialPublished.instagram = await attemptPublish('Instagram', () => publishInstagramVideo(meta), '5/5 Publishing to Instagram', 'Publishing the finished video as an Instagram Reel…') || ''; if (!socialPublished.instagram) delete socialPublished.instagram; }
      if (WANT.threads) { socialPublished.threads = await attemptPublish('Threads', () => publishThreadsVideo(meta), '5/5 Publishing to Threads', 'Publishing the finished video to Threads…') || ''; if (!socialPublished.threads) delete socialPublished.threads; }
    }
    if (!published && !Object.keys(socialPublished).length) {
      const firstError = Object.entries(publishErrors)[0]?.[1] || 'All selected publishing destinations rejected the video.';
      throw new PipelineError('publish_failed', firstError);
    }
    if (Object.keys(publishErrors).length) log(`⚠️ Partial publish: ${Object.keys(publishErrors).join(', ')} failed; successful destinations will be recorded so the same part is not posted twice.`);
  }
  const episode = await appRequest('POST', `${campaignPath()}/episodes`, {
    partNumber: CFG.partNumber, title: out.title, script: out.script, description,
    youtubeUrl: published?.url || '', videoId: published?.videoId || '', published: !!published || Object.keys(socialPublished).length > 0,
    socialUrls: socialPublished, socialErrors: publishErrors, privacyStatus: published?.privacy || '', format: CFG.format, aspectRatio: CFG.aspect,
    durationSec: Math.round(outDur), runId: CFG.runId, runUrl: CFG.runUrl, sources: out.sources, model: out.model
  });
  if (CFG.campaignId && CFG.appUrl && (!episode || episode.status >= 300)) console.warn(`⚠️ Could not record the episode in the app (${episode ? `HTTP ${episode.status}` : 'app unreachable'}).`);
  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  const links = [published?.url, ...Object.values(socialPublished)].filter(Boolean);
  await reportStatus('completed', links.length ? 'Published' : 'Video rendered', 100,
    links.length ? `✅ Part ${CFG.partNumber} published in ${secs}s: ${links.join(' · ')}` : `✅ Part ${CFG.partNumber} rendered in ${secs}s (not published).`,
    { youtubeUrl: published?.url || '', socialUrls: socialPublished, socialErrors: publishErrors });
  log(`Done in ${secs}s.`);
  return 0;
}

/**
 * Places the landscape podcast in the vertical frame: the full 16:9 picture in the
 * middle, a blurred, darkened copy filling the rest, the show name above it.
 */
async function landscapeIntoFrame(showName: string): Promise<void> {
  const src = path.join(WORK_DIR, 'landscape.mp4');
  fs.renameSync(OUTPUT_VIDEO, src);
  const font = path.join(HERE, 'assets/fonts/Poppins-Bold.ttf');
  const nameFile = path.join(WORK_DIR, 'showname.txt');
  fs.writeFileSync(nameFile, showName.toUpperCase().slice(0, 32));
  const fgH = Math.round((W * 9) / 16 / 2) * 2, top = Math.round((H - fgH) / 2);
  const text = showName && fs.existsSync(font)
    ? `,drawtext=fontfile='${font}':textfile='${nameFile}':fontcolor=white:fontsize=${Math.round(W * 0.06)}:x=(w-text_w)/2:y=${Math.max(20, top - Math.round(W * 0.12))}:shadowcolor=black@0.6:shadowx=2:shadowy=3`
    : '';
  const filter = `[0:v]split=2[a][b];[a]scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H},boxblur=28:2,eq=brightness=-0.22:saturation=1.15[bg];[b]scale=${W}:${fgH}[fg];[bg][fg]overlay=0:${top}${text}[v]`;
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-filter_complex', filter, '-map', '[v]', '-map', '0:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', OUTPUT_VIDEO], { timeoutMs: 20 * 60 * 1000 });
  if (r.code !== 0 || !fs.existsSync(OUTPUT_VIDEO)) {
    log(`⚠️ Could not place the landscape podcast in the vertical frame (${r.stderr.trim().split('\n').slice(-1)[0]}) — posting it as a landscape video.`);
    fs.renameSync(src, OUTPUT_VIDEO);
  } else log(`Landscape podcast placed in the ${W}x${H} frame.`);
}

// ---------------------------------------------------------------------------
// Thumbnail = the most engaging frame of the finished video. The whole video
// is decoded once at 4 frames a second (small), every frame is scored — sharp
// detail, colour, contrast, good brightness, not mid-cut or mid-flash — and
// moments the story marks as strong (a big reaction, a hit, a new picture)
// get a bonus. The winner is saved full size as a JPEG.
// ---------------------------------------------------------------------------
type Highlight = { t: number; w: number };
async function pickBestFrame(video: string, duration: number, highlights: Highlight[] = []): Promise<{ file: string; t: number } | null> {
  try {
    const portraitV = H > W;
    const sw = portraitV ? 108 : 192, sh = portraitV ? 192 : 108, FPSs = 4;
    const r = await run('ffmpeg', ['-v', 'error', '-i', video, '-vf', `fps=${FPSs},scale=${sw}:${sh}`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], { timeoutMs: 240000 });
    const size = sw * sh * 3, n = Math.floor(r.stdout.length / size);
    if (r.code !== 0 || n < 4) { log(`⚠️ Thumbnail: could not read the video frames (${r.stderr.trim().slice(-160)}).`); return null; }
    const luma = (f: number) => { const Y = new Float32Array(sw * sh); const o = f * size; for (let i = 0; i < sw * sh; i++) Y[i] = (0.299 * r.stdout[o + i * 3] + 0.587 * r.stdout[o + i * 3 + 1] + 0.114 * r.stdout[o + i * 3 + 2]) / 255; return Y; };
    const Ys = Array.from({ length: n }, (_, f) => luma(f));
    const diff = (a: Float32Array, b: Float32Array) => { let d = 0; for (let i = 0; i < a.length; i += 3) d += Math.abs(a[i] - b[i]); return d / (a.length / 3); };
    const stats = Ys.map((Y, f) => {
      const o = f * size;
      let sum = 0, sq = 0, rg = 0, rg2 = 0, yb = 0, yb2 = 0, lap = 0, lapC = 0, cnt = 0;
      for (let y = 1; y < sh - 1; y++) for (let x = 1; x < sw - 1; x++) {
        const i = y * sw + x, v = Y[i];
        sum += v; sq += v * v; cnt++;
        const l = Math.abs(4 * v - Y[i - 1] - Y[i + 1] - Y[i - sw] - Y[i + sw]);
        lap += l;
        // The middle of the frame is where the subject is.
        if (x > sw * 0.2 && x < sw * 0.8 && y > sh * 0.15 && y < sh * 0.75) lapC += l;
        const R = r.stdout[o + i * 3], G = r.stdout[o + i * 3 + 1], B = r.stdout[o + i * 3 + 2];
        const a1 = R - G, b1 = 0.5 * (R + G) - B; rg += a1; rg2 += a1 * a1; yb += b1; yb2 += b1 * b1;
      }
      const mean = sum / cnt, std = Math.sqrt(Math.max(0, sq / cnt - mean * mean));
      const sRG = Math.sqrt(Math.max(0, rg2 / cnt - (rg / cnt) ** 2)), sYB = Math.sqrt(Math.max(0, yb2 / cnt - (yb / cnt) ** 2));
      const color = Math.sqrt(sRG * sRG + sYB * sYB) + 0.3 * Math.sqrt((rg / cnt) ** 2 + (yb / cnt) ** 2);
      const motion = Math.max(f > 0 ? diff(Y, Ys[f - 1]) : 0, f < n - 1 ? diff(Y, Ys[f + 1]) : 0);
      return { t: f / FPSs, mean, std, color, detail: lap / cnt, centre: lapC / cnt, motion };
    });
    const endCard = duration > 12 ? 2.8 : 1.5;
    const ok = stats.filter((x) => x.t >= Math.min(1, duration * 0.1) && x.t <= duration - endCard);
    if (!ok.length) return null;
    const max = (k: 'detail' | 'centre' | 'color' | 'std') => Math.max(1e-6, ...ok.map((x) => x[k]));
    const mD = max('detail'), mC = max('centre'), mCol = max('color'), mS = max('std');
    const score = (x: typeof ok[number]) => {
      let s = 0.5 * x.detail / mD + 0.7 * x.centre / mC + 0.8 * x.color / mCol + 0.5 * x.std / mS;
      if (x.mean < 0.16) s -= (0.16 - x.mean) * 6; else if (x.mean > 0.85) s -= (x.mean - 0.85) * 6;
      if (x.motion > 0.1) s -= 1.5; else if (x.motion > 0.05) s -= (x.motion - 0.05) * 12; // cuts, flashes, blur
      for (const h of highlights) s += h.w * Math.exp(-((x.t - h.t - 0.35) ** 2) / (2 * 0.45 ** 2));
      return s;
    };
    const best = ok.reduce((a, b) => (score(b) > score(a) ? b : a));
    const out = path.join(WORK_DIR, 'thumbnail.jpg');
    // Full size (16:9 videos: 1280×720), under YouTube's 2 MB limit.
    const scale = portraitV ? 'scale=1080:-2' : 'scale=1280:-2';
    for (const q of [2, 4, 7]) {
      const e = await run('ffmpeg', ['-v', 'error', '-y', '-ss', best.t.toFixed(2), '-i', video, '-frames:v', '1', '-vf', scale, '-q:v', String(q), out], { timeoutMs: 60000 });
      if (e.code === 0 && fs.existsSync(out) && fs.statSync(out).size < 1.9e6) break;
    }
    if (!fs.existsSync(out)) return null;
    log(`Thumbnail: picked the frame at ${best.t.toFixed(1)}s of ${ok.length} candidates (${(fs.statSync(out).size / 1024).toFixed(0)} KB).`);
    return { file: out, t: best.t };
  } catch (e: any) { log(`⚠️ Thumbnail not picked (${e?.message || e}).`); return null; }
}

/** Keeps the thumbnail with the video in the app's Videos section. */
async function saveLibraryThumbnail(libId: string, file: string | null): Promise<void> {
  if (!libId || !file || !CFG.appUrl) return;
  try {
    const res = await fetch(`${CFG.appUrl}/api/videos/${libId}/thumbnail`, { method: 'PUT', headers: { 'Content-Type': 'image/jpeg', 'X-Animato-Runner-Key': CFG.runnerKey || '' }, body: fs.readFileSync(file), signal: AbortSignal.timeout(60000) });
    if (!res.ok) log(`⚠️ Thumbnail not saved to the Videos section (HTTP ${res.status}).`);
  } catch (e: any) { log(`⚠️ Thumbnail not saved to the Videos section (${e?.message || e}).`); }
}

/**
 * Sets the picked frame as the YouTube thumbnail. YouTube only accepts this
 * from verified channels; otherwise it keeps choosing a frame itself, and the
 * picked frame is still in the app's Videos section.
 */
async function setYouTubeThumbnail(videoId: string, file: string | null): Promise<void> {
  if (!file || !videoId) return;
  try {
    const token = await youtubeAccessToken();
    const r = await fetch(`${CFG.youtubeUploadBase}/thumbnails/set?videoId=${encodeURIComponent(videoId)}&uploadType=media`, {
      method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'image/jpeg' }, body: fs.readFileSync(file), signal: AbortSignal.timeout(60000)
    });
    if (r.ok) { log('Thumbnail set on YouTube ✔'); return; }
    const t = await r.text();
    log(/forbidden|verif|permission/i.test(t)
      ? 'Thumbnail: YouTube only takes custom thumbnails from verified channels, so it picks a frame itself for now (the picked frame is in the Videos section).'
      : `Thumbnail not set on YouTube (HTTP ${r.status}): ${t.slice(0, 200)}`);
  } catch (e: any) { log(`Thumbnail not set on YouTube (${e?.message || e}).`); }
}


// ---------------------------------------------------------------------------
// Musical videos — original ACE-Step song + real timed performance
// ---------------------------------------------------------------------------
interface MusicalSection { tag: string; lyrics: string; emotion: string }
interface MusicalSong {
  title: string;
  description: string;
  lyrics: string;
  sections: MusicalSection[];
  hashtags: string[];
  tags: string[];
}
const MUSICAL_PROFILES: Record<string, { style: string; emotion: string; instruments: string }> = {
  'sad / heartbreak': { style: 'cinematic emotional pop ballad, intimate lead vocal, piano, warm strings, restrained drums, spacious reverb', emotion: 'sad', instruments: 'piano and cinematic strings' },
  'happy / feel-good': { style: 'bright uplifting pop, catchy melodic hook, warm live drums, guitars, handclaps, polished radio mix', emotion: 'happy', instruments: 'bright guitars, handclaps and live drums' },
  'dance / party': { style: 'high-energy dance-pop, four-on-the-floor groove, glossy synths, punchy bass, festival-ready hook', emotion: 'excited', instruments: 'synths, punchy bass and dance drums' },
  'love / r&b': { style: 'smooth contemporary R&B, soulful lead vocal, electric piano, warm bass, tasteful percussion, intimate close-mic production', emotion: 'calm', instruments: 'electric piano and warm bass' },
  'pop anthem': { style: 'modern cinematic pop anthem, huge sing-along chorus, layered harmonies, drums, bass and wide synths', emotion: 'excited', instruments: 'wide synths and stadium drums' },
  'afrobeats': { style: 'contemporary Afrobeats, syncopated percussion, melodic bass, clean guitars, infectious hook, premium modern mix', emotion: 'happy', instruments: 'syncopated percussion, melodic bass and clean guitars' },
  'christian / gospel': { style: 'uplifting gospel worship, soulful lead, piano and organ, tasteful live drums, rich stacked choir responses, reverent and hopeful', emotion: 'calm', instruments: 'piano, organ, live drums and a stacked choir' },
  'muslim / nasheed': { style: 'respectful devotional nasheed-inspired contemporary song, warm lead vocal, organic frame-drum style percussion, melodic textures, restrained and reverent group response', emotion: 'calm', instruments: 'organic percussion and warm melodic textures' },
  'k-pop': { style: 'polished K-pop, explosive dance chorus, layered group vocals, punchy synth bass, crisp trap hats, bright pre-chorus build and a catchy post-chorus hook', emotion: 'excited', instruments: 'synth bass, trap hats and bright synths' },
  'hip-hop / rap': { style: 'modern hip-hop, confident rhythmic vocal flow with a sung hook, hard 808s, crisp hi-hats, dark melodic loop', emotion: 'serious', instruments: '808 bass, hi-hats and a melodic loop' },
  'rock': { style: 'anthemic rock, driving live drums, distorted guitars, powerful lead vocal and gang-vocal chorus', emotion: 'excited', instruments: 'electric guitars, bass and live rock drums' },
  'reggae / dancehall': { style: 'feel-good reggae / dancehall, offbeat skank guitar, deep bass, one-drop groove, sunny lead vocal', emotion: 'happy', instruments: 'skank guitar, deep bass and one-drop drums' },
  'amapiano': { style: 'amapiano, log-drum bass, shuffling shakers, airy piano chords, smooth soulful vocal and chant hooks', emotion: 'happy', instruments: 'log drums, shakers and piano chords' },
  'country': { style: 'modern country pop, acoustic and electric guitars, warm storytelling vocal, big singalong chorus', emotion: 'happy', instruments: 'acoustic guitar, pedal steel and live drums' },
  'jazz / soul': { style: 'smooth jazz-soul, warm vintage vocal, Rhodes piano, upright bass, brushed drums, horn stabs', emotion: 'calm', instruments: 'Rhodes, upright bass, brushed drums and horns' },
  'lo-fi / chill': { style: 'mellow lo-fi chill pop, soft intimate vocal, dusty keys, laid-back beat, warm tape texture', emotion: 'calm', instruments: 'dusty keys, soft beat and warm bass' },
  'edm / electronic': { style: 'festival EDM, euphoric build-up and drop, supersaw chords, pumping sidechain bass, soaring vocal', emotion: 'excited', instruments: 'supersaws, sidechained bass and four-on-the-floor kick' },
  'highlife': { style: 'West African highlife, sweet interlocking guitars, horns, bright percussion, joyful call-and-response vocals', emotion: 'happy', instruments: 'highlife guitars, horns and percussion' },
};
/** How each genre really sounds: tempo, key, groove, production and vocal delivery (fed to ACE-Step). */
const GENRE_FEEL: Record<string, { bpm: [number, number]; keys: string[]; tags: string; vocal: string }> = {
  'sad / heartbreak': { bpm: [68, 78], keys: ['A minor', 'D minor', 'E minor', 'F# minor'], tags: 'emotional ballad, slow tempo, minor key, soft piano intro, swelling strings in the chorus, sparse drums, wide reverb, heartfelt', vocal: 'breathy, vulnerable, emotional vocal with long held notes and a crack of pain on the hook' },
  'happy / feel-good': { bpm: [108, 122], keys: ['C major', 'G major', 'D major'], tags: 'feel-good pop, major key, bouncy groove, handclaps, bright guitars, uplifting chorus lift', vocal: 'bright, smiling, confident vocal with playful ad-libs' },
  'dance / party': { bpm: [120, 126], keys: ['A minor', 'F major', 'G minor'], tags: 'dance-pop, four-on-the-floor kick, pumping bass, build-up into an explosive chorus drop, club energy', vocal: 'energetic, punchy, chant-like vocal hook made for crowds' },
  'love / r&b': { bpm: [70, 88], keys: ['Eb major', 'Bb major', 'F minor'], tags: 'smooth R&B, laid-back groove, lush electric piano chords, deep warm bass, intimate', vocal: 'silky, sensual, soulful vocal with runs and harmonies' },
  'pop anthem': { bpm: [118, 128], keys: ['C major', 'D major', 'E major'], tags: 'stadium pop anthem, big drums, layered harmonies, huge singalong chorus, uplifting', vocal: 'powerful, soaring belted vocal on the hook' },
  'afrobeats': { bpm: [100, 112], keys: ['A minor', 'C major', 'G major', 'D minor'], tags: 'Afrobeats, Afropop, afro-swing groove, log drum and shekere percussion, syncopated afro drums, melodic bassline, highlife-tinged clean guitar riff, warm Lagos club vibe', vocal: 'smooth melodic Afrobeats vocal with Naija ad-libs, call-and-response hook, relaxed swagger' },
  'christian / gospel': { bpm: [72, 92], keys: ['Ab major', 'Db major', 'Eb major'], tags: 'gospel worship, Hammond organ, piano, choir responses, building to a powerful praise chorus', vocal: 'soulful gospel lead with passionate runs, answered by a choir' },
  'muslim / nasheed': { bpm: [80, 96], keys: ['D minor', 'A minor'], tags: 'nasheed-inspired devotional, frame drum and daf percussion, vocal harmonies, reverent and peaceful', vocal: 'warm, sincere, melodic vocal with group harmonies' },
  'k-pop': { bpm: [116, 128], keys: ['C# minor', 'F minor', 'A minor'], tags: 'K-pop, punchy synth bass, trap hats, explosive dance chorus, catchy post-chorus chant', vocal: 'crisp, polished idol-style vocal with a chanted hook' },
  'hip-hop / rap': { bpm: [84, 96], keys: ['C minor', 'F minor', 'G minor'], tags: 'modern hip-hop, hard 808 bass, crisp hi-hat rolls, dark melodic loop, head-nod groove', vocal: 'confident rhythmic flow with a sung melodic hook' },
  'rock': { bpm: [120, 140], keys: ['E minor', 'A major', 'D major'], tags: 'arena rock, driving drums, distorted guitar riff, big gang-vocal chorus', vocal: 'gritty, powerful rock vocal' },
  'reggae / dancehall': { bpm: [88, 100], keys: ['G major', 'A minor'], tags: 'reggae dancehall, one-drop groove, offbeat skank guitar, deep dub bass, sunny', vocal: 'relaxed, melodic island vocal with toasting ad-libs' },
  'amapiano': { bpm: [110, 115], keys: ['F minor', 'A minor', 'D minor'], tags: 'amapiano, deep log drum basslines, shakers, airy piano chords, slow-burn groove, South African club', vocal: 'smooth, soulful, chant-style vocal hooks' },
  'country': { bpm: [92, 110], keys: ['G major', 'D major', 'A major'], tags: 'modern country, acoustic guitar, pedal steel, warm storytelling, singalong chorus', vocal: 'warm storytelling country vocal' },
  'jazz / soul': { bpm: [76, 96], keys: ['Bb major', 'Eb major', 'F major'], tags: 'jazz soul, Rhodes, upright bass, brushed drums, horn section, smoky late-night club', vocal: 'velvet, expressive soul vocal with jazzy phrasing' },
  'lo-fi / chill': { bpm: [72, 86], keys: ['F major', 'C major'], tags: 'lo-fi chill pop, dusty keys, laid-back swing beat, warm tape saturation', vocal: 'soft, intimate, close-mic vocal' },
  'edm / electronic': { bpm: [124, 128], keys: ['F minor', 'A minor'], tags: 'festival EDM, supersaw chords, riser build-up, massive drop, sidechained bass', vocal: 'soaring, euphoric vocal hook' },
  'highlife': { bpm: [104, 118], keys: ['C major', 'G major'], tags: 'West African highlife, sweet interlocking guitars, brass horns, bright percussion, joyful', vocal: 'joyful, melodic vocal with call-and-response' },
};
/** This video's tempo / key / production for the genre (stable per video, varied between videos). */
function genreFeel(): { bpm: number; key: string; tags: string; vocal: string } {
  const k = String(CFG.subGenre || '').trim().toLowerCase();
  const g = GENRE_FEEL[k];
  const n = parseInt(crypto.createHash('md5').update(`${CFG.campaignId || CFG.campaignName}:${CFG.partNumber}:feel`).digest('hex').slice(0, 8), 16);
  if (!g) return { bpm: 0, key: '', tags: `${CFG.subGenre || 'pop'}, authentic to the genre, polished modern production`, vocal: 'expressive, emotional lead vocal' };
  return { bpm: g.bpm[0] + (n % (g.bpm[1] - g.bpm[0] + 1)), key: g.keys[Math.floor(n / 7) % g.keys.length], tags: g.tags, vocal: g.vocal };
}
function musicalProfile(): { style: string; emotion: string; instruments: string } {
  const k = String(CFG.subGenre || '').trim().toLowerCase();
  if (MUSICAL_PROFILES[k]) return MUSICAL_PROFILES[k];
  // A custom genre typed in the app: use it as the direction itself.
  const custom = String(CFG.subGenre || '').trim();
  if (custom) return { style: `${custom}, authentic to the genre, polished modern production, strong memorable chorus`, emotion: /sad|heart|blue|slow|ballad/i.test(custom) ? 'sad' : /dance|party|club|hype|energ/i.test(custom) ? 'excited' : 'happy', instruments: `the instruments typical of ${custom}` };
  return { style: 'cinematic contemporary pop, emotional lead vocal, polished commercial production', emotion: 'calm', instruments: 'piano, bass, drums and modern synth textures' };
}
/** ISO 639-1 code for the song language (music engines and transcription use it). */
const LANG_CODES: Record<string, string> = {
  english: 'en', french: 'fr', spanish: 'es', portuguese: 'pt', german: 'de', italian: 'it', dutch: 'nl', japanese: 'ja', korean: 'ko', chinese: 'zh', mandarin: 'zh', cantonese: 'zh',
  arabic: 'ar', hindi: 'hi', urdu: 'ur', bengali: 'bn', turkish: 'tr', russian: 'ru', polish: 'pl', swahili: 'sw', yoruba: 'yo', igbo: 'ig', hausa: 'ha', zulu: 'zu', xhosa: 'xh',
  amharic: 'am', indonesian: 'id', malay: 'ms', tagalog: 'tl', filipino: 'tl', vietnamese: 'vi', thai: 'th', greek: 'el', hebrew: 'he', persian: 'fa', farsi: 'fa', ukrainian: 'uk', 'nigerian pidgin': 'en', pidgin: 'en'
};
function musicLanguage(): { name: string; code: string } {
  const name = String(CFG.musicLanguage || 'English').trim() || 'English';
  return { name, code: LANG_CODES[name.toLowerCase()] || 'en' };
}
/** Lyric length that works for languages written without spaces (Japanese, Chinese, Thai…). */
function lyricWordCount(text: string): number {
  const spaced = text.split(/\s+/).filter(Boolean).length;
  const cjk = (text.match(/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0e00-\u0e7f]/g) || []).length;
  return Math.max(spaced, Math.round(cjk / 2));
}
function cleanMusicLyrics(v: string): string {
  return String(v || '').replace(/\r/g, '').replace(/^\s*\[[^\]]+\]\s*$/gm, '').replace(/["“”]/g, '').replace(/\n{3,}/g, '\n\n').trim();
}
function validMusicSections(v: any): MusicalSection[] {
  if (!Array.isArray(v)) return [];
  return v.map((x: any) => ({ tag: String(x?.tag || '').toLowerCase().replace(/[^a-z]/g, ''), lyrics: String(x?.lyrics || '').trim(), emotion: String(x?.emotion || '').toLowerCase().replace(/[^a-z_]/g, '') }))
    .filter((x) => ['intro','verse','prechorus','chorus','bridge','outro'].includes(x.tag) && x.lyrics.length >= 10);
}
function sectionLyricText(sections: MusicalSection[]): string {
  return sections.map((x) => `[${x.tag.toUpperCase()}]\n${x.lyrics}`).join('\n\n');
}
function musicHashtags(style: string): string[] {
  const out = ['originalmusic', 'musicvideo', String(style || '').toLowerCase().replace(/[^a-z0-9]+/g, '')];
  if (/christian|gospel/i.test(style)) out.push('gospelmusic', 'worshipmusic');
  if (/muslim|nasheed/i.test(style)) out.push('nasheed', 'devotionalmusic');
  if (/afrobeats/i.test(style)) out.push('afrobeats', 'afropop');
  if (IS_SHORTS) out.push('shorts');
  return Array.from(new Set(out.filter(Boolean))).slice(0, 10);
}
/**
 * What the song is ABOUT. A 30-second song only works if it says one clear, relatable thing, so every
 * video gets a concrete situation (picked fresh per video, matched to the genre's mood) instead of
 * vague "good vibes". The writer must build every line around it.
 */
const SONG_CONCEPTS: Record<string, string[]> = {
  party: [
    'Friday night after a long week: the singer finally switches off the phone and owns the dance floor',
    'the DJ drops the singer’s favourite song and the whole crowd sings it back',
    'a wedding reception where even the aunties outdance the young people',
    'a birthday where the singer celebrates surviving a hard year, not just getting older',
    'the first party after exams — free at last',
    'the light came back after a blackout and the whole street turns into a party',
    'payday weekend: no budget talk tonight, only dancing',
  ],
  love: [
    'falling for someone who makes even a traffic jam feel short',
    'loving someone through hard times — "I no get much, but my heart na your own"',
    'the shy moment of finally saying "I like you" after months of pretending',
    'a long-distance love counting down the days to the airport hug',
    'choosing the person who stayed when money was finished',
    'a love that feels like home-cooked food after a long trip',
  ],
  hustle: [
    'from sleeping on the floor to buying mama her first fridge',
    'the first payday after months of “no vacancy”',
    'proving the doubters wrong without saying a word — the work speaks',
    'grinding at night while the city sleeps, because tomorrow must be better',
    'turning a small market stall into a real business',
    'the day the hard work finally paid off and the family celebrated',
  ],
  heartbreak: [
    'deleting their number but still knowing it by heart',
    'smiling at the party while the heart is broken inside',
    'realising the love was one-sided and choosing to walk away with dignity',
    'the empty side of the bed and the song that still reminds you of them',
    'forgiving, but never going back',
  ],
  faith: [
    'thanking God for a door that opened when every other one was shut',
    'a mother’s prayers that carried the singer through',
    'peace in the storm — still standing after everything',
    'gratitude for small things: breath, family, another morning',
  ],
  uplift: [
    'telling a friend who wants to give up: your season is coming',
    'the confidence of finally loving yourself as you are',
    'starting over after failure, stronger than before',
    'home: the street, the food, the people that made the singer',
    'friendship that never switched up, from secondary school till now',
  ],
};
function songConcept(): { theme: string; concept: string } {
  const g = `${CFG.subGenre || ''} ${CFG.topic || ''}`.toLowerCase();
  const themes = /sad|heartbreak|ballad|blues/.test(g) ? ['heartbreak', 'love', 'uplift']
    : /gospel|worship|praise|nasheed|devotional|faith/.test(g) ? ['faith', 'uplift']
    : /love|r&b|rnb|soul|romantic/.test(g) ? ['love', 'love', 'heartbreak']
    : /hip|rap|drill|trap/.test(g) ? ['hustle', 'hustle', 'party']
    : /afro|amapiano|dance|party|edm|electro|dancehall|reggae|highlife|k-?pop|pop/.test(g) ? ['party', 'love', 'hustle', 'party', 'uplift']
    : ['love', 'party', 'hustle', 'uplift', 'heartbreak'];
  const n = parseInt(crypto.createHash('md5').update(`${CFG.campaignId || CFG.campaignName}:${CFG.partNumber}:concept`).digest('hex').slice(0, 8), 16);
  const theme = themes[n % themes.length], list = SONG_CONCEPTS[theme];
  return { theme, concept: list[Math.floor(n / themes.length) % list.length] };
}
/** Filler check: a song that keeps repeating empty words has no message. */
function fillerHeavy(lyrics: string): boolean {
  const words = String(lyrics ?? '').toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, ' ').split(/\s+/).filter(Boolean);
  if (words.length < 12) return true;
  const filler = words.filter((w) => /^(oh+|ah+|eh+|yeah+|yea|la+|na+|hey+|ooh+|woah+|whoa+|mm+|baby)$/.test(w)).length;
  const unique = new Set(words).size / words.length;
  return filler / words.length > 0.22 || unique < 0.38;
}
/**
 * The producer's pass: a hit songwriter reviews the draft (hook strength, rhyme and flow, story clarity,
 * genre authenticity, singability, line length) and returns an improved version with the same structure.
 */
async function polishSong(d: { title: string; hook: string; message: string; sections: MusicalSection[] }): Promise<{ title: string; sections: MusicalSection[] } | null> {
  const lang = musicLanguage(), f = genreFeel();
  const draft = d.sections.map((x) => `[${x.tag.toUpperCase()}]\n${x.lyrics}`).join('\n\n');
  const user = `You are the producer polishing this ${CFG.subGenre || 'pop'} song (${lang.name} lyrics) before it is recorded. Genre sound: ${f.tags}; vocal: ${f.vocal}${f.bpm ? `; ${f.bpm} BPM` : ''}.
Message: ${d.message || '(see lyrics)'} · Hook: ${d.hook || '(find the strongest one)'}

DRAFT:
${draft}

Make it a song people replay: (1) the hook is short, punchy and instantly memorable, and opens the chorus; (2) every line rhymes or flows naturally on the beat and is 4–8 words; (3) the verse tells one clear, concrete story that builds into the chorus; (4) it sounds authentically ${CFG.subGenre || 'pop'} — its slang, rhythm and attitude; (5) no filler, no clichés without meaning; (6) the outro pays the story off and lands the hook. Keep the SAME sections in the same order (same tags), the same language (${lang.name}) and roughly the same length. Return JSON only: {"title": "...", "sections": [{"tag": "...", "lyrics": "...", "emotion": "..."}]}`;
  for await (const a of LLM.attempts({ system: 'You are a Grammy-winning songwriter and producer. Original lyrics only.', user, saferUser: user, temperature: 0.7, maxTokens: 4000, json: true, task: 'musical_polish' })) {
    try {
      const j = extractJsonObject(a.text) as any;
      const secs = validMusicSections(j.sections);
      const tags = (x: MusicalSection[]) => x.map((y) => y.tag).join(',');
      if (secs.length && tags(secs) === tags(d.sections) && !fillerHeavy(secs.map((x) => x.lyrics).join('\n'))) {
        log(`🎵 Producer's pass: lyrics polished${j.title ? ` (“${String(j.title).slice(0, 60)}”)` : ''}.`);
        return { title: String(j.title || '').trim().slice(0, 90), sections: secs };
      }
    } catch {}
    break;   // one model try is enough — the draft is already good
  }
  return null;
}
async function writeMusicalSong(pastTitles: string[]): Promise<MusicalSong> {
  const profile = musicalProfile();
  const gender = CFG.gender === 'male' ? 'male' : 'female';
  const already = pastTitles.slice(-20).join(' | ');
  const lang = musicLanguage();
  const idea = songConcept();
  log(`🎵 Song idea (${idea.theme}): ${idea.concept}`);
  const meaning = `WHAT THE SONG IS ABOUT (mandatory): ${idea.concept}. Tell THIS story from the singer’s point of view. Every single line must belong to it — concrete people, places, objects and actions (a named street, mama’s kitchen, the bus stop, the DJ booth, a phone screen…), one clear emotion, and a message a listener can repeat in one sentence. No vague filler about “good vibes”, “just the good”, “feeling good tonight” with nothing behind it; at most one short ad-lib (e.g. “eh!”) per section. `;
  const user = `${meaning}Write one completely ORIGINAL ${IS_SHORTS ? 'short-form' : 'full-length'} song for a music video. LANGUAGE: write the title and ALL the lyrics in ${lang.name}${lang.name.toLowerCase() === 'english' ? '' : ` (natural, idiomatic ${lang.name} as native songwriters write it — not a translation; the description and tags may be in English)`}. Lead vocalist gender: ${gender}. Style/mood: ${CFG.subGenre || 'cinematic pop'}. Musical direction: ${profile.style}. Emotional performance must fit the style. The lyrics must be clean, singable, coherent and specific, with natural rhymes and a memorable chorus. ${CFG.musicSeconds <= 40 ? `The song is ONLY ${CFG.musicSeconds} SECONDS long, so every line must count. Use exactly this structure: INTRO (2 short lines that set the scene of the story — who, where, what just happened — so the listener is hooked and knows what the song is about), CHORUS (4 short, very catchy lines: the HOOK is one short memorable phrase (2–5 words) that states the song’s message, sung in line 1 and again in line 3 or 4; lines 2 and 4 add the feeling/detail and rhyme; this is most of the song), OUTRO (2 lines that land the story — a payoff, a twist or a punchline that answers the intro — ending on the hook). No verses, no bridge. Also return "hook" (the hook phrase) and "message" (the song’s point in one English sentence).` : CFG.musicSeconds <= 75 ? `The song is about ${CFG.musicSeconds} SECONDS long and must feel like a real hit single cut down to one minute, with a complete song structure. Write exactly five sections, in this order: INTRO (2 short lines — a call-out or vocal hook that sets the mood and hints at the story), VERSE (4 short lines that tell the story with concrete detail — who, where, what happened; natural rhymes), PRE-CHORUS (tag "prechorus": 2 short lines that lift the energy and build anticipation — a rising question, promise or turn — so the chorus explodes), CHORUS (4 short, very catchy lines: the HOOK is one short memorable phrase (2–5 words) that states the song’s message, sung in line 1 and repeated in line 3 or 4; lines 2 and 4 add the feeling and rhyme; simple, singable, built for people to sing along — the emotional release of the verse), OUTRO (2 lines that land the story — a payoff or twist — ending on the hook). Keep every line short (4–8 words) so it fits the beat. The CHORUS will be sung TWICE back to back. The emotion, rhythm and slang of every line must match the genre: ${(() => { const f = genreFeel(); return `${f.tags}; vocal: ${f.vocal}${f.bpm ? `; about ${f.bpm} BPM` : ''}`; })()}. Also return "hook" (the hook phrase) and "message" (the song’s point in one English sentence).` : 'Use this structure when it helps: intro, verse, pre-chorus, chorus, verse 2, pre-chorus, chorus, bridge, final chorus, outro.'} The chorus must be strong enough for a choir to answer behind the lead. Do not quote, adapt, imitate or reuse any copyrighted lyrics or named artist/song. Return JSON only with title, description, lyrics, sections (array of {tag,lyrics,emotion}), hashtags, tags. ${CFG.musicSeconds <= 75 ? `Use about ${Math.round(CFG.musicSeconds * 0.95)}–${Math.round(CFG.musicSeconds * 1.25)} lyric words in total across intro + verse + pre-chorus + chorus + outro (the chorus repeats, so the sung total is higher).` : `Aim for at least ${IS_SHORTS ? 90 : 220} lyric words.`} Previous titles to avoid repeating: ${already || '(none)'}${CFG.performanceNotes ? `\n\nWHAT IS WORKING ON THIS CHANNEL (learn from it — the mood, hook style and subjects of the best songs):\n${CFG.performanceNotes}` : ''}`;
  let last = '', fillerRejects = 0;
  for await (const a of LLM.attempts({
    system: 'You are a hit songwriter (think chart-topping Afrobeats/pop writers) and music-video creative director. A great short song tells one specific, relatable story with a hook people sing after one listen. Create original lyrics only. Never provide copyrighted lyrics. Make section labels and emotions explicit so a timed video renderer can stage the performance.',
    user, saferUser: user, temperature: 0.9, maxTokens: 6500, json: true, task: 'musical_songwriting'
  })) {
    try {
      const j = extractJsonObject(a.text) as any;
      const sections = validMusicSections(j.sections);
      let lyrics = cleanMusicLyrics(String(j.lyrics || sectionLyricText(sections)));
      if (!lyrics && sections.length) lyrics = sectionLyricText(sections);
      const hasChorus = sections.some((x) => x.tag === 'chorus');
      const words = lyricWordCount(lyrics);
      if (fillerRejects < 2 && fillerHeavy(lyrics)) { fillerRejects++; last = 'lyrics were mostly filler'; log('⚠️ Song draft rejected: lyrics were mostly filler words — rewriting.'); continue; }
      if (j.message) log(`🎵 Song message: ${String(j.message).slice(0, 160)}${j.hook ? ` · hook: “${String(j.hook).slice(0, 60)}”` : ''}`);
      if (String(j.title || '').trim() && words >= (CFG.musicSeconds <= 75 ? Math.round(CFG.musicSeconds * 0.6) : IS_SHORTS ? 70 : 150) && hasChorus) {
        // Too many words for the length → keep intro + chorus (+ outro) so it fits and stays in sync.
        if (CFG.musicSeconds <= 75 && words > CFG.musicSeconds * 2.4) { const allowed = ['intro', 'verse', 'prechorus', 'chorus', 'outro']; const seen = new Set<string>(); const keep = sections.filter((x) => allowed.includes(x.tag) && !seen.has(x.tag) && (seen.add(x.tag), true)); if (keep.some((x) => x.tag === 'chorus')) { sections.splice(0, sections.length, ...keep); lyrics = cleanMusicLyrics(sectionLyricText(sections)); } }
        // Producer's pass: review the draft like a hit-maker and rewrite it once (hook, rhyme, flow, story, genre).
        const polished = await polishSong({ title: String(j.title), hook: String(j.hook || ''), message: String(j.message || ''), sections: sections.map((x) => ({ ...x })) }).catch(() => null);
        if (polished) {
          sections.splice(0, sections.length, ...polished.sections);
          lyrics = cleanMusicLyrics(sectionLyricText(sections));
          if (polished.title) j.title = polished.title;
        }
        // One-minute song: intro → verse → pre-chorus → chorus → chorus again (the hook sticks) → outro.
        if (CFG.musicSeconds <= 75) {
          const intro = sections.find((x) => x.tag === 'intro'), verse = sections.find((x) => x.tag === 'verse'), pre = sections.find((x) => x.tag === 'prechorus'), chorus = sections.find((x) => x.tag === 'chorus'), outro = sections.find((x) => x.tag === 'outro');
          if (chorus) {
            const shaped = [intro, verse, pre, chorus, { ...chorus, emotion: chorus.emotion || 'excited' }, outro].filter(Boolean) as MusicalSection[];
            sections.splice(0, sections.length, ...shaped);
            lyrics = cleanMusicLyrics(sectionLyricText(sections));
            log(`🎵 Song shape: ${sections.map((x) => x.tag).join(' → ')} (${lyricWordCount(lyrics)} sung words).`);
          }
        }
        return {
          title: String(j.title).trim().slice(0, 90),
          description: String(j.description || `Original ${CFG.subGenre || 'musical'} song performed by a ${gender} lead singer with a cinematic stage and choir chorus.`).trim().slice(0, 1200),
          lyrics, sections, hashtags: Array.isArray(j.hashtags) ? j.hashtags.map((x: any) => String(x).replace(/^#+/, '').trim()).filter(Boolean).slice(0, 10) : musicHashtags(CFG.subGenre),
          tags: Array.isArray(j.tags) ? j.tags.map((x: any) => String(x).trim()).filter(Boolean).slice(0, 20) : ['original music', CFG.subGenre || 'music video', 'ai music']
        };
      }
      last = `invalid songwriter output from ${a.provider}/${a.model}`;
    } catch (e: any) { last = e?.message || String(e); }
  }
  throw new PipelineError('musical_song_retry', `The AI model on the runner could not write a usable song this time (${last || 'no valid answer'}). Nothing was posted; the next scheduled run will retry.`);
}
/** ACE-Step 1.5 (open-source, MIT) — the only music engine — runs on the GitHub runner itself:
 *  its own REST server (/release_task → /query_result → download), never an online API. */
let aceDeadline = 0, aceStarted = 0;
const mmss = (ms: number) => `${Math.floor(ms / 60000)}:${String(Math.floor((ms % 60000) / 1000)).padStart(2, '0')}`;
/** Shows what ACE-Step is doing on the automation card, so a slow song never looks frozen. */
async function aceStatus(what: string) {
  const el = Date.now() - aceStarted;
  try { await reportStatus('running', `2/6 Generating music (ACE-Step on the runner) · ${mmss(el)} of max ${CFG.acestepLocalMin}:00`, Math.min(40, 24 + Math.round((el / (CFG.acestepLocalMin * 60000)) * 16)), `🎵 ${what} (${mmss(el)} elapsed)`); } catch {}
}
const aceLeft = () => Math.max(0, aceDeadline - Date.now());
/**
 * ACE-Step everywhere: the hosted API first; when it is down / overloaded (Cloudflare 504s), the SAME
 * open-source model runs on the GitHub runner's CPU (installed in the background while the hosted API
 * is tried, and cached between runs), through ACE-Step's own REST server (/release_task → /query_result).
 */
let aceLocalRun = false;
/** ACE-Step runs ONLY on the GitHub runner (no online API): installed once, cached between runs. */
async function generateAceStepMusic(song: MusicalSong): Promise<{ file: string; duration: number; taskId?: string }> {
  return aceLocalGenerate(song, prepareLocalAce());
}
let aceLocalPrep: Promise<{ base: string; stop: () => void; lm: boolean }> | null = null;
let aceLocalWantLm = true;
/** The local server's own key (sent as both the Bearer header and ai_token, which it checks). */
const ACE_LOCAL_KEY = `animato-${crypto.randomBytes(8).toString('hex')}`;
function prepareLocalAce(): Promise<{ base: string; stop: () => void; lm: boolean }> {
  if (aceLocalPrep) return aceLocalPrep;
  aceLocalPrep = (async () => {
    const t0 = Date.now();
    const root = path.join(os.homedir(), 'acestep-local'), repo = path.join(root, 'repo'), venv = path.join(root, 'venv'), py = path.join(venv, 'bin', 'python');
    const ckpt = path.join(repo, 'checkpoints');
    const script = `set -e
mkdir -p "${root}"
if [ ! -d "${repo}/acestep" ]; then git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "${repo}"; fi
if [ ! -x "${py}" ]; then python3 -m venv "${venv}"; fi
if ! "${py}" -c "import acestep, torch, diffusers, transformers" 2>/dev/null; then
  "${py}" -m pip install -q --disable-pip-version-check --upgrade pip
  "${py}" -m pip install -q --disable-pip-version-check torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  "${py}" -m pip install -q --disable-pip-version-check "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "matplotlib>=3.7.5" "scipy>=1.10.1" "soundfile>=0.13.1" "loguru>=0.7.3" "einops>=0.8.1" "accelerate>=1.12.0" "fastapi>=0.110.0" diskcache "uvicorn[standard]>=0.27.0" "numba>=0.63.1" "vector-quantize-pytorch>=1.27.15" "torchao>=0.16.0,<0.17.0" toml "peft>=0.18.0" lycoris-lora modelscope "typer-slim>=0.21.1" "pytorch-wavelets>=1.3.0" "pywavelets>=1.9.0" "setuptools<72" huggingface_hub "gradio==6.2.0" "lightning>=2.0.0" "tensorboard>=2.20.0"
  "${py}" -m pip install -q --disable-pip-version-check --no-deps -e "${repo}"
fi
# Only what text-to-music needs (turbo DiT, VAE, text encoder) — not the 1.7B planning LM.
if [ ! -d "${ckpt}/acestep-v15-turbo" ] || [ ! -d "${ckpt}/vae" ] || [ ! -d "${ckpt}/Qwen3-Embedding-0.6B" ]; then
  "${py}" -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5', local_dir='${ckpt}', allow_patterns=['acestep-v15-turbo/*','vae/*','Qwen3-Embedding-0.6B/*','*.json','*.txt'])"
fi
# The small planning LM ("thinking": plans the song's sections and melody before the music is made).
if [ ! -d "${ckpt}/acestep-5Hz-lm-0.6B" ]; then
  "${py}" -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-0.6B', local_dir='${ckpt}/acestep-5Hz-lm-0.6B')" || echo "LM download failed (songs will be made without planning)"
fi
du -sh "${root}" 2>/dev/null || true
`;
    const scriptFile = path.join(WORK_DIR, 'acestep_local_setup.sh');
    fs.writeFileSync(scriptFile, script);
    log('🎵 Preparing ACE-Step on this runner (background; cached after the first run)…');
    const r = await run('bash', [scriptFile], { timeoutMs: 22 * 60000 });
    if (r.code !== 0) throw new Error(`local ACE-Step setup failed: ${r.stderr.split('\n').filter(Boolean).slice(-4).join(' | ').slice(0, 400)}`);
    log(`🎵 Local ACE-Step installed in ${((Date.now() - t0) / 1000).toFixed(0)}s (${r.stdout.toString().trim().split('\n').pop() || ''}).`);
    const useLm = aceLocalWantLm && fs.existsSync(path.join(ckpt, 'acestep-5Hz-lm-0.6B'));
    const port = 8011, threads = String(Math.max(1, os.cpus().length));
    const env = { ...process.env, ACESTEP_API_HOST: '127.0.0.1', ACESTEP_API_PORT: String(port), ACESTEP_API_KEY: ACE_LOCAL_KEY, ACESTEP_DEVICE: 'cpu', ACESTEP_INIT_LLM: useLm ? 'true' : 'false', ACESTEP_LM_BACKEND: 'pt', ACESTEP_LM_MODEL_PATH: 'acestep-5Hz-lm-0.6B', ACESTEP_LM_DEVICE: 'cpu',
      ACESTEP_USE_FLASH_ATTENTION: 'false', ACESTEP_CONFIG_PATH: 'acestep-v15-turbo', ACESTEP_PROJECT_ROOT: repo, ACESTEP_CHECKPOINTS_DIR: ckpt, OMP_NUM_THREADS: threads, MKL_NUM_THREADS: threads, TOKENIZERS_PARALLELISM: 'false',
      // ACE-Step stops a generation after 600 s by default (a GPU figure); CPUs need longer.
      ACESTEP_GENERATION_TIMEOUT: '3600' };
    const srv = spawn(py, ['-m', 'acestep.api_server'], { cwd: repo, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '';
    const keep = (d: Buffer) => { tail = (tail + d.toString()).slice(-4000); };
    srv.stdout.on('data', keep); srv.stderr.on('data', keep);
    let exited = false; srv.on('close', () => { exited = true; });
    const base = `http://127.0.0.1:${port}`;
    const until = Date.now() + 12 * 60000;
    while (Date.now() < until) {
      if (exited) throw new Error(`local ACE-Step server stopped: ${tail.split('\n').filter(Boolean).slice(-4).join(' | ').slice(0, 400)}`);
      try { const h = await fetch(`${base}/health`, { signal: AbortSignal.timeout(5000) }); if (h.ok) break; } catch {}
      await sleep(3000);
    }
    if (Date.now() >= until) { try { srv.kill('SIGKILL'); } catch {} throw new Error('local ACE-Step server did not start within 12 min'); }
    log(`🎵 Local ACE-Step server ready in ${((Date.now() - t0) / 1000).toFixed(0)}s${useLm ? ' (with the song-planning model)' : ''}.`);
    return { base, lm: useLm, stop: () => { try { srv.kill('SIGKILL'); } catch {} aceLocalPrep = null; } };
  })();
  return aceLocalPrep;
}
async function aceLocalGenerate(song: MusicalSong, prep: Promise<{ base: string; stop: () => void; lm: boolean }>): Promise<{ file: string; duration: number; taskId?: string }> {
  aceStarted = Date.now();
  aceDeadline = aceStarted + CFG.acestepLocalMin * 60000;
  const once = async (p: Promise<{ base: string; stop: () => void; lm: boolean }>) => {
    await aceStatus('setting up ACE-Step on the runner (CPU)');
    const beat = setInterval(() => { aceStatus('ACE-Step on the runner: installing / loading the model'); }, 30000);
    let srv: { base: string; stop: () => void; lm: boolean };
    try { srv = await p; } finally { clearInterval(beat); }
    const saved = CFG.acestepBase;
    (CFG as any).acestepBase = srv.base;
    aceLocalRun = true;
    try {
      const what = `ACE-Step on the runner is ${srv.lm ? 'planning and ' : ''}making the song (CPU)`;
      await aceStatus(what);
      const beat2 = setInterval(() => { aceStatus(what); }, 30000);
      try { return await aceStepNative(song, ACE_LOCAL_KEY, !srv.lm); }   // with the planning LM: thinking on
      finally { clearInterval(beat2); }
    } finally {
      (CFG as any).acestepBase = saved;
      aceLocalRun = false;
      srv.stop();   // free the RAM for the stems + the video render
    }
  };
  try { return await once(prep); }
  catch (e: any) {
    if (!aceLocalWantLm || aceLeft() < 5 * 60000) throw e;
    // The planning model failed (or the server could not load it): make the song without it.
    log(`⚠️ ACE-Step with the planning model failed (${String(e?.message || e).slice(0, 200)}) — trying again without it.`);
    aceLocalWantLm = false;
    aceLocalPrep = null;
    return once(prepareLocalAce());
  }
}
/** Set on a regeneration when the first song came out in the wrong voice. */
let aceGenderBoost = false;
/** When the hosted service is too slow for a full song, a shorter song (it renders faster) is asked for. */
let aceLengthScale = 1;
const aceFullTarget = () => CFG.musicSeconds;
/** Shorten the song in place to fit `seconds`: whole sections in order (always keeping the first chorus). */
function trimSongTo(song: MusicalSong, seconds: number) {
  if (!song.sections.length) return;
  const wps = 2.1, budget = Math.max(30, seconds * wps * 0.92);
  const out: MusicalSection[] = [];
  let words = 0, hasChorus = false;
  for (const sec of song.sections) {
    const n = sec.lyrics.split(/\s+/).filter(Boolean).length;
    if (out.length && words + n > budget && hasChorus) break;
    if (out.length && words + n > budget * 1.25) break;
    out.push(sec); words += n;
    if (sec.tag === 'chorus') hasChorus = true;
  }
  if (out.length < song.sections.length) {
    song.sections = out;
    song.lyrics = out.map((x) => x.lyrics).join('\n\n');
    log(`🎵 Song shortened to ${out.length} section(s) (~${words} words) for a ${Math.round(seconds)}s take.`);
  }
}
function aceStepRequest(song: MusicalSong) {
  const profile = musicalProfile();
  const gender = CFG.gender === 'male' ? 'male' : 'female';
  const target = Math.max(30, Math.round(aceFullTarget() * aceLengthScale));
  const lyrics = song.sections.length ? song.sections.map((x) => `[${x.tag === 'prechorus' ? 'Pre-Chorus' : x.tag[0].toUpperCase() + x.tag.slice(1)}]\n${x.lyrics}`).join('\n\n') : cleanMusicLyrics(song.lyrics);
  const voice = gender === 'female'
    ? 'female vocals, solo female singer, woman lead voice, feminine vocal tone'
    : 'male vocals, solo male singer, man lead voice, masculine vocal tone';
  const feel = genreFeel();
  const prompt = [feel.tags, voice + (aceGenderBoost ? (gender === 'female' ? ', high female voice, no male vocals' : ', deep male voice, no female vocals') : ''), feel.vocal,
    feel.bpm ? `${feel.bpm} BPM` : '', feel.key, profile.instruments, `sung in ${musicLanguage().name}`, `${profile.emotion} mood`,
    'clear song structure: short instrumental intro, catchy repeated chorus hook, satisfying ending', `${gender} backing harmonies on the chorus`, 'professional studio mix'].filter(Boolean).join(', ');
  return { prompt, lyrics, target, lang: musicLanguage().code, bpm: feel.bpm, key: feel.key };
}
const aceErrCode = (status: number, body: string) => status === 401 || status === 403 ? 'acestep_auth' : status === 429 || /quota|limit|credit|insufficient/i.test(body) ? 'acestep_quota' : status >= 500 || status === 408 ? 'acestep_busy' : 'acestep_failed';
async function saveAceAudio(buf: Buffer, label: string) {
  const out = path.join(WORK_DIR, `acestep_${crypto.randomBytes(4).toString('hex')}.mp3`);
  fs.writeFileSync(out, buf);
  if (fs.statSync(out).size < 10000) throw new PipelineError('acestep_busy', `ACE-Step (${label}) returned an empty audio clip.`);
  return { file: out, duration: await probeDuration(out) };
}
async function aceStepNative(song: MusicalSong, apiKey: string, fast: boolean): Promise<{ file: string; duration: number; taskId?: string }> {
  const r = aceStepRequest(song);
  const headers = { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json; charset=utf-8' };
  // "thinking" (the 5Hz planning LM) gives better songs but is slower; the retry rounds switch it off.
  const body: any = {
    prompt: r.prompt, caption: r.prompt, lyrics: r.lyrics, audio_duration: r.target, audio_format: 'mp3', batch_size: 1, vocal_language: r.lang,
    ...(r.bpm ? { bpm: r.bpm } : {}), ...(r.key ? { key_scale: r.key } : {}), time_signature: '4',
    thinking: !fast, use_cot_caption: false, ai_token: apiKey, ...(CFG.acestepModel ? { model: CFG.acestepModel } : {}),
  };
  let taskId = '';
  for (let attempt = 0; attempt < 3 && !taskId; attempt++) {
    if (attempt) await sleep(8000 * attempt);
    try {
      const rel = await fetch(`${CFG.acestepBase}/release_task`, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(180000) });
      const txt = await rel.text();
      let rj: any = {}; try { rj = JSON.parse(txt); } catch {}
      taskId = String(rj?.data?.task_id || rj?.task_id || '').trim();
      if (!rel.ok || !taskId) {
        const code = aceErrCode(rel.status, txt);
        const err = new PipelineError(code, `ACE-Step request failed (HTTP ${rel.status}): ${String(rj?.error || rj?.message || txt).slice(0, 200)}`);
        if (code !== 'acestep_busy' || attempt === 2) throw err;
        log(`⚠️ ${err.message} — retrying…`); taskId = '';
      }
    } catch (e: any) {
      if (e instanceof PipelineError) throw e;
      if (attempt === 2) throw new PipelineError('acestep_busy', `ACE-Step did not accept the song request (${e?.message || e}).`);
      log(`⚠️ ACE-Step request attempt ${attempt + 1} failed (${e?.message || e}); retrying…`);
    }
  }
  log(`🎵 ACE-Step task ${taskId.slice(0, 24)} accepted${fast ? ' (fast mode)' : ''}; waiting for the song…`);
  const started = Date.now();
  for (let attempt = 0; Date.now() - started < Math.min(aceLocalRun ? 60 * 60 * 1000 : 15 * 60 * 1000, aceLeft()); attempt++) {
    await sleep(attempt < 6 ? 5000 : 8000);
    let q: Response;
    try { q = await fetch(`${CFG.acestepBase}/query_result`, { method: 'POST', headers, body: JSON.stringify({ task_id_list: [taskId] }), signal: AbortSignal.timeout(45000) }); }
    catch { continue; }
    const qj: any = await q.json().catch(() => ({}));
    if (!q.ok) { if (q.status === 429 || q.status >= 500) continue; throw new PipelineError(aceErrCode(q.status, JSON.stringify(qj)), `ACE-Step status check failed (HTTP ${q.status}).`); }
    const item = Array.isArray(qj?.data) ? qj.data[0] : Array.isArray(qj) ? qj[0] : qj?.data;
    const status = Number(item?.status ?? 0);
    if (status === 2) throw new PipelineError('acestep_busy', `ACE-Step could not generate the song: ${String(item?.error || item?.result || 'failed').slice(0, 200)}`);
    if (status !== 1) { if (attempt % 8 === 0) log(`🎵 ACE-Step still generating (${Math.round((Date.now() - started) / 1000)}s)…`); continue; }
    let res: any = item?.result;
    if (typeof res === 'string') { try { res = JSON.parse(res); } catch { res = []; } }
    const first = Array.isArray(res) ? res[0] : res;
    const fileRef = String(first?.file || first?.audio_url || first?.url || '');
    if (!fileRef) throw new PipelineError('acestep_busy', 'ACE-Step finished but returned no audio file.');
    if (/^data:audio/i.test(fileRef)) return { ...(await saveAceAudio(Buffer.from(fileRef.split(',')[1] || '', 'base64'), 'async')), taskId };
    const url = /^https?:\/\//i.test(fileRef) ? fileRef : `${CFG.acestepBase}${fileRef.startsWith('/') ? '' : '/'}${fileRef}`;
    for (let d = 0; d < 3; d++) {
      try {
        const dl = await fetch(url, { headers: { Authorization: `Bearer ${apiKey}` }, signal: AbortSignal.timeout(240000) });
        if (!dl.ok) throw new Error(`HTTP ${dl.status}`);
        return { ...(await saveAceAudio(Buffer.from(await dl.arrayBuffer()), 'async')), taskId };
      } catch (e: any) { if (d === 2) throw new PipelineError('acestep_busy', `ACE-Step audio download failed: ${e?.message || e}`); await sleep(5000); }
    }
  }
  throw new PipelineError('acestep_busy', `ACE-Step task ${taskId} did not finish in time (${Math.round((Date.now() - started) / 60000)} min).`);
}
/** Google Lyria via the Gemini API (Interactions). PAID — only used when LYRIA_ENABLED=true and a billing-enabled Gemini key is set. */
async function generateLyriaMusic(song: MusicalSong): Promise<{ file: string; duration: number; taskId?: string }> {
  if (!CFG.lyriaEnabled) throw new PipelineError('lyria_disabled', 'Lyria is off (set LYRIA_ENABLED=true; it needs a billing-enabled Gemini key).');
  if (!CFG.geminiKeys.length) throw new PipelineError('lyria_missing_key', 'No Gemini API key is configured for Lyria.');
  const profile = musicalProfile();
  const gender = CFG.gender === 'male' ? 'male' : 'female';
  const len = `about ${CFG.musicSeconds} seconds long`;
  const input = `Create a ${profile.style} song in ${musicLanguage().name}, ${len}, with a ${gender} lead vocalist singing with ${profile.emotion} emotion over ${profile.instruments}, and a backing choir on every chorus. Use exactly these lyrics:\n\n${sectionLyricText(song.sections.length ? song.sections : [{ tag: 'verse', lyrics: cleanMusicLyrics(song.lyrics), emotion: profile.emotion }])}`;
  let lastErr = '';
  for (const key of CFG.geminiKeys) {
    const r = await fetch('https://generativelanguage.googleapis.com/v1beta/interactions', {
      method: 'POST', headers: { 'x-goog-api-key': key, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: CFG.lyriaModel, input }), signal: AbortSignal.timeout(300000),
    }).catch((e) => { lastErr = String(e?.message || e); return null; });
    if (!r) continue;
    const j: any = await r.json().catch(() => ({}));
    if (!r.ok) { lastErr = `HTTP ${r.status}: ${JSON.stringify(j?.error || j).slice(0, 240)}`; continue; }
    const steps = Array.isArray(j?.steps) ? j.steps : Array.isArray(j?.outputs) ? j.outputs : [];
    let b64 = '';
    for (const st of steps) for (const c of (Array.isArray(st?.content) ? st.content : [st])) if (c?.type === 'audio' && c?.data) { b64 = String(c.data); break; }
    if (!b64) { lastErr = 'Lyria returned no audio'; continue; }
    const out = path.join(WORK_DIR, `lyria_${crypto.randomBytes(4).toString('hex')}.mp3`);
    fs.writeFileSync(out, Buffer.from(b64, 'base64'));
    if (fs.statSync(out).size < 10000) { lastErr = 'Lyria returned an empty clip'; continue; }
    return { file: out, duration: await probeDuration(out) };
  }
  throw new PipelineError('lyria_failed', `Lyria music generation failed: ${lastErr || 'unknown error'}`);
}

const MUSIC_ENGINE_LABEL: Record<string, string> = { acestep: 'ACE-Step 1.5 (on the runner)', lyria: 'Google Lyria' };
/** ACE-Step is the music engine. (Google Lyria is only tried if LYRIA_ENABLED=true with a paid Gemini key.) */
async function generateMusicTrack(song: MusicalSong): Promise<{ file: string; duration: number; engine: string }> {
  const available = ['acestep', 'lyria'].filter((p) => (p === 'acestep' ? true : CFG.lyriaEnabled && CFG.geminiKeys.length > 0));
  const errors: string[] = [];
  for (const p of available) {
    try {
      await reportStatus('running', `2/6 Generating music (${MUSIC_ENGINE_LABEL[p]})`, 24, `Generating “${song.title}” with ${MUSIC_ENGINE_LABEL[p]}…`);
      const g = p === 'acestep' ? await generateAceStepMusic(song) : await generateLyriaMusic(song);
      if (g.duration < 5) throw new PipelineError('music_too_short', `${MUSIC_ENGINE_LABEL[p]} returned only ${g.duration.toFixed(1)}s of audio.`);
      log(`🎵 Music generated with ${MUSIC_ENGINE_LABEL[p]}: ${g.duration.toFixed(1)}s.`);
      return { file: g.file, duration: g.duration, engine: p };
    } catch (e: any) {
      const msg = String(e?.message || e).slice(0, 300);
      errors.push(`${MUSIC_ENGINE_LABEL[p]}: ${msg}`);
      log(`⚠️ ${MUSIC_ENGINE_LABEL[p]} failed (${msg}).`);
    }
  }
  throw new PipelineError('music_all_failed', `Music generation failed — ${errors.join(' | ')}. Nothing was posted; the next run will retry.`);
}

/**
 * Word-level lyric timing with Whisper on THIS runner (faster-whisper, CPU int8) — no API keys,
 * no outside transcription service.
 */
async function transcribeLocal(file: string): Promise<Word[]> {
  const python = ENV.PYTHON || 'python3';
  const chk = await run(python, ['-c', 'import faster_whisper'], { timeoutMs: 60000 });
  if (chk.code !== 0) {
    const ins = await run(python, ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', 'faster-whisper'], { timeoutMs: 10 * 60000 });
    if (ins.code !== 0) { log(`⚠️ faster-whisper could not be installed (${ins.stderr.slice(-200)}).`); return []; }
  }
  const py = path.join(WORK_DIR, 'lyrics_asr.py');
  fs.writeFileSync(py, `import json,sys,os
from faster_whisper import WhisperModel
m=WhisperModel("small",device="cpu",compute_type="int8",cpu_threads=os.cpu_count() or 2)
lang=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] not in ("","unknown","pcm") else None
segs,_=m.transcribe(sys.argv[1],word_timestamps=True,vad_filter=False,language=lang)
out=[]
for s in segs:
  for w in (s.words or []): out.append({"text":w.word.strip(),"start":w.start,"end":w.end})
print(json.dumps(out))
`);
  const r = await run(python, [py, file, musicLanguage().code], { timeoutMs: 20 * 60000 });
  if (r.code !== 0) { log(`⚠️ Local lyric transcription failed (${r.stderr.slice(-200)}).`); return []; }
  try {
    const list = JSON.parse(r.stdout.toString().trim().split('\n').pop() || '[]');
    return (Array.isArray(list) ? list : []).filter((w: any) => w.text && Number.isFinite(w.start) && Number.isFinite(w.end));
  } catch { return []; }
}

// ---------------------------------------------------------------------------
// Stem separation (Demucs, open-source, runs on the GitHub runner's CPU).
// Vocals → lip sync + "is she singing right now" + voice check; drums → drummer; other → pianist.
// Optional: any failure falls back to the full mix.
// ---------------------------------------------------------------------------
interface Stems { vocals: string; drums?: string; other?: string; bass?: string }
let demucsReady: boolean | null = null;
async function separateStems(file: string, duration: number): Promise<Stems | null> {
  if ((ENV.STEM_SEPARATION || '').toLowerCase() === 'off') return null;
  const python = ENV.PYTHON || 'python3';
  const t0 = Date.now();
  try {
    if (demucsReady === null) {
      const chk = await run(python, ['-c', 'import demucs, torch, soundfile'], { timeoutMs: 90000 });
      if (chk.code !== 0) {
        await reportStatus('running', '2/6 Preparing the vocal separator', 30, 'Installing the open-source vocal/drum separator (first run only, ~1–2 min)…');
        const pipArgs = ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-warn-script-location'];
        const a = await run(python, [...pipArgs, 'torch==2.3.1', 'torchaudio==2.3.1', '--index-url', 'https://download.pytorch.org/whl/cpu'], { timeoutMs: 8 * 60000 });
        if (a.code !== 0) throw new Error(`torch install failed: ${a.stderr.slice(-300)}`);
        const b = await run(python, [...pipArgs, 'demucs==4.0.1', 'soundfile', 'numpy<2'], { timeoutMs: 6 * 60000 });
        if (b.code !== 0) throw new Error(`demucs install failed: ${b.stderr.slice(-300)}`);
      }
      demucsReady = true;
    }
    if (!demucsReady) return null;
    await reportStatus('running', '2/6 Separating vocals, drums and keys', 34, 'Splitting the song into vocals / drums / instruments for exact lip sync and band timing…');
    const out = path.join(WORK_DIR, 'stems');
    const r = await run(python, ['-m', 'demucs', '-n', 'htdemucs', '-d', 'cpu', '-j', '2', '-o', out, file], { timeoutMs: Math.round((4 + (duration / 60) * 4) * 60000) });
    const dir = path.join(out, 'htdemucs', path.basename(file, path.extname(file)));
    const pathOf = (n: string) => { const f = path.join(dir, `${n}.wav`); return fs.existsSync(f) && fs.statSync(f).size > 10000 ? f : undefined; };
    const vocals = pathOf('vocals');
    if (r.code !== 0 || !vocals) throw new Error(`demucs failed (code ${r.code}): ${r.stderr.slice(-300)}`);
    log(`🎚 Stems ready in ${((Date.now() - t0) / 1000).toFixed(0)}s (vocals${pathOf('drums') ? ', drums' : ''}${pathOf('other') ? ', other' : ''}).`);
    return { vocals, drums: pathOf('drums'), other: pathOf('other'), bass: pathOf('bass') };
  } catch (e: any) {
    demucsReady = false;
    log(`⚠️ Stem separation unavailable (${String(e?.message || e).slice(0, 300)}); using the full mix.`);
    return null;
  }
}
/** Mono 8 kHz samples of an audio file (for level / pitch analysis). */
async function pcmOf(file: string, rate = 8000): Promise<Float32Array> {
  const r = await run('ffmpeg', ['-v', 'error', '-i', file, '-ac', '1', '-ar', String(rate), '-f', 's16le', '-'], { timeoutMs: 120000 });
  const b = r.stdout, n = Math.floor(b.length / 2), a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = b.readInt16LE(i * 2) / 32768;
  return a;
}
const rmsFrames = (a: Float32Array, rate: number, hop = 0.02) => {
  const step = Math.round(rate * hop), out: number[] = [];
  for (let i = 0; i + step <= a.length; i += step) { let s = 0; for (let j = i; j < i + step; j++) s += a[j] * a[j]; out.push(Math.sqrt(s / step)); }
  return out;
};
/** When is the singer actually singing? Spans of vocal activity from the isolated vocal track. */
async function vocalSpansOf(vocals: string): Promise<{ start: number; end: number }[]> {
  const a = await pcmOf(vocals), hop = 0.02, fr = rmsFrames(a, 8000, hop);
  if (!fr.length) return [];
  const sorted = [...fr].sort((x, y) => x - y), p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
  if (p95 < 0.004) return [];
  const on = Math.max(0.006, p95 * 0.16), off = on * 0.6;
  const spans: { start: number; end: number }[] = [];
  let cur: { start: number; end: number } | null = null;
  fr.forEach((v, i) => {
    const t = i * hop;
    if (!cur && v > on) cur = { start: t, end: t + hop };
    else if (cur && v > off) cur.end = t + hop;
    else if (cur && t - cur.end > 0.16) { spans.push(cur); cur = null; }
  });
  if (cur) spans.push(cur);
  // Merge tiny breaths, drop clicks.
  const merged: { start: number; end: number }[] = [];
  for (const sp of spans) { const last = merged[merged.length - 1]; if (last && sp.start - last.end < 0.22) last.end = sp.end; else merged.push({ ...sp }); }
  return merged.filter((sp) => sp.end - sp.start >= 0.14).map((sp) => ({ start: +sp.start.toFixed(2), end: +sp.end.toFixed(2) }));
}
/** Level of one stem relative to the whole song (0..1+). */
async function stemLevel(stem: string, mix: string): Promise<number> {
  const rm = (a: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); };
  const [x, y] = await Promise.all([pcmOf(stem), pcmOf(mix)]);
  return rm(x) / Math.max(1e-6, rm(y));
}
/** Median sung pitch (Hz) of the isolated vocals — YIN on the loud, voiced frames. */
async function medianPitch(vocals: string, spans: { start: number; end: number }[]): Promise<number> {
  const rate = 16000, a = await pcmOf(vocals, rate), N = 1024, hop = 640, minLag = Math.floor(rate / 900), maxLag = Math.floor(rate / 70);
  const inSpan = (t: number) => spans.some((sp) => t >= sp.start && t <= sp.end);
  const fr = rmsFrames(a, rate, hop / rate), sorted = [...fr].sort((p, q) => p - q), loud = sorted[Math.floor(sorted.length * 0.6)] || 0;
  const f0s: number[] = [];
  const d = new Float32Array(maxLag + 1);
  for (let i = 0, k = 0; i + N + maxLag < a.length; i += hop, k++) {
    if (!inSpan(i / rate) || (fr[k] || 0) < loud) continue;
    for (let tau = 1; tau <= maxLag; tau++) { let sum = 0; for (let j = 0; j < N; j++) { const v = a[i + j] - a[i + j + tau]; sum += v * v; } d[tau] = sum; }
    let run = 0, best = -1;
    for (let tau = 1; tau <= maxLag; tau++) {
      run += d[tau];
      const cm = run > 0 ? (d[tau] * tau) / run : 1;
      if (tau >= minLag && cm < 0.15) { while (tau + 1 <= maxLag && d[tau + 1] < d[tau]) tau++; best = tau; break; }
    }
    if (best > 0) f0s.push(rate / best);
  }
  if (f0s.length < 20) return 0;
  f0s.sort((p, q) => p - q);
  return f0s[Math.floor(f0s.length / 2)];
}
/** Does the voice clearly contradict the chosen singer? (conservative: only obvious mismatches) */
function voiceMismatch(f0: number): boolean {
  if (!f0) return false;
  return CFG.gender === 'female' ? f0 < 175 : f0 > 360;
}

async function transcribeSong(file: string): Promise<Word[]> {
  let words: Word[] = [];
  try { words = await transcribeLocal(file); } catch (e: any) { log(`⚠️ Lyric transcription error (${e?.message || e}).`); }
  if (words.length > 5) { log(`Lyric timing from Whisper on this runner (${words.length} words).`); return words; }
  log('⚠️ No transcription available; using estimated lyric timing.');
  return [];
}
/** Lay the lyric words over the moments she actually sings (vocal track), longer words taking longer. */
function wordsOverSpans(text: string, spans: { start: number; end: number }[]): Word[] {
  const toks = text.split(/\s+/).filter(Boolean);
  const total = spans.reduce((n, sp) => n + (sp.end - sp.start), 0);
  if (!toks.length || total <= 0) return [];
  const weight = (w: string) => 0.6 + Math.min(12, w.replace(/[^\p{L}\p{N}]/gu, '').length) * 0.18;
  const sumW = toks.reduce((n, w) => n + weight(w), 0);
  const out: Word[] = [];
  let si = 0, used = 0;
  for (const w of toks) {
    let need = (weight(w) / sumW) * total;
    while (si < spans.length && spans[si].end - spans[si].start - used < 0.04) { si++; used = 0; }
    if (si >= spans.length) break;
    const sp = spans[si], start = sp.start + used;
    const dur = Math.min(need, sp.end - start);
    out.push({ text: w, start: +start.toFixed(3), end: +(start + Math.max(0.08, dur)).toFixed(3), token: out.length });
    used += dur;
  }
  return out;
}
/** Languages written without spaces: split into real words (Intl.Segmenter) so lyrics can be timed and captioned. */
const NO_SPACE_LANGS = new Set(['ja', 'zh', 'th', 'lo', 'km', 'my']);
function lyricTokens(line: string, lang: string): string[] {
  if (NO_SPACE_LANGS.has(lang) && typeof (Intl as any).Segmenter === 'function') {
    try {
      const seg = new (Intl as any).Segmenter(lang, { granularity: 'word' });
      const out: string[] = [];
      for (const s of seg.segment(line)) { const w = String(s.segment).trim(); if (!w) continue; if (!s.isWordLike && out.length) out[out.length - 1] += w; else out.push(w); }
      return out;
    } catch {}
  }
  return line.split(/\s+/).filter(Boolean);
}
/** The sung lyric lines (in order, the repeated chorus included) and each token's line. */
function lyricLayout(song: MusicalSong): { lines: string[]; script: string; tokenLine: number[] } {
  const lang = musicLanguage().code;
  const lines = cleanMusicLyrics(song.lyrics).split('\n').map((x) => x.trim()).filter(Boolean);
  const toks: string[] = [], tokenLine: number[] = [];
  lines.forEach((l, i) => { for (const t of lyricTokens(l, lang)) { toks.push(t); tokenLine.push(i); } });
  return { lines, script: toks.join(' '), tokenLine };
}
/** English translation of each lyric line (one per line, same order) — for the second subtitle row. */
async function translateLyricLines(lines: string[]): Promise<string[]> {
  const lang = musicLanguage();
  if (!lines.length || lang.name.toLowerCase() === 'english') return [];
  const user = `Translate these ${lines.length} song lyric lines from ${lang.name} into natural, singable-sounding English (keep the meaning and feeling, not word-for-word). Return JSON only: {"lines": [ ... exactly ${lines.length} strings, one per input line, same order ... ]}.\n\n${lines.map((l, i) => `${i + 1}. ${l}`).join('\n')}`;
  try {
    for await (const a of LLM.attempts({ system: 'You are a professional song translator for subtitles.', user, saferUser: user, temperature: 0.2, maxTokens: 2500, json: true, task: 'lyric_translation' })) {
      try {
        const j = extractJsonObject(a.text) as any;
        const out = Array.isArray(j?.lines) ? j.lines.map((x: any) => String(x || '').trim()) : [];
        if (out.length === lines.length && out.every(Boolean)) return out;
      } catch {}
    }
  } catch {}
  log('⚠️ Lyric translation unavailable — showing the original-language captions only.');
  return [];
}
function timingForMusic(song: MusicalSong, transcript: Word[], duration: number, vocalSpans: { start: number; end: number }[] = []): { words: Word[]; sections: { start: number; end: number; tag: string; emotion: string }[] } {
  const lay = lyricLayout(song);
  const base = transcript.length ? alignWords(transcript, lay.script) : [];
  // No usable transcript: the lyrics go where she is really singing (never spread over the instrumental parts).
  const words0 = base.length > 5 ? base : vocalSpans.length ? wordsOverSpans(lay.script, vocalSpans) : estimateWordTimes(lay.script, duration);
  // Every word knows its lyric line (captions show whole lines, with the English translation under them).
  let lastLine = 0;
  const words = words0.map((w) => { const ln = typeof w.token === 'number' && w.token >= 0 ? lay.tokenLine[w.token] ?? lastLine : lastLine; lastLine = ln; return { ...w, line: ln } as Word; });
  const sections: { start: number; end: number; tag: string; emotion: string }[] = [];
  const totalWords = Math.max(1, song.lyrics.split(/\s+/).filter(Boolean).length);
  let cursor = 0;

  for (const sec of song.sections) {
    const count = Math.max(1, sec.lyrics.split(/\s+/).filter(Boolean).length);
    const ratio = count / totalWords;
    const roughEnd = Math.min(duration, cursor + Math.max(1.8, duration * ratio));
    const firstLyricWord = sec.lyrics.split(/\s+/)[0]?.replace(/[^a-z0-9']/gi, '').toLowerCase() || '';
    const first = words.find((w) => {
      const wt = w.text.replace(/[^a-z0-9']/gi, '').toLowerCase();
      return w.start >= cursor - 0.05 && w.start <= roughEnd + 0.05 && wt === firstLyricWord;
    });
    const start = clampNum(Math.max(cursor, first ? first.start : cursor), 0, duration);
    const minEnd = Math.min(duration, start + 0.8);
    const end = clampNum(Math.max(minEnd, roughEnd), minEnd, duration);
    if (end <= start) break;
    sections.push({ start, end, tag: sec.tag, emotion: sec.emotion || musicalProfile().emotion });
    // Cursor is monotonic: a matched transcript word can only advance section timing.
    cursor = Math.max(cursor, end);
    if (cursor >= duration) break;
  }

  if (sections.length) sections[sections.length - 1].end = duration;
  return { words, sections };
}
async function produceMusical(t0: number, pastTitles: string[]): Promise<number> {
  // Start installing / loading ACE-Step on this runner right away, while the lyrics are written.
  prepareLocalAce().catch(() => {});
  await reportStatus('running', '1/6 Writing the original song', 10, `Writing a ${CFG.subGenre || 'musical'} song for a ${CFG.gender} lead singer…`);
  const song = await writeMusicalSong(pastTitles);
  let generated = await generateMusicTrack(song);
  const engineName = generated.engine === 'lyria' ? 'google-lyria' : 'ace-step-1.5';
  let stems = await separateStems(generated.file, generated.duration);
  let vocalSpans = stems ? await vocalSpansOf(stems.vocals).catch(() => []) : [];
  // Voice check: the song must be sung by the singer the user chose (female / male).
  if (stems && vocalSpans.length) {
    const f0 = await medianPitch(stems.vocals, vocalSpans).catch(() => 0);
    log(`🎤 Lead voice pitch ≈ ${f0 ? f0.toFixed(0) + ' Hz' : 'unknown'} (wanted ${CFG.gender}).`);
    if (voiceMismatch(f0) && generated.engine === 'acestep') {
      log(`⚠️ The song came out in a ${CFG.gender === 'female' ? 'male' : 'female'}-sounding voice — regenerating once with a stronger ${CFG.gender} voice prompt.`);
      await reportStatus('running', '2/6 Regenerating (wrong singer voice)', 30, `The first take did not sound ${CFG.gender}; making a new take with a ${CFG.gender} singer…`);
      aceGenderBoost = true;
      try {
        const again = await generateMusicTrack(song);
        const st2 = await separateStems(again.file, again.duration);
        const sp2 = st2 ? await vocalSpansOf(st2.vocals).catch(() => []) : [];
        const f2 = st2 && sp2.length ? await medianPitch(st2.vocals, sp2).catch(() => 0) : 0;
        log(`🎤 Second take pitch ≈ ${f2 ? f2.toFixed(0) + ' Hz' : 'unknown'}.`);
        if (!voiceMismatch(f2) || Math.abs((f2 || 0) - (CFG.gender === 'female' ? 300 : 150)) < Math.abs(f0 - (CFG.gender === 'female' ? 300 : 150))) { generated = again; stems = st2; vocalSpans = sp2; }
      } catch (e: any) { log(`⚠️ Second take failed (${e?.message || e}); keeping the first.`); }
    }
  }
  const drumLevel = stems?.drums ? await stemLevel(stems.drums, generated.file).catch(() => -1) : -1;
  const keysLevel = stems?.other ? await stemLevel(stems.other, generated.file).catch(() => -1) : -1;
  if (stems) log(`🎚 ${vocalSpans.length} sung phrases; drums level ${drumLevel.toFixed(2)}, keys/other level ${keysLevel.toFixed(2)}.`);
  // Transcribe the isolated vocals when we have them (far cleaner than the full mix).
  const transcript = await transcribeSong(stems?.vocals || generated.file);
  const timing = timingForMusic(song, transcript, generated.duration, vocalSpans);
  if (!transcript.length && vocalSpans.length) log(`Lyric timing: no transcript — ${timing.words.length} words laid over ${vocalSpans.length} sung phrases.`);
  const cues = timing.sections.flatMap((sg) => [{ t: sg.start, tag: sg.emotion }, ...(sg.tag === 'chorus' ? [{ t: sg.start + 0.15, tag: 'excited' }] : [])]).filter((x) => EMOTION_TAGS.includes(x.tag));
  const chorusIntervals = timing.sections.filter((s) => s.tag === 'chorus').map((s) => ({ start: s.start, end: s.end }));
  await reportStatus('running', '3/6 Building the music-video performance', 45, `Timed ${timing.words.length} words; ${chorusIntervals.length} chorus sections will bring in the choir.`);
  const lyricLines = lyricLayout(song).lines;
  const lyricEn = await translateLyricLines(lyricLines);
  if (lyricEn.length) log(`🌍 Bilingual captions: ${musicLanguage().name} + English (${lyricEn.length} lines).`);
  const narration: Narration = { audioPath: generated.file, duration: generated.duration, words: timing.words, wordsReliable: transcript.length > 5 || vocalSpans.length > 0, engine: engineName, neural: true };
  const title = song.title;
  const badge = (CFG.subGenre || 'MUSICAL').toUpperCase().slice(0, 24);
  const endCard = 'Follow for the next original song';
  const stage = await renderWithStage({ narration, scenes: timing.sections.map((s) => ({ narration: s.tag === 'chorus' ? 'CHORUS' : s.tag, shot: 'scene', emotion: s.emotion, imageQuery: '', imageCredit: '' })) as unknown as Scene[], times: timing.sections.map((s) => ({ start: s.start, end: s.end })), cues, images: timing.sections.map(() => null), title, badge, endCard, music: null, duration: generated.duration, musicalStage: true, chorusIntervals, lyrics: { lines: lyricLines.map((text, i) => ({ text, en: lyricEn[i] || '' })), noSpaces: NO_SPACE_LANGS.has(musicLanguage().code) }, musical: { stageId: CFG.stageId || 0, stageAutoSeed: `${CFG.campaignId || CFG.campaignName || 'local'}:${CFG.partNumber}`, style: CFG.subGenre || '', sections: timing.sections, vocalSpans, drumLevel, keysLevel }, stems: stems || undefined });
  if (!stage.ok) throw new PipelineError('musical_render_failed', `Musical stage render failed: ${stage.reason || 'unknown error'}`);
  const outDur = await probeDuration(OUTPUT_VIDEO);
  if (outDur < 5) throw new PipelineError('musical_render_failed', `Rendered musical video is only ${outDur.toFixed(1)}s long.`);
  const tags = Array.from(new Set([...(song.tags || []), ...(song.hashtags || []), engineName, 'music video']));
  const hashtagLine = youtubeHashtagLine([...(song.hashtags || musicHashtags(CFG.subGenre)), ...(IS_SHORTS ? ['shorts'] : [])]);
  const description = sanitizeYouTubeDescription([song.description, `🎤 Lead singer: ${CFG.gender}.`, `🎵 Style: ${CFG.subGenre || 'Musical'}.`, '🎬 Original cinematic music video with live-style stage performance, microphone, choir chorus, lip-sync and timed captions.', hashtagLine].join('\n\n'));
  fs.writeFileSync(OUTPUT_META, JSON.stringify({ campaignId: CFG.campaignId, partNumber: CFG.partNumber, format: CFG.format, aspect: CFG.aspect, title, description, hashtags: song.hashtags, tags, model: engineName, lyrics: song.lyrics, sections: timing.sections, cues, voice: engineName, character: stage.character, durationSec: outDur, createdAt: new Date().toISOString() }, null, 2));
  const libId = await saveToLibrary({ title, topic: CFG.topic || CFG.subGenre, kind: 'musical', durationSec: Math.round(outDur), width: W, height: H });
  const picked = await pickBestFrame(OUTPUT_VIDEO, outDur, chorusIntervals.map((c) => ({ t: c.start + 1, w: 0.9 })));
  await saveLibraryThumbnail(libId, picked?.file || null);
  let published: { videoId: string; url: string; privacy: string } | null = null;
  const socialPublished: Record<string, string> = {}, publishErrors: Record<string, string> = {};
  if (PUBLISH && !CFG.dryRun) {
    let done = 0; const totalTargets = Number(WANT.youtube) + Number(WANT.facebook) + Number(WANT.instagram) + Number(WANT.threads);
    const attemptPublish = async (platform: string, fn: () => Promise<string>, message: string) => {
      await reportStatus('running', '6/6 Publishing the music video', Math.min(98, 82 + Math.round((done / Math.max(1, totalTargets)) * 16)), message);
      for (let attempt = 1; attempt <= 3; attempt++) { try { const u = await fn(); done++; return u; } catch (e: any) { if (attempt === 3) { publishErrors[platform] = String(e?.message || e).slice(0, 500); log(`❌ ${platform} publish failed after 3 attempts: ${publishErrors[platform]}`); } else await sleep(4000 * attempt); } }
      return '';
    };
    if (WANT.youtube) {
      const u = await attemptPublish('YouTube', async () => { const p = await uploadToYouTube({ title, description, tags, synthetic: true }); published = p; await setYouTubeThumbnail(p.videoId, picked?.file || null); await linkLibraryVideo(libId, p.url); return p.url; }, `Uploading the ${IS_SHORTS ? 'Short' : 'video'} to YouTube…`); if (!u) log('YouTube did not publish; other selected platforms will still be attempted.');
    }
    if (WANT.facebook || WANT.instagram || WANT.threads) {
      const videoUrl = await publicVideoUrl(libId), meta = { title, description, videoUrl };
      if (WANT.facebook) { const u = await attemptPublish('Facebook', () => publishFacebookVideo(meta), 'Uploading the finished music video to Facebook…'); if (u) socialPublished.facebook = u; }
      if (WANT.instagram) { const u = await attemptPublish('Instagram', () => publishInstagramVideo(meta), 'Publishing the finished music video as an Instagram Reel…'); if (u) socialPublished.instagram = u; }
      if (WANT.threads) { const u = await attemptPublish('Threads', () => publishThreadsVideo(meta), 'Publishing the finished music video to Threads…'); if (u) socialPublished.threads = u; }
    }
    if (!published && !Object.keys(socialPublished).length) throw new PipelineError('publish_failed', Object.values(publishErrors)[0] || 'All selected publishing destinations rejected the music video.');
  }
  const episode = await appRequest('POST', `${campaignPath()}/episodes`, { partNumber: CFG.partNumber, title, script: song.lyrics, description, youtubeUrl: published?.url || '', videoId: published?.videoId || '', published: !!published || Object.keys(socialPublished).length > 0, socialUrls: socialPublished, socialErrors: publishErrors, privacyStatus: published?.privacy || '', format: CFG.format, aspectRatio: CFG.aspect, durationSec: Math.round(outDur), runId: CFG.runId, runUrl: CFG.runUrl, model: engineName, voice: engineName, character: stage.character });
  if (CFG.campaignId && CFG.appUrl && (!episode || episode.status >= 300)) console.warn(`⚠️ Could not record the musical episode in the app (${episode ? `HTTP ${episode.status}` : 'app unreachable'}).`);
  const secs = ((Date.now() - t0) / 1000).toFixed(0), links = [published?.url, ...Object.values(socialPublished)].filter(Boolean);
  await reportStatus('completed', links.length ? 'Published' : 'Video rendered', 100, links.length ? `✅ Musical video published in ${secs}s: ${links.join(' · ')}` : `✅ Musical video rendered in ${secs}s (not published).`, { youtubeUrl: published?.url || '', socialUrls: socialPublished, socialErrors: publishErrors });
  return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Studio → Create animation: the user's own script / voice → one finished MP4 for the app.
// ---------------------------------------------------------------------------
/** Speech → words with timings (faster-whisper on this runner, installed on first use). */
async function transcribeVoice(audio: string, hint: string): Promise<Word[] | null> {
  const venv = path.join(os.homedir(), 'asr-venv');
  let py = path.join(venv, 'bin', 'python');
  if (!fs.existsSync(py) || (await run(py, ['-c', 'import faster_whisper'], { timeoutMs: 60000 })).code !== 0) {
    log('Setting up speech recognition (first time only)…');
    if ((await run('python3', ['-m', 'venv', venv], { timeoutMs: 180000 })).code !== 0) py = 'python3';
    let r = await run(py, ['-m', 'pip', 'install', '-q', '--disable-pip-version-check', 'faster-whisper'], { timeoutMs: 900000 });
    if (r.code !== 0) r = await run(py, ['-m', 'pip', 'install', '-q', '--disable-pip-version-check', '--break-system-packages', 'faster-whisper'], { timeoutMs: 900000 });
    if ((await run(py, ['-c', 'import faster_whisper'], { timeoutMs: 60000 })).code !== 0) { log(`Speech recognition unavailable: ${r.stderr.slice(-200)}`); return null; }
  }
  const script = path.join(WORK_DIR, 'studio_asr.py');
  fs.writeFileSync(script, `import json,sys
from faster_whisper import WhisperModel
audio,hint,out=sys.argv[1],sys.argv[2],sys.argv[3]
m=WhisperModel("small",device="cpu",compute_type="int8")
segs,info=m.transcribe(audio,word_timestamps=True,beam_size=1,condition_on_previous_text=False,initial_prompt=(hint or None))
words=[]
for s in segs:
  for w in (s.words or []):
    t=w.word.strip()
    if t: words.append({"text":t,"start":round(w.start,3),"end":round(w.end,3)})
json.dump(words,open(out,"w"))
`);
  const wav = path.join(WORK_DIR, 'studio_asr.wav');
  if ((await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', audio, '-ac', '1', '-ar', '16000', wav], { timeoutMs: 120000 })).code !== 0) return null;
  const out = path.join(WORK_DIR, 'studio_asr.json');
  const r = await run(py, [script, wav, hint.slice(0, 220), out], { timeoutMs: 30 * 60_000 });
  if (r.code !== 0 || !fs.existsSync(out)) { log(`Transcription failed: ${r.stderr.slice(-200)}`); return null; }
  const list: { text: string; start: number; end: number }[] = JSON.parse(fs.readFileSync(out, 'utf8') || '[]');
  return list.length ? list.map((w, i) => ({ text: w.text.replace(/\s+/g, ''), start: w.start, end: w.end, token: i })) : null;
}

/** The user's recording: trimmed to the plan's length, levelled, and turned into timed words. */
async function studioVoice(S: StudioJob): Promise<{ narration: Narration; text: string }> {
  const src = path.join(WORK_DIR, 'studio_voice_src');
  const url = /^https?:/.test(S.voiceUrl) ? S.voiceUrl : `${CFG.appUrl}${S.voiceUrl}`;
  const res = await fetch(url, { headers: { 'X-Animato-Runner-Key': CFG.runnerKey || '' }, signal: AbortSignal.timeout(180000) }).catch(() => null);
  if (!res?.ok) throw new PipelineError('voice_missing', 'Your voice recording could not be loaded. Please upload it again.');
  fs.writeFileSync(src, Buffer.from(await res.arrayBuffer()));
  const mp3 = path.join(WORK_DIR, 'studio_voice.mp3');
  const ff = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', src, '-t', String(S.maxSeconds), '-vn', '-ac', '1', '-ar', '44100', '-af', 'highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11', '-c:a', 'libmp3lame', '-b:a', '160k', mp3], { timeoutMs: 300000 });
  const duration = fs.existsSync(mp3) ? await probeDuration(mp3) : 0;
  if (ff.code !== 0 || duration < 1) throw new PipelineError('voice_unreadable', 'That audio file could not be read. Try an MP3, M4A or WAV recording.');
  await reportStatus('running', 'Listening to your voice', 20, `Voice recording: ${duration.toFixed(0)}s.`);
  const heard = await transcribeVoice(mp3, S.script);
  if (heard?.length) {
    const text = heard.map((w) => w.text).join(' ');
    return { narration: { audioPath: mp3, duration, words: heard, wordsReliable: true, engine: 'your voice', neural: true }, text };
  }
  if (!S.script.trim()) throw new PipelineError('voice_unclear', 'We could not make out the words in your recording. Add the words as text too, or record again somewhere quieter.');
  const text = cleanForSpeech(S.script);
  return { narration: { audioPath: mp3, duration, words: estimateWordTimes(text, duration), wordsReliable: false, engine: 'your voice', neural: true }, text };
}

/** Split the user's words into scenes (keeping every word exactly) and pick a picture for each. */
async function studioScenes(S: StudioJob, text: string): Promise<Script> {
  const words = text.split(/\s+/).filter(Boolean);
  // Sentences, grouped into scenes of roughly 18–32 words.
  const sentences = text.match(/[^.!?…]+[.!?…]+["')\]]*|[^.!?…]+$/g)?.map((x) => x.trim()).filter(Boolean) || [text];
  const groups: string[] = [];
  let cur = '';
  for (const se of sentences) { const n = (cur + ' ' + se).trim().split(/\s+/).length; if (cur && n > 32) { groups.push(cur); cur = se; } else cur = (cur + ' ' + se).trim(); }
  if (cur) groups.push(cur);
  // Never split words differently from the voice: rebuild each group from the word list.
  const parts: string[] = [];
  let k = 0;
  for (const g of groups) { const n = g.split(/\s+/).filter(Boolean).length; parts.push(words.slice(k, k + n).join(' ')); k += n; }
  if (k < words.length) parts[parts.length - 1] = `${parts[parts.length - 1]} ${words.slice(k).join(' ')}`.trim();
  const style = S.kind === 'story' ? 'storybook illustration, 3D animated film still, warm cinematic light, rich colours' : S.kind === 'kids' ? 'bright cheerful cartoon illustration for children, soft shapes, pastel colours' : 'clean modern illustration, soft light, vibrant colours';
  let prompts: { imagePrompt: string; searchQuery: string; emotion: string; title?: string }[] = parts.map((p) => ({ imagePrompt: `${style}, ${p.slice(0, 220)}`, searchQuery: p.split(/\s+/).filter((w) => w.length > 4).slice(0, 4).join(' '), emotion: 'neutral' }));
  let title = S.title;
  if (!CFG.offline) {
    const user = `Here is a narration split into ${parts.length} scenes. For EACH scene write a vivid image prompt (what the picture shows; ${style}), a 2-4 word stock-photo search query, and the speaker's emotion (one of: ${EMOTION_TAGS.join(', ')}). Also give the video a catchy title (max 60 chars).
SCENES:
${parts.map((p, i) => `${i + 1}. ${p}`).join('\n')}
Return ONLY JSON: {"title": "", "scenes": [{"imagePrompt": "", "searchQuery": "", "emotion": "neutral"}]}`;
    for await (const a of LLM.attempts({ system: 'You are a storyboard artist for animated videos. You answer with one JSON object.', user, json: true, temperature: 0.7, maxTokens: 3000, timeoutMs: 90000, task: 'studio_storyboard' })) {
      try {
        const j = extractJsonObject(a.text);
        if (!Array.isArray(j.scenes) || j.scenes.length < parts.length) throw new Error('scene count');
        prompts = parts.map((_, i) => ({
          imagePrompt: `${style}, ${String(j.scenes[i].imagePrompt || prompts[i].imagePrompt).slice(0, 400)}`,
          searchQuery: String(j.scenes[i].searchQuery || prompts[i].searchQuery).slice(0, 60),
          emotion: EMOTION_TAGS.includes(String(j.scenes[i].emotion)) ? String(j.scenes[i].emotion) : 'neutral',
        }));
        if (!title && j.title) title = String(j.title).slice(0, 80);
        break;
      } catch (e: any) { log(`Storyboard from ${a.provider}/${a.model} unusable (${e?.message}).`); }
    }
  }
  return {
    title: title || parts[0].split(/\s+/).slice(0, 7).join(' ').replace(/[.,!?]+$/, ''),
    description: '', hashtags: [], tags: [], visualStyle: style, characters: '', usedFallbackTemplate: false,
    scenes: parts.map((p, i) => ({ narration: p, cues: [{ index: 0, tag: prompts[i].emotion }], shot: i === 0 ? 'scene' : (i % 3 === 2 ? 'panel' : 'scene'), emotion: prompts[i].emotion, imagePrompt: prompts[i].imagePrompt, searchQuery: prompts[i].searchQuery })),
  } as Script;
}

/** Hand the finished video to the app in pieces (the app keeps it for 24 hours; the device keeps a copy). */
async function studioDeliver(S: StudioJob, meta: { title: string; durationSec: number }) {
  const size = fs.statSync(OUTPUT_VIDEO).size;
  const start = await appRequest('POST', `/api/animate/runner/${S.id}/upload`, { size }, 30000);
  if (!start || start.status >= 300 || !start.data?.upload) throw new PipelineError('upload_failed', `The finished video could not be handed over (${start ? `HTTP ${start.status} ${start.data?.error || ''}` : 'app unreachable'}).`);
  const up = start.data.upload, chunk = Number(up.chunkSize) || 6 * 1024 * 1024;
  const n = Math.max(1, Math.ceil(size / chunk));
  const etags: string[] = new Array(n).fill('');
  const fd = fs.openSync(OUTPUT_VIDEO, 'r');
  try {
    for (let i = 0; i < n; i++) {
      const len = Math.min(chunk, size - i * chunk);
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, i * chunk);
      for (let a = 0; a < 5 && !etags[i]; a++) {
        try {
          const res = await fetch(`${CFG.appUrl}${up.base}/${i}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream', 'X-Animato-Runner-Key': CFG.runnerKey || '' }, body: buf, signal: AbortSignal.timeout(120000) });
          const j: any = await res.json().catch(() => ({}));
          if (res.ok && j.etag) etags[i] = String(j.etag); else await sleep(2000 * (a + 1));
        } catch { await sleep(2000 * (a + 1)); }
      }
      if (!etags[i]) throw new PipelineError('upload_failed', `Piece ${i + 1} of the video could not be sent.`);
      await reportStatus('running', 'Delivering your animation', Math.min(99, 92 + Math.round((7 * (i + 1)) / n)), `Sent ${i + 1}/${n}.`);
    }
  } finally { fs.closeSync(fd); }
  const done = await appRequest('POST', `/api/animate/runner/${S.id}/complete`, { parts: etags, size, width: W, height: H, ...meta }, 60000);
  if (!done || done.status >= 300) throw new PipelineError('upload_failed', `The video could not be saved (${done ? `HTTP ${done.status} ${done.data?.error || ''}` : 'app unreachable'}).`);
}

async function produceStudio(t0: number): Promise<number> {
  const S = STUDIO!;
  log(`Studio animation ${S.id}: ${S.kind}, ${S.voiceUrl ? 'your voice' : 'AI voice'}, ${S.cast.length} character(s), up to ${S.maxSeconds}s.`);
  await reportStatus('running', 'Preparing your animation', 6, 'Started.');
  if (S.cast[0]?.spec) CFG.characterSpec = S.cast[0].spec;
  CFG.gender = S.cast[0]?.gender || S.voiceGender;

  // ---- Dialogue films: several characters talking in a place (the 2D film engine) ----
  if (S.kind === 'film') {
    const ids = new Map<string, string>();
    const characters = S.cast.map((c, i) => { ids.set(c.name.toLowerCase(), `c${i + 1}`); return { id: `c${i + 1}`, name: c.name, gender: c.gender, look: '', spec: c.spec ? { ...c.spec, name: c.name } : null }; });
    const scenes = (S.scenes.length ? S.scenes : [{ lines: [] as any[] }]).map((sc) => ({
      location: sc.location || S.location, time: sc.time === 'night' ? 'night' : S.time, card: '',
      cast: characters.map((c) => c.id),
      lines: (sc.lines || []).map((l) => ({ speaker: ids.get(String(l.speaker || '').toLowerCase()) || 'c1', text: String(l.text || '').slice(0, 400) })).filter((l) => l.text),
    })).filter((sc) => sc.lines.length);
    if (!scenes.length) throw new PipelineError('script_empty', 'Write at least one line of dialogue.');
    (CFG as any).studioFilm = { title: S.title || 'My animation', logline: '', description: '', hashtags: [], mood: 'story', characters, scenes };
    CFG.category = 'animation';
    (CFG as any).animStyle = 'film';
    await reportStatus('running', 'Recording the voices', 25, `${characters.length} character(s), ${scenes.reduce((n, s) => n + s.lines.length, 0)} lines.`);
    const kit = animKit();
    const out = await runAnimated(kit as any);
    const outDur = await probeDuration(OUTPUT_VIDEO);
    if (outDur < 2) throw new PipelineError('render_failed', `Rendered video is only ${outDur.toFixed(1)}s long.`);
    if (S.watermark) await studioWatermark();
    await reportStatus('running', 'Delivering your animation', 92, `Rendered ${outDur.toFixed(0)}s.`);
    await studioDeliver(S, { title: out.title || S.title || 'My animation', durationSec: Math.round(outDur) });
    await reportStatus('completed', 'Your animation is ready', 100, `Done in ${Math.round((Date.now() - t0) / 1000)}s.`);
    return 0;
  }

  // ---- A talking character (with pictures behind): talk, story, kids, news, cooking, tech, ads ----
  CFG.category = ({ story: 'stories', kids: 'stories', talk: 'tech', motivation: 'tech', news: 'news', cooking: 'cooking', tech: 'tech', ads: 'ads' } as Record<string, string>)[S.kind] || 'tech';
  if (CFG.category === 'stories' && !CFG.offline) void startImageGenerator();
  let narration: Narration, text: string;
  if (S.voiceUrl) ({ narration, text } = await studioVoice(S));
  else {
    text = cleanForSpeech(S.script).split(/\s+/).slice(0, Math.round(S.maxSeconds * 2.6)).join(' ');
    if (text.split(/\s+/).length < 3) throw new PipelineError('script_empty', 'Write a few words for your character to say.');
    await reportStatus('running', 'Recording the voice', 18, `${text.split(/\s+/).length} words.`);
    CFG.gender = S.voiceGender;
    narration = await synthesizeNarration(text);
    text = cleanForSpeech(text);
  }
  await reportStatus('running', 'Planning the scenes', 32, `Voice ready (${narration.duration.toFixed(0)}s).`);
  const script = await studioScenes(S, text);
  const duration = +(narration.duration + 1.0).toFixed(3);
  const times = timeScenes(script.scenes, narration.words, duration);
  const cues = timeCues(script.scenes, narration.words, times);
  await reportStatus('running', 'Drawing the scenes', 45, `${script.scenes.length} scenes.`);
  const [{ files: images, credits }, music] = await Promise.all([gatherImages(script), findMusic(duration, narration)]);
  await reportStatus('running', 'Animating your character', 62, 'Animating.');
  const badge = S.kind === 'kids' ? 'KIDS' : S.kind === 'story' ? 'STORY' : S.kind === 'talk' || S.kind === 'motivation' ? '' : S.kind.toUpperCase();
  const stage = await renderWithStage({ narration, scenes: script.scenes, times, cues, images, title: script.title, badge, endCard: S.watermark ? 'Made with Animato Studio Pro' : '', music, duration, credits });
  if (!stage.ok) {
    log(`⚠️ Character renderer unavailable (${stage.reason}).`);
    await renderFallback({ narration, times, images: images as string[], title: script.title, badge, music, duration });
  }
  const outDur = await probeDuration(OUTPUT_VIDEO);
  if (outDur < 2) throw new PipelineError('render_failed', `Rendered video is only ${outDur.toFixed(1)}s long.`);
  if (S.watermark) await studioWatermark();
  await reportStatus('running', 'Delivering your animation', 92, `Rendered ${outDur.toFixed(0)}s.`);
  await studioDeliver(S, { title: script.title, durationSec: Math.round(outDur) });
  await reportStatus('completed', 'Your animation is ready', 100, `Done in ${Math.round((Date.now() - t0) / 1000)}s.`);
  return 0;
}

/** Free plan: a small "Animato Studio Pro" mark in the corner (Premium has none). */
async function studioWatermark() {
  const tmp = path.join(WORK_DIR, 'wm.mp4');
  const font = [path.join(HERE, 'assets', 'Poppins-Bold.ttf'), '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'].find((f) => fs.existsSync(f));
  const fontArg = font ? `fontfile='${font.replace(/'/g, "\\'")}':` : '';
  const size = Math.round(Math.min(W, H) * 0.028);
  const r = await run('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', OUTPUT_VIDEO, '-vf', `drawtext=${fontArg}text='Animato Studio Pro':fontcolor=white@0.78:fontsize=${size}:borderw=2:bordercolor=black@0.35:x=w-tw-${Math.round(size * 0.9)}:y=${Math.round(size * 1.1)}`, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', tmp], { timeoutMs: 20 * 60_000 });
  if (r.code === 0 && fs.existsSync(tmp) && fs.statSync(tmp).size > 10000) fs.renameSync(tmp, OUTPUT_VIDEO);
  else log(`Watermark skipped: ${r.stderr.slice(-200)}`);
}

/** The toolkit the 2D animation engine needs (shared by automations and Studio animations). */
function animKit() {
  const landscapePodcast = CFG.category === 'podcast' && H >= W;
  const RW = landscapePodcast ? 1920 : W, RH = landscapePodcast ? 1080 : H;
  return {
    hostVoice: '',
    CFG, W: RW, H: RH, FPS, IS_SHORTS, WORK_DIR, HERE, LLM, extractJsonObject, log, reportStatus,
    run: (cmd: string, args: string[], o: { timeoutMs?: number } = {}) => run(cmd, args, o),
    probeDuration, pastTitles: [] as string[], PipelineError,
    composeMusic: (mood: string, seconds: number, seed: string) => composeBuffers(mood as any, Math.max(10, seconds), seed),
    automateLevel, eqForVoice,
    runStage: (o: { stageFile: string; job: any; audioFinal: string; files: Record<string, string>; duration: number }) => runStage({
      stageFile: o.stageFile, job: { ...o.job, wardrobeSeed: `${STUDIO?.id || CFG.campaignId || 'local'}:${CFG.partNumber}` }, duration: o.duration, files: o.files, size: { w: RW, h: RH },
      audio: { inputs: ['-i', o.audioFinal], filter: '[1:a]aresample=48000,apad[aout]' }
    })
  };
}


async function main() {
  const t0 = Date.now();
  console.log('='.repeat(64));
  console.log('ANIMATO CLOUD RENDERER');
  console.log(`campaign=${CFG.campaignId || '(none)'} part=${CFG.partNumber}/${CFG.arcParts} category=${CFG.category} format=${CFG.format} ${W}x${H} gender=${CFG.gender} autoPost=${CFG.autoPost}`);
  console.log('='.repeat(64));

  // Stories are drawn by the local image generator: start loading its model now, while the script is written.
  if (CFG.category === 'stories' && !CFG.offline) void startImageGenerator();

  let pastStory = CFG.previousScript ? `PART ${CFG.partNumber - 1}:\n${CFG.previousScript}` : '';
  let pastTitles: string[] = [];
  let pastSources: string[] = [];
  if (CFG.campaignId && CFG.appUrl) {
    const camp = await appRequest('GET', campaignPath());
    if (camp?.status === 404) { log('This automation was deleted in the app — nothing to do.'); return 0; }
    if (camp?.status === 200 && camp.data?.campaign?.status === 'paused') {
      log('This automation is paused in the app — skipping this run.');
      await reportStatus('skipped', 'Skipped (automation paused)', 0, 'Run skipped because the automation is paused.');
      return 0;
    }
    if (!camp) log('⚠️ The app is not reachable from GitHub right now — continuing without live status updates.');
    const hist = await appRequest('GET', `${campaignPath()}/history`);
    const episodes: any[] = Array.isArray(hist?.data?.storyHistory) ? hist!.data.storyHistory : [];
    // Titles of recent videos (scripts are kept 5 days) + the last 30 video titles the app keeps for good.
    const kept: string[] = Array.isArray(hist?.data?.coveredTitles) ? hist!.data.coveredTitles.map((x: any) => String(x || '')) : [];
    pastTitles = Array.from(new Set([...kept, ...episodes.map((e) => String(e.title || ''))].filter(Boolean)));
    pastSources = [
      ...episodes.flatMap((e) => (Array.isArray(e.sources) ? e.sources : [])),
      ...(Array.isArray(hist?.data?.usedHeadlines) ? hist!.data.usedHeadlines : [])
    ].map((x: any) => String(x || '')).filter(Boolean);
    if (CFG.category === 'stories' && episodes.length) {
      pastStory = episodes.filter((e) => Number(e.partNumber) < CFG.partNumber).slice(-4)
        .map((e) => `PART ${e.partNumber} — ${e.title}:\n${String(e.script || '').slice(0, 1600)}`).join('\n\n') || pastStory;
    }
  }

  // Titles made by everyone else in this category: avoided too, so no two users get the same video.
  if (CFG.globalTitles.length) pastTitles = Array.from(new Set([...CFG.globalTitles.slice(-20), ...pastTitles]));

  if (STUDIO) return await produceStudio(t0);

  await reportStatus('running', CFG.category === 'musical' ? '1/6 Starting the musical video' : '1/5 Writing the script', 8, `GitHub runner started Part ${CFG.partNumber} (${CFG.format === 'shorts' ? 'YouTube Short' : 'YouTube video'}, ${CFG.aspect}).`);

  if (!CFG.dryRun) {
    await loadSocialCredentials();
    if (WANT.youtube) {
      await youtubeAccessToken();
      log('YouTube connection verified.');
    }
    if (WANT.facebook) log('Facebook publishing connection verified.');
    if (WANT.instagram) log('Instagram publishing connection verified.');
    if (WANT.threads) log('Threads publishing connection verified.');
  }

  // 2D animation (stickman, short films) and podcasts have their own pipeline.
  if (ANIM_CATEGORIES.has(CFG.category)) return await produceAnimated(t0, pastTitles);

  if (CFG.category === 'musical') return await produceMusical(t0, pastTitles);

  // 1. Script
  const script = await generateScript(pastStory, pastTitles, pastSources);
  polishMetadata(script);
  log(`Metadata: ${script.description.split(/\s+/).length}-word description, hashtags: ${script.hashtags.map((h) => `#${h}`).join(' ')}`);
  if (script.usedFallbackTemplate && !CFG.allowFallbackPublish) {
    // Keys work but every free model was busy / rate-limited / filtered: retry later (no pause).
    throw new PipelineError('script_retry', `The AI model on the runner could not write a usable script this time (${script.aiError || 'unknown error'}). Nothing was posted; the next attempt runs automatically.`);
  }
  const fullText = script.scenes.map((s) => s.narration).join(' ');
  await reportStatus('running', '2/5 Recording the voice-over', 22, `Script ready: "${script.title}" (${script.scenes.length} scenes${script.model ? `, ${script.model}` : ''}).`);

  // 2. Voice (+ start fetching images in parallel)
  const imagesPromise = gatherImages(script);
  const narration = await synthesizeNarration(fullText);
  if (!narration.neural && PUBLISH && !CFG.allowFallbackPublish) {
    throw new PipelineError('tts_failed', `The neural voice (Microsoft Edge TTS) failed on every voice and retry, so only a robotic fallback voice was available. Nothing was posted; the next run will retry automatically. Last error: ${LAST_TTS_ERROR.slice(0, 300) || 'unknown'}`);
  }
  const duration = +(narration.duration + (CFG.category === 'stories' ? 1.6 : 1.2)).toFixed(3);
  const times = timeScenes(script.scenes, narration.words, duration);
  const cues = timeCues(script.scenes, narration.words, times);
  log(`Performance: ${cues.length} cues (${cues.filter((c) => EMOTION_TAGS.includes(c.tag)).length} expression changes, ${cues.filter((c) => c.tag.startsWith('look')).length} looks) pinned to word timings.`);
  await reportStatus('running', '3/5 Finding an image for every scene', 38, `Voice-over recorded (${narration.duration.toFixed(0)}s, ${narration.engine}).`);

  // 3. Images, character rig, music
  const [{ files: images, aiCount, credits }, music] = await Promise.all([imagesPromise, findMusic(duration, narration)]);
  const presenter = CFG.characterSpec ? `${CFG.characterSpec.name || 'custom'} (designed in the app)` : `default ${CFG.gender} presenter`;
  await reportStatus('running', '4/5 Rendering the video', 58, `${images.filter(Boolean).length} scene images ready; character: ${presenter}.`);

  // 4. Render
  const badge = CFG.category === 'ads' ? (CFG.adProfile.company ? CFG.adProfile.company.toUpperCase().slice(0, 28) : 'SPONSORED') : CFG.category === 'stories' ? `${CFG.partNumber >= CFG.arcParts ? 'FINALE' : `PART ${CFG.partNumber}`}${CFG.subGenre ? ` · ${CFG.subGenre.toUpperCase()}` : ''}`
    : CFG.category === 'cooking' ? 'RECIPE' : CFG.category === 'tech' ? 'TECH' : CFG.category === 'news' ? 'NEWS' : CFG.category.toUpperCase();
  const endCard = CFG.category === 'stories'
    ? (CFG.partNumber >= CFG.arcParts ? 'New story next — follow!' : `Part ${CFG.partNumber + 1} next — follow!`)
    : CFG.category === 'ads' && CFG.adProfile.service ? `Message ${CFG.adProfile.company || 'us'} to set it up`.slice(0, 44)
    : 'Follow for more';
  const title = script.title.replace(/\s*\(part \d+\)\s*$/i, '');
  // Ads: on a Shorts turn, the ad creative itself always renders in the normal
  // YouTube video ratio (16:9) — the switch does not squeeze or crop it — and is
  // then placed in the middle of the vertical Short frame with a blurred backdrop,
  // the same way a landscape podcast is framed into a Short (see landscapeIntoFrame).
  const adLandscapeShort = CFG.category === 'ads' && IS_SHORTS;
  const renderSize = adLandscapeShort ? { w: 1920, h: 1080 } : undefined;
  const stage = await renderWithStage({ narration, scenes: script.scenes, times, cues, images, title, badge, endCard, music, duration, credits, size: renderSize });
  let characterMode = stage.character;
  if (!stage.ok) {
    log(`⚠️ Character renderer unavailable (${stage.reason}). Rendering scenes + captions without the character.`);
    await reportStatus('running', '4/5 Rendering (fallback)', 70, `⚠️ Character renderer failed: ${stage.reason}. Using the fallback renderer.`);
    await renderFallback({ narration, times, images: images as string[], title, badge, music, duration, size: renderSize });
    characterMode = 'none (fallback)';
  }
  if (adLandscapeShort) {
    await landscapeIntoFrame('');
    log(`Ad rendered at 16:9, then framed into the ${W}x${H} Short with a blurred backdrop.`);
  }
  const outDur = await probeDuration(OUTPUT_VIDEO);
  if (outDur < 3) throw new PipelineError('render_failed', `Rendered video is only ${outDur.toFixed(1)}s long.`);
  log(`Video: ${(fs.statSync(OUTPUT_VIDEO).size / 1e6).toFixed(1)} MB, ${outDur.toFixed(1)}s, character: ${characterMode}.`);

  // Keep hashtags relevant and valid; never generate ##foo or generic spam tags.
  const hashtagLine = youtubeHashtagLine([
    ...(Array.isArray(script.hashtags) ? script.hashtags : []),
    ...(IS_SHORTS ? ['shorts'] : [])
  ]);
  const series = CFG.category === 'stories'
    ? (CFG.partNumber >= CFG.arcParts
      ? (CFG.arcParts > 1 ? `📺 This is the finale of the story. A brand-new story starts on the next upload — follow so you don't miss it.` : '')
      : `📺 Part ${CFG.partNumber} of ${CFG.arcParts}. Part ${CFG.partNumber + 1} is coming next — follow so you don't miss it.`)
    : '';
  const sourcesBlock = [
    script.sources?.length ? `📰 SOURCES
${script.sources.map((x) => `• ${x}`).join('\n')}` : '',
    script.sourceLinks?.length ? script.sourceLinks.map((u) => `• ${u}`).join('\n') : '',
    script.factChecked ? (script.factPack?.singleSource ? `ℹ️ Based on reporting by ${script.factPack.singleSource}; every number was matched against its article before publishing.` : '✅ Written from the sources above; every number was matched against them before publishing.') : '',
    script.officialUrl ? `🔗 Official site: ${script.officialUrl}` : ''
  ].filter(Boolean).join('\n');
  const creditsBlock = [
    script.imageAttributions?.length
      ? `🖼️ IMAGE CREDITS (images may be cropped)
${script.imageAttributions.map((a) => `• ${a}`).join('\n')}`.slice(0, 1800)
      : '',
    script.imageCredits?.some((c) => c.startsWith('Screenshot:'))
      ? (CFG.category === 'news'
        ? `Screenshots show the source page this story was reported on (${script.imageCredits.filter((c) => c.startsWith('Screenshot:')).map((c) => c.replace('Screenshot: ', '')).join(', ')}).`
        : `Screenshots of ${script.officialUrl ? hostOf(script.officialUrl) : 'the official website'} are shown to explain how to use it.`)
      : '',
    CFG.category !== 'stories' ? '' : 'Story art is original and AI-generated for this video.',
    '🎵 Music: original, composed for this video.'
  ].filter(Boolean).join('\n');
  const P = CFG.adProfile;
  const adContact = CFG.category !== 'ads' || !(P.company || P.contact) ? ''
    : P.service ? `📩 Want this for your business? Message ${P.company || 'us'}${P.contact ? `: ${P.contact}` : ''} — we set it up for you.`
    : P.company ? `By ${P.company}.` : '';
  const description = sanitizeYouTubeDescription([
    script.description || script.title,
    adContact,
    descriptionDetails(script),
    series,
    sourcesBlock,
    creditsBlock,
    hashtagLine
  ].filter(Boolean).join('\n\n').trim());

  fs.writeFileSync(OUTPUT_META, JSON.stringify({
    campaignId: CFG.campaignId, partNumber: CFG.partNumber, format: CFG.format, aspect: CFG.aspect,
    title: script.title, description, hashtags: script.hashtags, tags: script.tags, model: script.model,
    scenes: script.scenes.map((s, i) => ({ ...s, start: times[i].start, end: times[i].end, image: images[i] ? path.basename(images[i]!) : null, imageCredit: credits[i] || null })),
    imageSource: CFG.category === 'stories' ? 'generated (animated story art)' : 'real, freely licensed images (public domain / CC0 / CC BY) + official-site screenshots', imageCredits: script.imageCredits || [], imageAttributions: script.imageAttributions || [], officialUrl: script.officialUrl || '',
    cues,
    voice: narration.engine, character: characterMode, durationSec: outDur, createdAt: new Date().toISOString()
  }, null, 2));

  // 5. Keep a copy in the app's video library, then publish on YouTube.
  const libId = await saveToLibrary({ title: script.title, topic: CFG.topic || CFG.subGenre, kind: CFG.category, durationSec: Math.round(outDur), width: W, height: H });
  // Strong moments: big expressions and each new picture on screen.
  const highlights: Highlight[] = [
    ...cues.filter((c) => ['excited', 'surprised', 'happy', 'laugh', 'shocked', 'angry', 'scared'].includes(c.tag)).map((c) => ({ t: c.t, w: 0.5 })),
    ...times.map((x, i) => ((images as (string | null)[])[i] ? { t: x.start + 0.6, w: 0.35 } : null)).filter(Boolean) as Highlight[],
  ];
  const picked = await pickBestFrame(OUTPUT_VIDEO, outDur, highlights);
  const thumb = picked?.file || null;
  await saveLibraryThumbnail(libId, thumb);
  let published: { videoId: string; url: string; privacy: string } | null = null;
  const socialPublished: Record<string, string> = {};
  const publishErrors: Record<string, string> = {};
  if (!PUBLISH) {
    log('Publishing is OFF for this automation — the video is saved as a run artifact only.');
  } else if (CFG.dryRun) {
    log('Dry run — skipping publishing.');
  } else {
    let done = 0;
    const totalTargets = Number(WANT.youtube) + Number(WANT.facebook) + Number(WANT.instagram) + Number(WANT.threads);
    const publishStep = (label: string) => Math.min(98, 82 + Math.round((done / Math.max(1, totalTargets)) * 16));
    const attemptPublish = async (platform: string, fn: () => Promise<string>, step: string, message: string) => {
      await reportStatus('running', step, publishStep(platform), message);
      let last: any = null;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const url = await fn();
          done++;
          return url;
        } catch (e: any) {
          last = e;
          if (attempt < 3) await new Promise((z) => setTimeout(z, 4000 * attempt));
        }
      }
      const msg = String(last?.message || last || `${platform} publishing failed`);
      publishErrors[platform] = msg.slice(0, 500);
      log(`❌ ${platform} publish failed after 3 attempts: ${msg}`);
      return '';
    };
    if (WANT.youtube) {
      const url = await attemptPublish('YouTube', async () => {
        const p = await uploadToYouTube({ title: script.title, description, tags: script.tags, synthetic: aiCount > 0 });
        log(`Published on YouTube: ${p.url} (privacy: ${p.privacy})`);
        await setYouTubeThumbnail(p.videoId, thumb);
        await linkLibraryVideo(libId, p.url);
        published = p;
        return p.url;
      }, '5/5 Publishing to YouTube', `Uploading the ${IS_SHORTS ? 'Short' : 'video'} to YouTube…`);
      if (!url) log('YouTube did not publish; other selected platforms will still be attempted.');
    }
    if (WANT.facebook || WANT.instagram || WANT.threads) {
      const videoUrl = await publicVideoUrl(libId);
      const meta = { title: script.title, description, videoUrl };
      if (WANT.facebook) { socialPublished.facebook = await attemptPublish('Facebook', () => publishFacebookVideo(meta), '5/5 Publishing to Facebook', 'Uploading the finished video to Facebook…') || ''; if (!socialPublished.facebook) delete socialPublished.facebook; }
      if (WANT.instagram) { socialPublished.instagram = await attemptPublish('Instagram', () => publishInstagramVideo(meta), '5/5 Publishing to Instagram', 'Publishing the finished video as an Instagram Reel…') || ''; if (!socialPublished.instagram) delete socialPublished.instagram; }
      if (WANT.threads) { socialPublished.threads = await attemptPublish('Threads', () => publishThreadsVideo(meta), '5/5 Publishing to Threads', 'Publishing the finished video to Threads…') || ''; if (!socialPublished.threads) delete socialPublished.threads; }
    }
    if (!published && !Object.keys(socialPublished).length) {
      const firstError = Object.entries(publishErrors)[0]?.[1] || 'All selected publishing destinations rejected the video.';
      throw new PipelineError('publish_failed', firstError);
    }
    if (Object.keys(publishErrors).length) log(`⚠️ Partial publish: ${Object.keys(publishErrors).join(', ')} failed; successful destinations will be recorded so the same part is not posted twice.`);
  }

  // 6. Report back — the app records the episode and schedules the next one.
  const episode = await appRequest('POST', `${campaignPath()}/episodes`, {
    partNumber: CFG.partNumber, title: script.title, script: fullText, description,
    youtubeUrl: published?.url || '', videoId: published?.videoId || '', published: !!published || Object.keys(socialPublished).length > 0,
    socialUrls: socialPublished, socialErrors: publishErrors, privacyStatus: published?.privacy || '', format: CFG.format, aspectRatio: CFG.aspect,
    usedFallbackTemplate: script.usedFallbackTemplate, voice: narration.engine, character: characterMode,
    durationSec: Math.round(outDur), runId: CFG.runId, runUrl: CFG.runUrl,
    sources: script.sources || [], model: script.model || '',
    // Story arc: the app keeps this so the next part continues the same story — and ends it at part ${CFG.arcParts}.
    storyPremise: script.premise || '', storyCharacters: script.characters || '', storyTitle: script.title.replace(/\s*\((part \d+|finale)\)\s*$/i, ''),
    arcPart: CFG.partNumber, arcParts: CFG.arcParts, arcComplete: CFG.category === 'stories' && CFG.partNumber >= CFG.arcParts
  });
  if (CFG.campaignId && CFG.appUrl && (!episode || episode.status >= 300)) {
    console.warn(`⚠️ Could not record the episode in the app (${episode ? `HTTP ${episode.status}` : 'app unreachable'}).`);
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  const links = [published?.url, ...Object.values(socialPublished)].filter(Boolean);
  await reportStatus('completed', links.length ? 'Published' : 'Video rendered', 100,
    links.length ? `✅ Part ${CFG.partNumber} published in ${secs}s: ${links.join(' · ')}` : `✅ Part ${CFG.partNumber} rendered in ${secs}s (not published).`,
    { youtubeUrl: published?.url || '', socialUrls: socialPublished, socialErrors: publishErrors });
  log(`Done in ${secs}s.`);
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch(async (err: any) => {
    const code = err instanceof PipelineError ? err.code : 'unexpected';
    // A plain crash (TypeError…) says nothing about WHERE it happened — add the first stack frame from our own code.
    const frame = err instanceof PipelineError ? '' : String(err?.stack || '').split('\n').slice(1).map((l: string) => l.trim()).find((l: string) => /animato-cloud|renderer|runner/.test(l) && !/node_modules/.test(l)) || '';
    const where = frame ? ` [at ${frame.replace(/^at\s+/, '').replace(/^.*[\\/]animato-cloud[\\/]/, 'animato-cloud/').slice(0, 140)}]` : '';
    const message = `${err?.message || String(err)}${where}`;
    console.error(`❌ ${message}`);
    if (!(err instanceof PipelineError)) console.error(err?.stack || err);
    await reportStatus('failed', 'Failed', 0, `❌ ${message}`, { error: message, errorCode: code });
    if (IN_ACTIONS) console.log(`::error title=Animato render failed::${message.replace(/\r?\n/g, ' ')}`);
    process.exit(1);
  });
