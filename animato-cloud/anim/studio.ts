/**
 * Podcast studio scenes — one generator shared by the app (the studio
 * designer's live preview) and the cloud stage (the video), so what you design
 * is what gets rendered.
 *
 * A studio is a ready-made preset plus editable lighting and effects:
 *   accent (LED / neon colour), key-light colour and brightness, haze, bokeh,
 *   vignette, the desk finish and the show name on the sign.
 *
 * Three layers (1920×1080 SVG, cropped to the video's frame by the stage):
 *   back  — walls, shelves, window, sign, lamps (behind the hosts)
 *   front — the desk and one microphone per host (in front of the hosts)
 *   light — the light the room throws on everything (tint, rim glow, vignette)
 */

export type StudioPreset = 'neon_night' | 'warm_wood' | 'clean_white' | 'late_show' | 'brick_loft';
export type DeskFinish = 'wood' | 'black' | 'white';

export interface StudioSpec {
  preset: StudioPreset;
  /** LED strips, neon sign, rim light. */
  accent: string;
  /** Colour of the key light falling on the hosts. */
  keyColor: string;
  /** 0.4 (moody) … 1.4 (bright). */
  brightness: number;
  /** 0 … 1 */
  haze: number;
  /** 0 … 1 — soft out-of-focus lights in the background. */
  bokeh: number;
  /** 0 … 1 */
  vignette: number;
  desk: DeskFinish;
  showName: string;
}

export const STUDIO_PRESETS: { id: StudioPreset; name: string; blurb: string; spec: Omit<StudioSpec, 'showName'> }[] = [
  { id: 'neon_night', name: 'Neon Night', blurb: 'Dark room, glowing LED strips and an ON AIR sign', spec: { preset: 'neon_night', accent: '#a855f7', keyColor: '#ffe9d6', brightness: 0.9, haze: 0.35, bokeh: 0.6, vignette: 0.55, desk: 'black' } },
  { id: 'warm_wood', name: 'Warm Wood', blurb: 'Wooden slats, bookshelves and warm lamps', spec: { preset: 'warm_wood', accent: '#f59e0b', keyColor: '#ffd9a8', brightness: 1.0, haze: 0.15, bokeh: 0.35, vignette: 0.45, desk: 'wood' } },
  { id: 'clean_white', name: 'Clean White', blurb: 'Bright, minimal and modern with a big screen', spec: { preset: 'clean_white', accent: '#06b6d4', keyColor: '#ffffff', brightness: 1.25, haze: 0, bokeh: 0.15, vignette: 0.2, desk: 'white' } },
  { id: 'late_show', name: 'Late Show', blurb: 'Curtains, a city skyline and spotlights', spec: { preset: 'late_show', accent: '#3b82f6', keyColor: '#fff3e0', brightness: 1.05, haze: 0.25, bokeh: 0.5, vignette: 0.5, desk: 'wood' } },
  { id: 'brick_loft', name: 'Brick Loft', blurb: 'Exposed brick, hanging bulbs and records on the wall', spec: { preset: 'brick_loft', accent: '#ef4444', keyColor: '#ffe2b8', brightness: 0.95, haze: 0.2, bokeh: 0.45, vignette: 0.5, desk: 'wood' } },
];

const HEX = /^#[0-9a-f]{6}$/i;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const num = (v: any, d: number, a: number, b: number) => (Number.isFinite(Number(v)) ? clamp(Number(v), a, b) : d);

export function defaultStudio(preset: StudioPreset = 'neon_night', showName = 'The Show'): StudioSpec {
  const p = STUDIO_PRESETS.find((x) => x.id === preset) || STUDIO_PRESETS[0];
  return { ...p.spec, showName };
}

export function normalizeStudio(input: any): StudioSpec {
  const base = defaultStudio(STUDIO_PRESETS.some((p) => p.id === input?.preset) ? input.preset : 'neon_night');
  return {
    preset: base.preset,
    accent: HEX.test(input?.accent) ? input.accent : base.accent,
    keyColor: HEX.test(input?.keyColor) ? input.keyColor : base.keyColor,
    brightness: num(input?.brightness, base.brightness, 0.4, 1.4),
    haze: num(input?.haze, base.haze, 0, 1),
    bokeh: num(input?.bokeh, base.bokeh, 0, 1),
    vignette: num(input?.vignette, base.vignette, 0, 1),
    desk: ['wood', 'black', 'white'].includes(input?.desk) ? input.desk : base.desk,
    showName: String(input?.showName || base.showName).replace(/[<>&"]/g, '').slice(0, 28) || 'The Show',
  };
}

// ---------------------------------------------------------------------------
// Plain-words edits ("make it purple", "warmer light", "more fog", "no bokeh")
// ---------------------------------------------------------------------------
const NAMED: Record<string, string> = {
  red: '#ef4444', orange: '#f97316', amber: '#f59e0b', gold: '#eab308', yellow: '#facc15', lime: '#84cc16', green: '#22c55e', teal: '#14b8a6',
  cyan: '#06b6d4', blue: '#3b82f6', indigo: '#6366f1', purple: '#a855f7', violet: '#8b5cf6', magenta: '#d946ef', pink: '#ec4899', white: '#f8fafc',
};

export function applyStudioPrompt(spec: StudioSpec, text: string): { spec: StudioSpec; changed: string[] } {
  const t = ` ${String(text || '').toLowerCase()} `;
  const s: StudioSpec = { ...spec };
  const changed: string[] = [];
  for (const p of STUDIO_PRESETS) {
    const words = p.name.toLowerCase().split(' ');
    if (words.every((w) => t.includes(w)) || (p.id === 'brick_loft' && /\bbrick\b/.test(t)) || (p.id === 'late_show' && /\b(curtain|skyline|talk show)\b/.test(t)) || (p.id === 'warm_wood' && /\bwood(en)? (wall|studio|room)\b/.test(t)) || (p.id === 'neon_night' && /\bneon (room|studio)\b/.test(t)) || (p.id === 'clean_white' && /\b(white|minimal) (room|studio)\b/.test(t))) {
      Object.assign(s, p.spec); changed.push(`studio: ${p.name}`); break;
    }
  }
  const colour = Object.keys(NAMED).find((c) => new RegExp(`\\b${c}\\b`).test(t));
  if (colour && !/\b(desk|table)\b/.test(t)) {
    if (/\b(key|face|skin) ?light\b/.test(t)) { s.keyColor = NAMED[colour]; changed.push(`key light ${colour}`); }
    else { s.accent = NAMED[colour]; changed.push(`accent ${colour}`); }
  }
  const hex = t.match(/#[0-9a-f]{6}\b/);
  if (hex) { s.accent = hex[0]; changed.push(`accent ${hex[0]}`); }
  if (/\bwarm(er)?\b/.test(t)) { s.keyColor = '#ffd2a0'; changed.push('warmer light'); }
  if (/\b(cool(er)?|cold(er)?)\b/.test(t)) { s.keyColor = '#dbeafe'; changed.push('cooler light'); }
  if (/\b(bright(er)?|lighter|more light)\b/.test(t)) { s.brightness = clamp(s.brightness + 0.2, 0.4, 1.4); changed.push('brighter'); }
  if (/\b(dark(er)?|moody|dim(mer)?|less light)\b/.test(t)) { s.brightness = clamp(s.brightness - 0.2, 0.4, 1.4); changed.push('darker'); }
  const more = (w: string) => new RegExp(`\\b(more|add|lots of|heavy|thick)( \\w+)? ${w}`).test(t);
  const less = (w: string) => new RegExp(`\\b(less|no|remove|without|light)( \\w+)? ${w}`).test(t);
  for (const [key, word] of [['haze', '(haze|fog|smoke)'], ['bokeh', '(bokeh|blur(ry)? lights|background lights)'], ['vignette', '(vignette|dark edges)']] as const) {
    if (less(word)) { (s as any)[key] = /\bno\b|\bremove\b|\bwithout\b/.test(t) ? 0 : clamp((s as any)[key] - 0.3, 0, 1); changed.push(`less ${key}`); }
    else if (more(word)) { (s as any)[key] = clamp((s as any)[key] + 0.3, 0, 1); changed.push(`more ${key}`); }
  }
  const desk = t.match(/\b(wood(en)?|black|white) (desk|table)\b/);
  if (desk) { s.desk = desk[1].startsWith('wood') ? 'wood' : (desk[1] as DeskFinish); changed.push(`${s.desk} desk`); }
  const name = text.match(/(?:call(?:ed)?|name(?:d)?|title(?:d)?|sign(?: says)?)\s*[:"']?\s*["']?([^"'\n]{2,28})["']?\s*$/i);
  if (name) { s.showName = name[1].trim(); changed.push(`sign "${s.showName}"`); }
  return { spec: normalizeStudio(s), changed };
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
function hexRgb(h: string): [number, number, number] { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mix(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexRgb(a), [r2, g2, b2] = hexRgb(b);
  const c = (x: number, y: number) => Math.round(x + (y - x) * t).toString(16).padStart(2, '0');
  return `#${c(r1, r2)}${c(g1, g2)}${c(b1, b2)}`;
}
const seeded = (seed: number) => () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const esc = (s: string) => s.replace(/[<>&"']/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' }[c] as string));
const svg = (body: string, defs = '') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" width="1920" height="1080"><defs>${defs}</defs>${body}</svg>`;

/** Horizontal centre of each seat (fraction of the 1920 width). */
export function seatsFor(count: number): number[] {
  return count >= 3 ? [0.2, 0.5, 0.8] : count === 2 ? [0.3, 0.7] : [0.5];
}
/** Top of the desk (fraction of the 1080 height). */
export const DESK_TOP = 0.8;

function backLayer(s: StudioSpec): string {
  const A = s.accent;
  const R = seeded(7 + s.preset.length);
  let defs = `<linearGradient id="floorG" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.55"/></linearGradient>
<radialGradient id="glowA" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="${A}" stop-opacity="0.85"/><stop offset="1" stop-color="${A}" stop-opacity="0"/></radialGradient>
<radialGradient id="warm" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#ffd08a" stop-opacity="0.9"/><stop offset="1" stop-color="#ffd08a" stop-opacity="0"/></radialGradient>
<filter id="blur8"><feGaussianBlur stdDeviation="8"/></filter><filter id="blur3"><feGaussianBlur stdDeviation="3"/></filter>`;
  let b = '';
  const sign = (x: number, y: number, w: number, h: number, text: string, color: string) =>
    `<g><rect x="${x - 12}" y="${y - 12}" width="${w + 24}" height="${h + 24}" rx="22" fill="url(#glowA)" opacity="0.55" filter="url(#blur8)"/>
<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="#07070c" stroke="${color}" stroke-width="5"/>
<text x="${x + w / 2}" y="${y + h / 2 + h * 0.16}" font-family="Poppins, Arial, sans-serif" font-weight="800" font-size="${Math.round(h * 0.46)}" fill="${color}" text-anchor="middle" letter-spacing="3">${esc(text)}</text></g>`;
  const ledStrip = (y: number) => `<rect x="0" y="${y - 4}" width="1920" height="8" fill="${A}"/><rect x="0" y="${y - 30}" width="1920" height="60" fill="${A}" opacity="0.25" filter="url(#blur8)"/>`;
  const plant = (x: number, y: number, k = 1) => `<g transform="translate(${x},${y}) scale(${k})"><path d="M0,0 Q-60,-120 -10,-230 Q10,-120 0,0" fill="#1f6f3f"/><path d="M0,0 Q70,-110 40,-210 Q10,-110 0,0" fill="#2f8f52"/><path d="M0,0 Q-100,-60 -120,-150 Q-40,-80 0,0" fill="#237a47"/><path d="M-40,0 h80 l-12,90 h-56 z" fill="#3a2a22"/></g>`;
  const shelf = (x: number, y: number, w: number) => {
    let books = '';
    let bx = x + 10;
    while (bx < x + w - 30) { const bw = 14 + R() * 22, bh = 60 + R() * 50; const hue = ['#b91c1c', '#1d4ed8', '#15803d', '#a16207', '#6d28d9', '#0f766e', '#9f1239', '#334155'][Math.floor(R() * 8)]; books += `<rect x="${bx.toFixed(0)}" y="${(y - bh).toFixed(0)}" width="${bw.toFixed(0)}" height="${bh.toFixed(0)}" fill="${hue}" rx="2"/>`; bx += bw + 3; if (R() > 0.85) bx += 40; }
    return `${books}<rect x="${x}" y="${y}" width="${w}" height="14" fill="#2a1a10"/>`;
  };
  switch (s.preset) {
    case 'neon_night': {
      b += `<rect width="1920" height="1080" fill="#0b0b14"/>`;
      // Acoustic foam panels
      for (let i = 0; i < 6; i++) for (let j = 0; j < 3; j++) {
        const x = 140 + i * 280, y = 150 + j * 150;
        if (i === 2 || i === 3) continue;
        b += `<rect x="${x}" y="${y}" width="250" height="130" rx="8" fill="#16162a"/><path d="M${x} ${y + 30} h250 M${x} ${y + 65} h250 M${x} ${y + 100} h250" stroke="#0e0e1c" stroke-width="10"/>`;
      }
      // City window behind the centre
      b += `<rect x="700" y="130" width="520" height="440" rx="10" fill="#0a1330"/>`;
      for (let i = 0; i < 16; i++) { const w = 26 + R() * 40, h = 90 + R() * 260, x = 705 + R() * 480; b += `<rect x="${x.toFixed(0)}" y="${(570 - h).toFixed(0)}" width="${w.toFixed(0)}" height="${h.toFixed(0)}" fill="#111a3a"/>`; for (let k = 0; k < 6; k++) if (R() > 0.45) b += `<rect x="${(x + 4 + R() * (w - 10)).toFixed(0)}" y="${(570 - h + 10 + R() * (h - 20)).toFixed(0)}" width="5" height="7" fill="#fde68a" opacity="0.8"/>`; }
      b += `<rect x="700" y="130" width="520" height="440" rx="10" fill="none" stroke="#1d1d33" stroke-width="18"/><line x1="960" y1="130" x2="960" y2="570" stroke="#1d1d33" stroke-width="12"/>`;
      b += sign(820, 40, 280, 70, 'ON AIR', '#ff3b5c');
      b += ledStrip(640) + `<rect x="0" y="640" width="1920" height="440" fill="#0d0d18"/>`;
      b += sign(1360, 60, 380, 76, s.showName, s.accent);
      b += plant(110, 700, 1.2) + plant(1810, 700, 1.1);
      break;
    }
    case 'warm_wood': {
      b += `<rect width="1920" height="1080" fill="#3a2518"/>`;
      for (let x = 0; x < 1920; x += 48) b += `<rect x="${x}" y="0" width="40" height="700" fill="${mix('#6b4226', '#8a5a36', R())}"/>`;
      b += shelf(90, 300, 420) + shelf(90, 480, 420) + shelf(1410, 300, 420) + shelf(1410, 480, 420);
      b += `<ellipse cx="660" cy="250" rx="150" ry="150" fill="url(#warm)" opacity="0.8"/><ellipse cx="1260" cy="250" rx="150" ry="150" fill="url(#warm)" opacity="0.8"/>`;
      b += `<path d="M640,120 h40 l30,90 h-100 z" fill="#1c130c"/><path d="M1240,120 h40 l30,90 h-100 z" fill="#1c130c"/>`;
      b += sign(760, 120, 400, 110, s.showName, s.accent);
      b += `<rect x="0" y="700" width="1920" height="380" fill="#24160d"/>`;
      b += plant(1850, 700, 1.25) + plant(60, 700, 1.1);
      break;
    }
    case 'clean_white': {
      b += `<rect width="1920" height="1080" fill="#eef1f5"/><rect x="0" y="0" width="1920" height="700" fill="#f7f9fb"/>`;
      b += `<rect x="560" y="120" width="800" height="440" rx="18" fill="#0f172a"/><rect x="580" y="140" width="760" height="400" rx="10" fill="${mix(s.accent, '#0f172a', 0.55)}"/>`;
      b += `<text x="960" y="370" font-family="Poppins, Arial, sans-serif" font-weight="800" font-size="84" fill="#ffffff" text-anchor="middle">${esc(s.showName)}</text><rect x="760" y="400" width="400" height="10" rx="5" fill="${s.accent}"/>`;
      b += `<rect x="120" y="160" width="300" height="420" rx="14" fill="#e3e8ef"/><rect x="1500" y="160" width="300" height="420" rx="14" fill="#e3e8ef"/>`;
      b += `<rect x="0" y="700" width="1920" height="380" fill="#dfe4ea"/><rect x="0" y="696" width="1920" height="8" fill="${s.accent}" opacity="0.8"/>`;
      b += plant(270, 700, 1.1) + plant(1650, 700, 1.1);
      break;
    }
    case 'late_show': {
      b += `<rect width="1920" height="1080" fill="#060b1d"/>`;
      // Skyline backdrop
      b += `<rect x="0" y="0" width="1920" height="700" fill="#0b1640"/>`;
      for (let i = 0; i < 40; i++) { const w = 40 + R() * 70, h = 120 + R() * 380, x = R() * 1880; b += `<rect x="${x.toFixed(0)}" y="${(700 - h).toFixed(0)}" width="${w.toFixed(0)}" height="${h.toFixed(0)}" fill="${mix('#101d52', '#1a2a6c', R())}"/>`; for (let k = 0; k < 8; k++) if (R() > 0.5) b += `<rect x="${(x + 5 + R() * (w - 12)).toFixed(0)}" y="${(700 - h + 12 + R() * (h - 24)).toFixed(0)}" width="6" height="8" fill="#fef3c7" opacity="0.85"/>`; }
      b += `<circle cx="1500" cy="160" r="60" fill="#fef9c3" opacity="0.9"/>`;
      // Curtains
      for (const side of [0, 1]) { let c = ''; for (let i = 0; i < 7; i++) c += `<rect x="${side ? 1920 - 260 + i * 38 : i * 38}" y="0" width="34" height="760" fill="${i % 2 ? '#7f1d1d' : '#991b1b'}"/>`; b += c; }
      b += `<rect x="0" y="0" width="1920" height="60" fill="#450a0a"/>`;
      b += sign(710, 90, 500, 100, s.showName, '#fde68a');
      b += `<rect x="0" y="700" width="1920" height="380" fill="#0a0f24"/>`;
      break;
    }
    case 'brick_loft': {
      b += `<rect width="1920" height="1080" fill="#5a2a1c"/>`;
      for (let row = 0; row < 18; row++) for (let col = -1; col < 22; col++) { const x = col * 96 + (row % 2) * 48, y = row * 40; b += `<rect x="${x}" y="${y}" width="90" height="34" fill="${mix('#7a3522', '#9c4a2e', R())}"/>`; }
      for (let i = 0; i < 5; i++) { const x = 300 + i * 330; b += `<line x1="${x}" y1="0" x2="${x}" y2="${140 + (i % 2) * 60}" stroke="#111" stroke-width="3"/><circle cx="${x}" cy="${150 + (i % 2) * 60}" r="60" fill="url(#warm)"/><circle cx="${x}" cy="${150 + (i % 2) * 60}" r="13" fill="#ffe6a8"/>`; }
      for (let i = 0; i < 4; i++) { const x = 1530 + (i % 2) * 190, y = 250 + Math.floor(i / 2) * 190; b += `<circle cx="${x}" cy="${y}" r="80" fill="#111"/><circle cx="${x}" cy="${y}" r="26" fill="${i % 2 ? s.accent : '#f5f5f4'}"/><circle cx="${x}" cy="${y}" r="5" fill="#111"/>`; }
      b += sign(710, 40, 500, 100, s.showName, s.accent);
      b += `<rect x="0" y="700" width="1920" height="380" fill="#2a1710"/>`;
      b += plant(1830, 700, 1.2);
      break;
    }
  }
  b += `<rect x="0" y="700" width="1920" height="380" fill="url(#floorG)"/>`;
  return svg(b, defs);
}

function frontLayer(s: StudioSpec, count: number): string {
  const top = DESK_TOP * 1080;
  const deskFill = s.desk === 'wood' ? ['#6b4226', '#4a2c18'] : s.desk === 'white' ? ['#f1f5f9', '#cbd5e1'] : ['#1f2937', '#0b0f17'];
  const defs = `<linearGradient id="deskG" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${deskFill[0]}"/><stop offset="1" stop-color="${deskFill[1]}"/></linearGradient>
<linearGradient id="mesh" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#1b1b1f"/><stop offset="0.5" stop-color="#4b4b55"/><stop offset="1" stop-color="#1b1b1f"/></linearGradient>`;
  let b = `<path d="M-40 ${top} L1960 ${top} L1960 1120 L-40 1120 Z" fill="url(#deskG)"/><rect x="-40" y="${top}" width="2000" height="10" fill="#ffffff" opacity="${s.desk === 'white' ? 0.9 : 0.18}"/>`;
  b += `<rect x="-40" y="${top + 10}" width="2000" height="6" fill="${s.accent}" opacity="0.7"/>`;
  for (const f of seatsFor(count)) {
    const x = f * 1920;
    // Boom-arm microphone, slightly to the side of the host's mouth.
    const mx = x + (f < 0.5 ? 150 : f > 0.5 ? -150 : 150);
    b += `<g><rect x="${mx - 6}" y="${top - 170}" width="12" height="170" fill="#222"/><rect x="${mx - 40}" y="${top - 4}" width="80" height="12" rx="6" fill="#111"/>
<rect x="${mx - 34}" y="${top - 300}" width="68" height="140" rx="34" fill="url(#mesh)"/><rect x="${mx - 38}" y="${top - 180}" width="76" height="14" rx="7" fill="#111"/>
<rect x="${mx - 30}" y="${top - 292}" width="60" height="6" rx="3" fill="#666" opacity="0.6"/></g>`;
    // Mug / laptop details
    b += f !== 0.5 ? `<rect x="${x + (f < 0.5 ? -170 : 120)}" y="${top - 40}" width="44" height="46" rx="6" fill="#e5e7eb"/><path d="M${x + (f < 0.5 ? -126 : 164)} ${top - 30} q18 12 0 24" stroke="#e5e7eb" stroke-width="6" fill="none"/>` : '';
  }
  return svg(b, defs);
}

function lightLayer(s: StudioSpec): string {
  const dark = 1 - clamp(s.brightness, 0.4, 1.4);
  const defs = `<radialGradient id="vig" cx="0.5" cy="0.48" r="0.75"><stop offset="0.55" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="${(0.85 * s.vignette).toFixed(2)}"/></radialGradient>
<radialGradient id="key" cx="0.5" cy="0.35" r="0.6"><stop offset="0" stop-color="${s.keyColor}" stop-opacity="${(0.16 * s.brightness).toFixed(2)}"/><stop offset="1" stop-color="${s.keyColor}" stop-opacity="0"/></radialGradient>
<linearGradient id="rimL" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${s.accent}" stop-opacity="0.28"/><stop offset="0.35" stop-color="${s.accent}" stop-opacity="0"/></linearGradient>
<linearGradient id="rimR" x1="1" y1="0" x2="0" y2="0"><stop offset="0" stop-color="${s.accent}" stop-opacity="0.28"/><stop offset="0.35" stop-color="${s.accent}" stop-opacity="0"/></linearGradient>`;
  let b = `<rect width="1920" height="1080" fill="url(#key)"/><rect width="1920" height="1080" fill="url(#rimL)"/><rect width="1920" height="1080" fill="url(#rimR)"/>`;
  if (dark > 0) b += `<rect width="1920" height="1080" fill="#000" opacity="${(dark * 0.6).toFixed(2)}"/>`;
  else b += `<rect width="1920" height="1080" fill="#fff" opacity="${(-dark * 0.18).toFixed(2)}"/>`;
  b += `<rect width="1920" height="1080" fill="url(#vig)"/>`;
  return svg(b, defs);
}

export function studioLayers(spec: StudioSpec, hostCount: number): { back: string; front: string; light: string } {
  const s = normalizeStudio(spec);
  return { back: backLayer(s), front: frontLayer(s, hostCount), light: lightLayer(s) };
}

/** Deterministic bokeh / haze particles (the stage animates them; the preview draws them still). */
export function bokehDots(spec: StudioSpec, n = 26): { x: number; y: number; r: number; a: number; phase: number }[] {
  const R = seeded(99);
  const count = Math.round(n * normalizeStudio(spec).bokeh);
  return Array.from({ length: count }, () => ({ x: R(), y: 0.05 + R() * 0.6, r: 18 + R() * 46, a: 0.12 + R() * 0.25, phase: R() * Math.PI * 2 }));
}

/**
 * One flat SVG of the whole studio (for the app's preview), with simple host
 * silhouettes in the given colours at their seats.
 */
export function studioPreviewSvg(spec: StudioSpec, hosts: { skin: string; hair: string; outfit: string }[]): string {
  const s = normalizeStudio(spec);
  const inner = (x: string) => x.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '');
  const seats = seatsFor(hosts.length || 2);
  let people = '';
  (hosts.length ? hosts : [{ skin: '#c68642', hair: '#2b1d14', outfit: '#334155' }, { skin: '#8d5524', hair: '#111', outfit: '#7c2d12' }]).forEach((h, i) => {
    const x = (seats[i] ?? 0.5) * 1920, top = DESK_TOP * 1080;
    people += `<g><path d="M${x - 190} ${top + 30} Q${x - 180} ${top - 260} ${x} ${top - 270} Q${x + 180} ${top - 260} ${x + 190} ${top + 30} Z" fill="${h.outfit}"/>
<rect x="${x - 34}" y="${top - 330}" width="68" height="80" fill="${h.skin}"/><ellipse cx="${x}" cy="${top - 420}" rx="92" ry="112" fill="${h.skin}"/>
<path d="M${x - 96} ${top - 420} Q${x - 100} ${top - 545} ${x} ${top - 545} Q${x + 100} ${top - 545} ${x + 96} ${top - 420} Q${x + 60} ${top - 490} ${x} ${top - 495} Q${x - 60} ${top - 490} ${x - 96} ${top - 420} Z" fill="${h.hair}"/>
<ellipse cx="${x - 32}" cy="${top - 420}" rx="10" ry="12" fill="#1b1b1b"/><ellipse cx="${x + 32}" cy="${top - 420}" rx="10" ry="12" fill="#1b1b1b"/><path d="M${x - 30} ${top - 370} Q${x} ${top - 350} ${x + 30} ${top - 370}" stroke="#7a2e2e" stroke-width="7" fill="none" stroke-linecap="round"/></g>`;
  });
  const dots = bokehDots(s).map((d) => `<circle cx="${(d.x * 1920).toFixed(0)}" cy="${(d.y * 1080).toFixed(0)}" r="${d.r.toFixed(0)}" fill="${s.accent}" opacity="${d.a.toFixed(2)}"/>`).join('');
  const L = studioLayers(s, hosts.length || 2);
  const haze = s.haze > 0 ? `<rect width="1920" height="1080" fill="#ffffff" opacity="${(s.haze * 0.12).toFixed(2)}"/>` : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" width="1920" height="1080"><g>${inner(L.back)}</g><g filter="url(#pvBlur)">${dots}</g>${haze}<g>${people}</g><g>${inner(L.front)}</g><g>${inner(L.light)}</g><defs><filter id="pvBlur"><feGaussianBlur stdDeviation="6"/></filter></defs></svg>`;
}
