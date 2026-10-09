/**
 * Stickman 3D arenas + a cinematic camera.
 *
 *  - A pinhole camera looks at the fight plane (z = 0) from distance D with a
 *    height (horizon), a yaw (orbit around the fight) and a roll. Fighters stay
 *    flat sprites on the fight plane, but every point is projected, so an orbit
 *    makes the far fighter smaller and higher, the floor grid swings towards a
 *    moving vanishing point and the background layers slide at their true depth.
 *  - Arenas are drawn from layers at real depths: sky, far silhouettes, a
 *    perspective floor, mid props, and foreground props that pass the lens;
 *    plus weather particles, fog and rim light for dark sets.
 *  - A shot director cuts between wide, medium, orbit, low-angle, high-angle,
 *    dutch, push-in and crash-zoom shots on the beats of the fight, with
 *    bullet-time orbits around slow-motion finishers. Everything is seeded, so a
 *    render is reproducible but no two rounds are shot the same way.
 */

export type Ctx = CanvasRenderingContext2D;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const smooth = (t: number) => { const x = clamp(t, 0, 1); return x * x * (3 - 2 * x); };
const hashN = (n: number) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// ---------------------------------------------------------------------------
// Camera / projection
// ---------------------------------------------------------------------------
export interface Cam3 { W: number; H: number; camX: number; scale: number; groundY: number; horizon: number; yaw: number; D: number }
export interface Pt { x: number; y: number; s: number; z: number }
export interface Proj {
  cam: Cam3; f: number; camH: number;
  /** World point (x along the fight, y up, z into the screen; fight plane z = 0). */
  pt: (X: number, Y: number, Z: number) => Pt | null;
  /** The fight plane at x: screen x, the ground line under it, pixels per unit there. */
  ground: (X: number) => { x: number; gy: number; s: number };
  /** Half-width (units) of the view at depth Z (for placing props). */
  span: (Z: number) => number;
}

export function makeProj(c: Cam3): Proj {
  const f = c.scale * c.D;
  const camH = (c.groundY - c.horizon) / c.scale;
  const sy = Math.sin(c.yaw), cy = Math.cos(c.yaw);
  const pt = (X: number, Y: number, Z: number): Pt | null => {
    const dx = X - c.camX;
    const xp = dx * cy - Z * sy, zp = dx * sy + Z * cy + c.D;
    if (zp < 30) return null;
    const k = f / zp;
    return { x: c.W / 2 + xp * k, y: c.horizon + (camH - Y) * k, s: k, z: zp };
  };
  const ground = (X: number) => {
    const dx = X - c.camX;
    const zp = Math.max(c.D * 0.35, dx * sy + c.D);
    const k = f / zp;
    return { x: c.W / 2 + dx * cy * k, gy: c.horizon + camH * k, s: k };
  };
  const span = (Z: number) => ((c.W * 0.75) / f) * (Z + c.D) / Math.max(0.3, cy) + Math.abs(Z * sy) + 200;
  return { cam: c, f, camH, pt, ground, span };
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
function hexRgb(h: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(h || '');
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
/** Mix a colour towards the fog colour (k = 0..1). */
export function mix(a: string, b: string, k: number): string {
  const A = hexRgb(a), B = hexRgb(b), t = clamp(k, 0, 1);
  return `rgb(${Math.round(lerp(A[0], B[0], t))},${Math.round(lerp(A[1], B[1], t))},${Math.round(lerp(A[2], B[2], t))})`;
}
const rgba = (h: string, a: number) => { const [r, g, b] = hexRgb(h); return `rgba(${r},${g},${b},${a})`; };

// ---------------------------------------------------------------------------
// Arena definitions
// ---------------------------------------------------------------------------
export interface Arena {
  id: string; name: string;
  sky: string[];               // top → horizon
  fog: string; fogDist: number; // depth fog colour and distance (units)
  floor: [string, string];     // near, far
  grid: { color: string; alpha: number; tile: number; along?: boolean; across?: boolean; glow?: boolean };
  reflect: number;             // floor reflection strength (wet / glass floors)
  dark: boolean;               // fighters get a rim light
  rim: string;
  groundInk: string;           // rubble, cracks, debris
  dust: string;                // dust clouds
  particles?: { kind: 'snow' | 'rain' | 'embers' | 'petals' | 'dust' | 'sparks' | 'leaves'; color: string; count: number };
  far: (ctx: Ctx, P: Proj, t: number, A: Arena) => void;
  mid: (ctx: Ctx, P: Proj, t: number, A: Arena, front: boolean) => void;
}

/** Repeating props along x at depth z (seeded per slot). */
function forSlots(P: Proj, Z: number, spacing: number, seed: number, fn: (X: number, k: number, r: number) => void, halfWidth = 0) {
  const c = P.cam.camX, half = P.span(Z);
  const k0 = Math.floor((c - half) / spacing), k1 = Math.ceil((c + half) / spacing);
  for (let k = k0; k <= k1; k++) {
    const r = hashN(k * 13.7 + seed), X = k * spacing + (r - 0.5) * spacing * 0.5;
    // Props between the camera and the fight only frame the shot: never in the middle where the fighters are.
    if (Z < 0) { const p = P.pt(X, 0, Z); if (!p || Math.abs(p.x - P.cam.W / 2) - halfWidth * p.s < P.cam.W * 0.3) continue; }
    fn(X, k, r);
  }
}
const fogK = (A: Arena, z: number) => 1 - Math.exp(-Math.max(0, z) / A.fogDist);

/** A camera-facing box standing on the floor (pillar, building, crate…). */
function box(ctx: Ctx, P: Proj, A: Arena, X: number, Z: number, w: number, h: number, color: string, y0 = 0) {
  const b = P.pt(X, y0, Z); if (!b) return null;
  const fc = mix(color, A.fog, fogK(A, b.z));
  const x = b.x - (w / 2) * b.s, top = b.y - h * b.s;
  ctx.fillStyle = fc; ctx.fillRect(x, top, w * b.s, h * b.s);
  return { x, top, w: w * b.s, h: h * b.s, s: b.s, z: b.z, base: b.y };
}
/** A far ridge (mountains, skyline, dunes) as one polygon at depth Z. */
function ridge(ctx: Ctx, P: Proj, A: Arena, Z: number, step: number, amp: number, base: number, seed: number, color: string, sharp = false) {
  const c = P.cam.camX, half = P.span(Z) + step * 2;
  ctx.beginPath();
  let first = true, lastX = 0;
  for (let X = Math.floor((c - half) / step) * step; X <= c + half; X += step) {
    const k = X / step;
    const h = base + amp * (sharp ? Math.abs(hashN(k + seed) - 0.5) * 2 : 0.5 + 0.5 * Math.sin(k * 0.9 + seed) * hashN(k * 0.37 + seed));
    const p = P.pt(X, h, Z); if (!p) continue;
    if (first) { const g = P.pt(X, -2000, Z); ctx.moveTo(p.x, (g?.y ?? p.y) + 2); first = false; }
    ctx.lineTo(p.x, p.y); lastX = X;
  }
  const g = P.pt(lastX, -2000, Z); if (g) ctx.lineTo(g.x, g.y + 2);
  ctx.closePath(); ctx.fillStyle = mix(color, A.fog, fogK(A, Z + P.cam.D) * 0.85); ctx.fill();
}
function glowDot(ctx: Ctx, x: number, y: number, r: number, color: string, a = 1) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, rgba(color, 0.9 * a)); g.addColorStop(0.35, rgba(color, 0.35 * a)); g.addColorStop(1, rgba(color, 0));
  ctx.fillStyle = g; ctx.fillRect(x - r, y - r, r * 2, r * 2);
}
function disc(ctx: Ctx, P: Proj, X: number, Y: number, Z: number, r: number, color: string, glow = 0) {
  const p = P.pt(X, Y, Z); if (!p) return;
  if (glow) glowDot(ctx, p.x, p.y, r * p.s * glow, color, 0.8);
  ctx.beginPath(); ctx.arc(p.x, p.y, r * p.s, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

const SKY_FAR = 40000;

export const ARENAS: Record<string, Arena> = {
  dojo: {
    id: 'dojo', name: 'Dojo', sky: ['#3b2412', '#5b3a1e', '#7c5230'], fog: '#6b4a2b', fogDist: 5200,
    floor: ['#b8844c', '#7a5230'], grid: { color: '#5a3a1c', alpha: 0.55, tile: 70, along: true, across: false },
    reflect: 0.12, dark: false, rim: '#fde68a', groundInk: '#3b2412', dust: '#d6b48a',
    particles: { kind: 'dust', color: '#fde68a', count: 46 },
    far: (ctx, P, t, A) => {
      // Back wall: shoji panels with lattice, dark beams; a scroll and lanterns.
      const Z = 900;
      const wall = box(ctx, P, A, P.cam.camX, Z, P.span(Z) * 2.4, 520, '#e9d8b4');
      if (!wall) return;
      forSlots(P, Z, 200, 3, (X) => {
        const p = box(ctx, P, A, X, Z, 14, 520, '#4a2e14'); if (!p) return;
        for (let y = 60; y < 480; y += 70) { const q = P.pt(X - 100, y, Z), q2 = P.pt(X + 100, y, Z); if (q && q2) { ctx.fillStyle = mix('#8a6a44', A.fog, 0.3); ctx.fillRect(q.x, q.y, q2.x - q.x, Math.max(1, 3 * q.s)); } }
      });
      box(ctx, P, A, P.cam.camX, Z, P.span(Z) * 2.4, 40, '#3a2410', 520);
      box(ctx, P, A, P.cam.camX, Z, P.span(Z) * 2.4, 26, '#3a2410', 0);
      forSlots(P, 880, 900, 5, (X, k, r) => { if (r < 0.5) { const s = box(ctx, P, A, X, 880, 110, 230, '#f5ecd7', 160); if (s) { ctx.fillStyle = mix('#1f1208', A.fog, 0.2); ctx.font = `900 ${Math.max(8, 70 * s.s)}px serif`; ctx.textAlign = 'center'; ctx.fillText(['武', '道', '心', '力'][k & 3], s.x + s.w / 2, s.top + s.h * 0.55); } } });
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) forSlots(P, 420, 340, 7, (X) => { const p = P.pt(X, 330 + Math.sin(t * 1.4 + X) * 4, 420); if (!p) return; glowDot(ctx, p.x, p.y, 120 * p.s, '#ffb35c', 0.55); ctx.fillStyle = '#c2410c'; ctx.beginPath(); ctx.ellipse(p.x, p.y, 22 * p.s, 30 * p.s, 0, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#fde68a'; ctx.beginPath(); ctx.ellipse(p.x, p.y, 14 * p.s, 22 * p.s, 0, 0, Math.PI * 2); ctx.fill(); });
      else forSlots(P, -520, 900, 11, (X, k, r) => { if (r < 0.45) box(ctx, P, A, X, -520, 46, 900, '#2a1a0c'); }, 23);
    },
  },
  rooftop: {
    id: 'rooftop', name: 'Night Rooftop', sky: ['#05070f', '#141638', '#3b2a6b'], fog: '#2a2550', fogDist: 9000,
    floor: ['#3a3a44', '#1f1f2a'], grid: { color: '#55556a', alpha: 0.5, tile: 160, along: true, across: true },
    reflect: 0.16, dark: true, rim: '#e0e7ff', groundInk: '#18181f', dust: '#6b7280',
    particles: { kind: 'dust', color: '#c7d2fe', count: 30 },
    far: (ctx, P, t, A) => {
      disc(ctx, P, P.cam.camX * 0.2 + 3000, 7000, SKY_FAR, 900, '#f8fafc', 3.2);
      // Skyline with lit windows (two depths).
      for (const [Z, seed, col, hmax] of [[14000, 1, '#1e1b4b', 2600], [7000, 2, '#141233', 1700]] as [number, number, string, number][]) {
        forSlots(P, Z, 420, seed, (X, k, r) => {
          const h = 500 + r * hmax, w = 300 + hashN(k + seed) * 160;
          const b = box(ctx, P, A, X, Z, w, h, col); if (!b || b.w < 2) return;
          if (b.w > 6) for (let y = 0; y < 14; y++) for (let x = 0; x < 4; x++) {
            if (hashN(k * 31 + y * 7 + x + seed) < 0.55) continue;
            ctx.fillStyle = hashN(k + y + x) < 0.2 ? '#fde68a' : '#93c5fd';
            ctx.globalAlpha = 0.55 + 0.35 * Math.sin(t * 0.7 + k + y);
            ctx.fillRect(b.x + b.w * (0.12 + x * 0.21), b.top + b.h * (0.06 + y * 0.065), b.w * 0.1, b.h * 0.025);
          }
          ctx.globalAlpha = 1;
        });
      }
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) {
        forSlots(P, 1500, 1400, 4, (X, k, r) => {
          if (r < 0.5) { const tank = box(ctx, P, A, X, 1500, 260, 220, '#3f3a52', 260); box(ctx, P, A, X - 100, 1500, 16, 260, '#2a2638'); box(ctx, P, A, X + 100, 1500, 16, 260, '#2a2638'); if (tank) { ctx.fillStyle = mix('#57507a', A.fog, 0.4); ctx.beginPath(); ctx.moveTo(tank.x, tank.top); ctx.lineTo(tank.x + tank.w / 2, tank.top - 50 * tank.s); ctx.lineTo(tank.x + tank.w, tank.top); ctx.fill(); } }
          else { const n = box(ctx, P, A, X, 1500, 360, 120, '#151325', 380); if (n) { ctx.save(); glowDot(ctx, n.x + n.w / 2, n.top + n.h * 0.5, n.w * 0.7, '#f0abfc', 0.5); ctx.fillStyle = '#f0abfc'; ctx.font = `900 ${Math.max(6, 80 * n.s)}px sans-serif`; ctx.textAlign = 'center'; ctx.globalAlpha = 0.75 + 0.25 * Math.sin(t * 9 + k); ctx.fillText(['OPEN', 'BAR', 'HOTEL', 'NEON'][k & 3], n.x + n.w / 2, n.top + n.h * 0.72); ctx.restore(); } box(ctx, P, A, X, 1500, 14, 380, '#1c1a2e'); }
        });
        // Railing at the roof edge
        const rail = P.span(700);
        for (let X = Math.floor((P.cam.camX - rail) / 120) * 120; X < P.cam.camX + rail; X += 120) box(ctx, P, A, X, 700, 8, 110, '#71717a');
        box(ctx, P, A, P.cam.camX, 700, rail * 2.2, 8, '#a1a1aa', 106);
      } else forSlots(P, -600, 1300, 9, (X, k, r) => { if (r < 0.35) { box(ctx, P, A, X, -600, 220, 160, '#0b0b12'); } }, 110);
    },
  },
  cyber: {
    id: 'cyber', name: 'Neon Grid', sky: ['#020208', '#0d0221', '#2d0b52'], fog: '#1a0638', fogDist: 7000,
    floor: ['#07070f', '#0b0418'], grid: { color: '#22d3ee', alpha: 0.85, tile: 120, along: true, across: true, glow: true },
    reflect: 0.38, dark: true, rim: '#a5f3fc', groundInk: '#0e7490', dust: '#67e8f9',
    particles: { kind: 'sparks', color: '#67e8f9', count: 60 },
    far: (ctx, P, t, A) => {
      disc(ctx, P, P.cam.camX * 0.1, 4200, SKY_FAR, 2600, '#f472b6', 1.6);
      // Sun bands
      const p = P.pt(P.cam.camX * 0.1, 4200, SKY_FAR);
      if (p) { ctx.fillStyle = '#0d0221'; for (let i = 0; i < 7; i++) ctx.fillRect(p.x - 2600 * p.s, p.y + (i * 0.14 - 0.1) * 2600 * p.s, 5200 * p.s, (0.025 + i * 0.012) * 2600 * p.s); }
      // Wireframe mountains
      ctx.save(); ctx.strokeStyle = '#e879f9'; ctx.lineWidth = 2;
      const Z = 16000, step = 1400, c = P.cam.camX, half = P.span(Z);
      ctx.beginPath(); let first = true;
      for (let X = Math.floor((c - half) / step) * step; X <= c + half; X += step) { const h = 400 + Math.abs(hashN(X / step) - 0.5) * 6000; const q = P.pt(X, h, Z); if (!q) continue; if (first) { ctx.moveTo(q.x, q.y); first = false; } else ctx.lineTo(q.x, q.y); }
      ctx.stroke(); ctx.restore();
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) forSlots(P, 1200, 700, 6, (X, k, r) => {
        const h = 500 + r * 700;
        const b = box(ctx, P, A, X, 1200, 70, h, '#0b1020'); if (!b) return;
        ctx.save(); ctx.fillStyle = r < 0.5 ? '#22d3ee' : '#e879f9';
        ctx.globalAlpha = 0.7 + 0.3 * Math.sin(t * 3 + k);
        ctx.fillRect(b.x + b.w * 0.42, b.top, b.w * 0.16, b.h); ctx.restore();
        // Hologram ring
        const ring = P.pt(X, h * 0.7 + Math.sin(t * 2 + k) * 30, 1200);
        if (ring) { ctx.save(); ctx.strokeStyle = rgba('#67e8f9', 0.6); ctx.lineWidth = 2; ctx.beginPath(); ctx.ellipse(ring.x, ring.y, 120 * ring.s, 26 * ring.s, 0, 0, Math.PI * 2); ctx.stroke(); ctx.restore(); }
      });
      else forSlots(P, -650, 1200, 13, (X, k, r) => { if (r < 0.3) { const b = box(ctx, P, A, X, -650, 60, 1400, '#050510'); if (b) { ctx.fillStyle = '#e879f9'; ctx.fillRect(b.x + b.w * 0.45, b.top, b.w * 0.1, b.h); } } }, 30);
    },
  },
  temple: {
    id: 'temple', name: 'Forest Temple', sky: ['#9fb8e8', '#f7c6b5', '#fde7c2'], fog: '#f3dcc6', fogDist: 4200,
    floor: ['#a8a29e', '#8a817c'], grid: { color: '#6b625c', alpha: 0.45, tile: 150, along: true, across: true },
    reflect: 0, dark: false, rim: '#ffffff', groundInk: '#57534e', dust: '#e7e5e4',
    particles: { kind: 'petals', color: '#f9a8d4', count: 70 },
    far: (ctx, P, t, A) => {
      disc(ctx, P, P.cam.camX * 0.15 - 2000, 5200, SKY_FAR, 1500, '#fff7ed', 2.2);
      ridge(ctx, P, A, 22000, 1800, 5200, 1600, 3, '#7c8db5', true);
      ridge(ctx, P, A, 12000, 1100, 2400, 700, 9, '#5f7a8c');
      ridge(ctx, P, A, 6000, 500, 900, 600, 4, '#3f6b4f');
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) {
        forSlots(P, 1800, 1600, 2, (X, k, r) => {
          if (r < 0.55) { // torii gate
            const red = '#c2271c';
            box(ctx, P, A, X - 260, 1800, 40, 620, red); box(ctx, P, A, X + 260, 1800, 40, 620, red);
            box(ctx, P, A, X, 1800, 720, 50, '#1f1f1f', 620); box(ctx, P, A, X, 1800, 600, 34, red, 520);
          } else { // cherry tree
            box(ctx, P, A, X, 1800, 50, 380, '#4a3328');
            for (let i = 0; i < 6; i++) { const p = P.pt(X + (hashN(k + i) - 0.5) * 420, 420 + hashN(k * 3 + i) * 220, 1800); if (p) { ctx.fillStyle = mix(i % 2 ? '#f9a8d4' : '#f472b6', A.fog, fogK(A, p.z) * 0.8); ctx.beginPath(); ctx.arc(p.x, p.y, (110 + hashN(i + k) * 80) * p.s, 0, Math.PI * 2); ctx.fill(); } }
          }
        });
        forSlots(P, 700, 900, 8, (X, k, r) => { if (r < 0.6) { box(ctx, P, A, X, 700, 70, 120, '#78716c'); box(ctx, P, A, X, 700, 120, 40, '#57534e', 120); box(ctx, P, A, X, 700, 60, 50, '#a8a29e', 160); const p = P.pt(X, 185, 700); if (p) glowDot(ctx, p.x, p.y, 50 * p.s, '#fbbf24', 0.45); box(ctx, P, A, X, 700, 140, 26, '#44403c', 210); } });
      } else forSlots(P, -480, 200, 15, (X, k, r) => { if (r < 0.33) { const b = box(ctx, P, A, X, -480, 26, 2200, '#2f5d34'); if (b) for (let y = 0; y < 2200; y += 260) { const q = P.pt(X, y, -480); if (q) { ctx.fillStyle = '#1e3d22'; ctx.fillRect(b.x - 2, q.y, b.w + 4, Math.max(2, 10 * q.s)); } } } }, 13);
    },
  },
  canyon: {
    id: 'canyon', name: 'Desert Canyon', sky: ['#f59e0b', '#fb923c', '#fde68a'], fog: '#f6c98a', fogDist: 7000,
    floor: ['#d9a35f', '#c0864a'], grid: { color: '#a8703a', alpha: 0.28, tile: 260, along: false, across: true },
    reflect: 0, dark: false, rim: '#ffffff', groundInk: '#7c4a1e', dust: '#e8c48f',
    particles: { kind: 'dust', color: '#fcd9a8', count: 80 },
    far: (ctx, P, t, A) => {
      disc(ctx, P, P.cam.camX * 0.1 + 1500, 2800, SKY_FAR, 2400, '#fff4d6', 2.4);
      // Mesas: flat-topped ridges
      for (const [Z, seed, col] of [[18000, 5, '#b45309'], [9000, 8, '#9a3412']] as [number, number, string][]) {
        forSlots(P, Z, 3000, seed, (X, k, r) => { if (r < 0.6) { const w = 1600 + r * 2200, h = 900 + r * 1500; const b = box(ctx, P, A, X, Z, w, h, col); if (b) { ctx.fillStyle = mix('#7c2d12', A.fog, fogK(A, b.z) * 0.8); ctx.fillRect(b.x, b.top + b.h * 0.25, b.w, b.h * 0.06); } } });
      }
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) forSlots(P, 1100, 650, 3, (X, k, r) => {
        if (r < 0.45) { const c = '#3f6212'; box(ctx, P, A, X, 1100, 46, 320, c); box(ctx, P, A, X - 50, 1100, 30, 120, c, 140); box(ctx, P, A, X - 66, 1100, 20, 110, c, 230); box(ctx, P, A, X + 52, 1100, 30, 90, c, 180); box(ctx, P, A, X + 66, 1100, 20, 90, c, 250); }
        else if (r < 0.75) { const p = P.pt(X, 0, 1100); if (p) { ctx.fillStyle = mix('#92400e', A.fog, fogK(A, p.z)); ctx.beginPath(); ctx.ellipse(p.x, p.y, 150 * p.s, 90 * p.s, 0, Math.PI, 0); ctx.fill(); } }
      });
      else forSlots(P, -560, 1500, 17, (X, k, r) => { if (r < 0.3) { const p = P.pt(X, 0, -560); if (p) { ctx.fillStyle = '#5c3313'; ctx.beginPath(); ctx.ellipse(p.x, p.y, 260 * p.s, 170 * p.s, 0, Math.PI, 0); ctx.fill(); } } }, 260);
    },
  },
  volcano: {
    id: 'volcano', name: 'Volcano', sky: ['#0c0402', '#2b0a04', '#7c2d12'], fog: '#3b1206', fogDist: 6500,
    floor: ['#24140f', '#170c08'], grid: { color: '#f97316', alpha: 0.0, tile: 200, along: false, across: false },
    reflect: 0, dark: true, rim: '#fed7aa', groundInk: '#0f0705', dust: '#57534e',
    particles: { kind: 'embers', color: '#fb923c', count: 90 },
    far: (ctx, P, t, A) => {
      const Z = 15000;
      const top = P.pt(P.cam.camX * 0.3, 6200, Z), l = P.pt(P.cam.camX * 0.3 - 9000, 0, Z), r = P.pt(P.cam.camX * 0.3 + 9000, 0, Z);
      if (top && l && r) {
        ctx.fillStyle = '#1c0904'; ctx.beginPath(); ctx.moveTo(l.x, l.y + 4); ctx.lineTo(top.x - 900 * top.s, top.y); ctx.lineTo(top.x + 900 * top.s, top.y); ctx.lineTo(r.x, r.y + 4); ctx.fill();
        glowDot(ctx, top.x, top.y, 2600 * top.s, '#f97316', 0.9 + 0.1 * Math.sin(t * 2));
        ctx.strokeStyle = rgba('#f97316', 0.85); ctx.lineWidth = Math.max(2, 120 * top.s);
        for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(top.x + (i - 1) * 400 * top.s, top.y); ctx.quadraticCurveTo(top.x + (i - 1) * 1600 * top.s, top.y + 2500 * top.s, top.x + (i - 1) * 3000 * top.s + (i === 1 ? 600 * top.s : 0), l.y); ctx.stroke(); }
        // Smoke plume
        for (let i = 0; i < 8; i++) { const k = ((t * 0.05 + i / 8) % 1); const p = P.pt(P.cam.camX * 0.3 + Math.sin(i * 2 + t * 0.2) * 1200 * k, 6400 + k * 9000, Z); if (p) { ctx.fillStyle = `rgba(30,20,20,${0.5 * (1 - k)})`; ctx.beginPath(); ctx.arc(p.x, p.y, (900 + 2500 * k) * p.s, 0, Math.PI * 2); ctx.fill(); } }
      }
      ridge(ctx, P, A, 6000, 600, 1100, 300, 6, '#140703', true);
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) {
        // Lava cracks in the floor (glowing), and jagged rocks behind.
        ctx.save(); ctx.lineCap = 'round';
        forSlots(P, 300, 260, 21, (X, k, r) => {
          let x = X, z = 60 + r * 900; const pts: Pt[] = [];
          for (let i = 0; i < 5; i++) { const p = P.pt(x, 0, z); if (p) pts.push(p); x += (hashN(k + i) - 0.5) * 260; z += 70 + hashN(k * 2 + i) * 160; }
          if (pts.length < 2) return;
          const path = () => { ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y); ctx.stroke(); };
          ctx.strokeStyle = rgba('#f97316', 0.22); ctx.lineWidth = Math.max(4, 30 * pts[0].s); path();
          ctx.strokeStyle = rgba('#fdba74', 0.8 + 0.2 * Math.sin(t * 3 + k)); ctx.lineWidth = Math.max(1.5, 8 * pts[0].s); path();
        });
        ctx.restore();
        forSlots(P, 1500, 700, 23, (X, k, r) => { if (r < 0.6) { const p = P.pt(X, 0, 1500), q = P.pt(X + (r - 0.3) * 200, 300 + r * 600, 1500); if (p && q) { ctx.fillStyle = mix('#1f0d07', A.fog, fogK(A, p.z) * 0.6); ctx.beginPath(); ctx.moveTo(p.x - 160 * p.s, p.y); ctx.lineTo(q.x, q.y); ctx.lineTo(p.x + 180 * p.s, p.y); ctx.fill(); } } });
      } else forSlots(P, -580, 1400, 29, (X, k, r) => { if (r < 0.3) { const p = P.pt(X, 0, -580), q = P.pt(X, 900, -580); if (p && q) { ctx.fillStyle = '#080302'; ctx.beginPath(); ctx.moveTo(p.x - 260 * p.s, p.y); ctx.lineTo(q.x, q.y); ctx.lineTo(p.x + 200 * p.s, p.y); ctx.fill(); } } }, 260);
    },
  },
  snow: {
    id: 'snow', name: 'Snow Peak', sky: ['#7dd3fc', '#bae6fd', '#eef7ff'], fog: '#e8f2fb', fogDist: 4800,
    floor: ['#f8fafc', '#dbe7f2'], grid: { color: '#9fb6cc', alpha: 0.25, tile: 240, along: false, across: true },
    reflect: 0.06, dark: false, rim: '#ffffff', groundInk: '#64748b', dust: '#ffffff',
    particles: { kind: 'snow', color: '#ffffff', count: 120 },
    far: (ctx, P, t, A) => {
      ridge(ctx, P, A, 24000, 2200, 8000, 2000, 2, '#94a3b8', true);
      ridge(ctx, P, A, 12000, 1300, 3400, 900, 7, '#cbd5e1', true);
      ridge(ctx, P, A, 5000, 700, 700, 300, 11, '#e2e8f0');
    },
    mid: (ctx, P, t, A, front) => {
      const pine = (X: number, Z: number, h: number, dark: string) => {
        const b = P.pt(X, 0, Z), top = P.pt(X, h, Z); if (!b || !top) return;
        const c = mix(dark, A.fog, fogK(A, b.z));
        for (let i = 0; i < 4; i++) { const y0 = lerp(b.y, top.y, i / 4 * 0.85), y1 = lerp(b.y, top.y, (i + 1.6) / 4 * 0.85 + 0.05); const w = (1 - i / 4.4) * h * 0.32 * b.s; ctx.fillStyle = c; ctx.beginPath(); ctx.moveTo(b.x - w, y0); ctx.lineTo(b.x, Math.min(y1, top.y + 2)); ctx.lineTo(b.x + w, y0); ctx.fill(); ctx.fillStyle = mix('#ffffff', A.fog, 0.2); ctx.beginPath(); ctx.moveTo(b.x - w * 0.35, lerp(y0, y1, 0.55)); ctx.lineTo(b.x, Math.min(y1, top.y + 2)); ctx.lineTo(b.x + w * 0.35, lerp(y0, y1, 0.55)); ctx.fill(); }
      };
      if (!front) { forSlots(P, 2200, 520, 12, (X, k, r) => pine(X, 2200, 500 + r * 500, '#1e3a2f')); forSlots(P, 1000, 760, 13, (X, k, r) => { if (r < 0.55) pine(X, 1000, 420 + r * 300, '#14532d'); }); }
      else forSlots(P, -520, 1100, 31, (X, k, r) => { if (r < 0.35) pine(X, -520, 1100, '#0b2a1a'); }, 360);
    },
  },
  alley: {
    id: 'alley', name: 'Rain Alley', sky: ['#020617', '#0f172a', '#1e293b'], fog: '#1e2a3d', fogDist: 3600,
    floor: ['#1a1f2b', '#0e121b'], grid: { color: '#334155', alpha: 0.4, tile: 180, along: true, across: false },
    reflect: 0.42, dark: true, rim: '#fbcfe8', groundInk: '#020617', dust: '#64748b',
    particles: { kind: 'rain', color: '#cbd5e1', count: 160 },
    far: (ctx, P, t, A) => {
      // Brick wall with lit windows and fire escapes.
      const Z = 820, wall = box(ctx, P, A, P.cam.camX, Z, P.span(Z) * 2.4, 2000, '#3a2228');
      if (!wall) return;
      const row = P.pt(P.cam.camX, 40, Z);
      if (row) { ctx.strokeStyle = 'rgba(0,0,0,0.18)'; ctx.lineWidth = 1; for (let y = 0; y < 2000; y += 40) { const q = P.pt(P.cam.camX, y, Z); if (q) { ctx.beginPath(); ctx.moveTo(wall.x, q.y); ctx.lineTo(wall.x + wall.w, q.y); ctx.stroke(); } } }
      forSlots(P, Z, 380, 17, (X, k, r) => {
        for (const y of [520, 980, 1440]) { const w = box(ctx, P, A, X, Z - 1, 150, 210, hashN(k + y) < 0.45 ? '#fde68a' : '#111827', y); if (w && hashN(k * y) < 0.3) { ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(w.x, w.top + w.h * 0.5, w.w, 2); } }
        if (r < 0.4) { for (const y of [470, 930]) box(ctx, P, A, X, Z - 30, 300, 10, '#0b0f17', y); box(ctx, P, A, X - 150, Z - 30, 8, 920, '#0b0f17', 0); }
      });
      forSlots(P, Z - 40, 1600, 19, (X, k, r) => { if (r < 0.6) { const s = box(ctx, P, A, X, Z - 40, 380, 120, '#160b18', 700); if (s) { ctx.save(); glowDot(ctx, s.x + s.w / 2, s.top + s.h * 0.5, s.w * 0.75, k & 1 ? '#f472b6' : '#38bdf8', 0.55); ctx.fillStyle = k & 1 ? '#f472b6' : '#38bdf8'; ctx.globalAlpha = 0.6 + 0.4 * (Math.sin(t * 13 + k) > -0.8 ? 1 : 0.2); ctx.font = `900 ${Math.max(6, 74 * s.s)}px sans-serif`; ctx.textAlign = 'center'; ctx.fillText(['RAMEN', 'BAR', '24H', 'CLUB'][k & 3], s.x + s.w / 2, s.top + s.h * 0.72); ctx.restore(); } } });
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) forSlots(P, 500, 1100, 37, (X, k, r) => { if (r < 0.5) { box(ctx, P, A, X, 500, 14, 640, '#111827'); const p = P.pt(X + 50, 630, 500); if (p) { glowDot(ctx, p.x, p.y + 40 * p.s, 300 * p.s, '#fde68a', 0.5); ctx.fillStyle = '#fef3c7'; ctx.fillRect(p.x - 30 * p.s, p.y, 60 * p.s, 14 * p.s); } } else box(ctx, P, A, X, 500, 160, 110, '#1f2937'); });
      else forSlots(P, -520, 1500, 41, (X, k, r) => { if (r < 0.3) box(ctx, P, A, X, -520, 120, 1700, '#05070d'); }, 60);
    },
  },
  colosseum: {
    id: 'colosseum', name: 'Colosseum', sky: ['#60a5fa', '#a5c8f5', '#e0ecfa'], fog: '#dbe6f2', fogDist: 6000,
    floor: ['#e3c093', '#cfa874'], grid: { color: '#b58a58', alpha: 0.22, tile: 300, along: false, across: true },
    reflect: 0, dark: false, rim: '#ffffff', groundInk: '#6b4a2b', dust: '#ead2ae',
    particles: { kind: 'dust', color: '#f5e1c0', count: 40 },
    far: (ctx, P, t, A) => {
      // Tiered stands with a cheering crowd, arches above.
      const Z = 2600, half = P.span(Z) * 1.2, c = P.cam.camX;
      for (let tier = 0; tier < 4; tier++) {
        const y0 = 300 + tier * 260;
        box(ctx, P, A, c, Z + tier * 260, half * 2, 260, tier % 2 ? '#c9b48f' : '#b8a27b', y0 - 300 + tier * 0);
        const q0 = P.pt(c, y0 - 120, Z + tier * 260);
        if (!q0) continue;
        for (let X = Math.floor((c - half) / 60) * 60; X < c + half; X += 60) {
          const r = hashN(X * 0.37 + tier);
          const p = P.pt(X, y0 - 160 + Math.abs(Math.sin(t * 6 + X)) * 22 * (r < 0.3 ? 1 : 0), Z + tier * 260);
          if (!p) continue;
          ctx.fillStyle = mix(['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#f8fafc', '#a855f7'][Math.floor(r * 6)], A.fog, 0.35);
          ctx.beginPath(); ctx.arc(p.x, p.y, 16 * p.s, 0, Math.PI * 2); ctx.fill();
        }
      }
      forSlots(P, 3800, 600, 43, (X) => { const a = box(ctx, P, A, X, 3800, 520, 900, '#d8c7a3', 1200); if (a) { ctx.fillStyle = mix('#7a6a50', A.fog, 0.5); ctx.beginPath(); ctx.ellipse(a.x + a.w / 2, a.top + a.h * 0.55, a.w * 0.32, a.h * 0.42, 0, 0, Math.PI * 2); ctx.fill(); } });
    },
    mid: (ctx, P, t, A, front) => {
      if (!front) {
        const wall = P.span(1300);
        box(ctx, P, A, P.cam.camX, 1300, wall * 2.4, 180, '#a08860');
        forSlots(P, 1290, 900, 47, (X, k) => { const b = box(ctx, P, A, X, 1290, 120, 520, k & 1 ? '#b91c1c' : '#1d4ed8', 120); if (b) { ctx.fillStyle = '#facc15'; ctx.fillRect(b.x + b.w * 0.3, b.top + b.h * 0.3, b.w * 0.4, b.h * 0.08); } });
      } else forSlots(P, -560, 1600, 53, (X, k, r) => { if (r < 0.3) box(ctx, P, A, X, -560, 140, 1500, '#6f5a3a'); }, 70);
    },
  },
};
export const ARENA_IDS = Object.keys(ARENAS);

/** A different arena for every round, seeded by the video (no repeats until all have been used). */
export function arenaOrder(seed: string, n: number): string[] {
  let h = 2166136261; for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  const ids = ARENA_IDS.slice();
  for (let i = ids.length - 1; i > 0; i--) { h = Math.imul(h ^ (h >>> 13), 0x5bd1e995) >>> 0; const j = h % (i + 1); [ids[i], ids[j]] = [ids[j], ids[i]]; }
  return Array.from({ length: n }, (_, i) => ids[i % ids.length]);
}

// ---------------------------------------------------------------------------
// Drawing an arena
// ---------------------------------------------------------------------------
/** Sky, far layers, the floor and the props behind the fight. */
export function drawArenaBack(ctx: Ctx, A: Arena, P: Proj, t: number) {
  const { W, H, horizon } = P.cam;
  const pad = Math.max(W, H) * 0.35;
  // Sky
  const sky = ctx.createLinearGradient(0, horizon - H * 1.2, 0, horizon);
  A.sky.forEach((c, i) => sky.addColorStop(i / Math.max(1, A.sky.length - 1), c));
  ctx.fillStyle = sky; ctx.fillRect(-pad, -pad, W + pad * 2, horizon + pad + 2);
  A.far(ctx, P, t, A);
  // Floor: gradient from the far (foggy) edge to the near floor colour.
  const fl = ctx.createLinearGradient(0, horizon, 0, H + pad);
  fl.addColorStop(0, mix(A.floor[1], A.fog, 0.75)); fl.addColorStop(0.12, A.floor[1]); fl.addColorStop(1, A.floor[0]);
  ctx.fillStyle = fl; ctx.fillRect(-pad, horizon, W + pad * 2, H - horizon + pad);
  drawFloorGrid(ctx, A, P);
  // Haze band on the horizon (depth).
  const hz = ctx.createLinearGradient(0, horizon - H * 0.08, 0, horizon + H * 0.05);
  hz.addColorStop(0, rgba(A.fog, 0)); hz.addColorStop(0.6, rgba(A.fog, 0.55)); hz.addColorStop(1, rgba(A.fog, 0));
  ctx.fillStyle = hz; ctx.fillRect(-pad, horizon - H * 0.08, W + pad * 2, H * 0.13);
  A.mid(ctx, P, t, A, false);
}

function drawFloorGrid(ctx: Ctx, A: Arena, P: Proj) {
  const g = A.grid;
  if (!g.alpha) return;
  const c = P.cam, sy = Math.sin(c.yaw), cy = Math.cos(c.yaw);
  ctx.save();
  ctx.strokeStyle = g.color; ctx.lineCap = 'round';
  // Glowing grids: a wide faint stroke under the line (cheap; no canvas shadow blur).
  const line = (x0: number, y0: number, x1: number, y1: number, w: number) => {
    if (g.glow) { const a = ctx.globalAlpha; ctx.globalAlpha = a * 0.22; ctx.lineWidth = w * 4.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.globalAlpha = a; }
    ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  };
  const zNear = -c.D * 0.85, zFar = 9000;
  const clipZ = (X: number) => (60 - c.D - (X - c.camX) * sy) / Math.max(0.2, cy);
  if (g.along !== false) {
    const half = 3200;
    for (let X = Math.floor((c.camX - half) / g.tile) * g.tile; X <= c.camX + half; X += g.tile) {
      const z0 = Math.max(zNear, clipZ(X)), p0 = P.pt(X, 0, z0), p1 = P.pt(X, 0, zFar);
      if (!p0 || !p1) continue;
      ctx.globalAlpha = g.alpha * clamp(1 - Math.abs(X - c.camX) / half, 0.15, 1);
      line(p0.x, p0.y, p1.x, p1.y, Math.max(1, Math.min(4, p0.s * 2.2)));
    }
  }
  if (g.across !== false) {
    for (let Z = Math.ceil(zNear / g.tile) * g.tile; Z < zFar; Z += g.tile * (Z > 2000 ? 2 : 1)) {
      const a = P.pt(c.camX - 6000, 0, Z), b = P.pt(c.camX + 6000, 0, Z);
      if (!a || !b) continue;
      ctx.globalAlpha = g.alpha * clamp(1 - (a.z - c.D * 0.2) / 7000, 0.1, 1);
      line(a.x, a.y, b.x, b.y, Math.max(1, Math.min(4, a.s * 2.2)));
    }
  }
  ctx.restore();
}

/** Props between the camera and the fighters, weather and atmosphere. */
export function drawArenaFront(ctx: Ctx, A: Arena, P: Proj, t: number) {
  A.mid(ctx, P, t, A, true);
  const p = A.particles;
  if (p) {
    const c = P.cam, box = 3200;
    ctx.save();
    for (let i = 0; i < p.count; i++) {
      const r1 = hashN(i * 1.7), r2 = hashN(i * 3.1 + 2), r3 = hashN(i * 5.3 + 4);
      const Z = -400 + r3 * 2600;
      let X = r1 * box, Y = r2 * 1400;
      if (p.kind === 'snow') { X += t * 40 + Math.sin(t * 1.3 + i) * 60; Y = 1400 - ((r2 * 1400 + t * (90 + r3 * 60)) % 1400); }
      if (p.kind === 'rain') { X += t * 120; Y = 1600 - ((r2 * 1600 + t * 1700) % 1600); }
      if (p.kind === 'embers' || p.kind === 'sparks') { X += Math.sin(t * 0.8 + i) * 80; Y = (r2 * 1300 + t * (60 + r3 * 140)) % 1300; }
      if (p.kind === 'petals' || p.kind === 'leaves') { X += t * 90 + Math.sin(t * 1.7 + i) * 90; Y = 1300 - ((r2 * 1300 + t * 70) % 1300); }
      if (p.kind === 'dust') { X += t * 25 + Math.sin(t * 0.5 + i) * 40; Y = 40 + r2 * 700 + Math.sin(t * 0.7 + i) * 30; }
      X = c.camX - box / 2 + (((X - c.camX + box / 2) % box) + box) % box;
      const q = P.pt(X, Y, Z); if (!q) continue;
      const sz = Math.max(0.8, (p.kind === 'snow' ? 7 : p.kind === 'rain' ? 2.2 : p.kind === 'petals' ? 9 : p.kind === 'embers' ? 4.5 : 3) * q.s);
      ctx.globalAlpha = clamp(1.1 - q.z / 3600, 0.15, 0.95) * (p.kind === 'dust' ? 0.45 : 0.9);
      ctx.fillStyle = p.color; ctx.strokeStyle = p.color;
      if (p.kind === 'rain') { ctx.lineWidth = sz; ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(q.x - 0.07 * 220 * q.s, q.y + 220 * q.s * 0.5); ctx.stroke(); }
      else if (p.kind === 'petals') { ctx.save(); ctx.translate(q.x, q.y); ctx.rotate(t * 2 + i); ctx.beginPath(); ctx.ellipse(0, 0, sz, sz * 0.55, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore(); }
      else if (p.kind === 'embers' || p.kind === 'sparks') { const a0 = ctx.globalAlpha; ctx.globalAlpha = a0 * 0.25; ctx.beginPath(); ctx.arc(q.x, q.y, sz * 2.6, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = a0; ctx.beginPath(); ctx.arc(q.x, q.y, sz, 0, Math.PI * 2); ctx.fill(); }
      else { ctx.beginPath(); ctx.arc(q.x, q.y, sz, 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.restore();
  }
}

/** Screen-space finishing: vignette and a grade for dark sets. */
export function drawArenaGrade(ctx: Ctx, A: Arena, W: number, H: number) {
  const v = ctx.createRadialGradient(W / 2, H * 0.55, Math.min(W, H) * 0.35, W / 2, H * 0.55, Math.max(W, H) * 0.8);
  v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, A.dark ? 'rgba(0,0,0,0.55)' : 'rgba(0,0,0,0.22)');
  ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
}

// ---------------------------------------------------------------------------
// Shot director
// ---------------------------------------------------------------------------
export type ShotKind = 'wide' | 'medium' | 'orbit' | 'low' | 'high' | 'dutch' | 'push' | 'close' | 'bullet' | 'over';
export interface Shot {
  kind: ShotKind; t0: number; t1: number;
  zoom0: number; zoom1: number;       // >1 = tighter
  horizon: number;                    // horizon height above the ground line (fraction of H)
  ground: number;                     // ground line (fraction of H)
  yaw0: number; yaw1: number;         // radians (orbit)
  roll0: number; roll1: number;       // radians (dutch)
  hand: number;                       // handheld shake amount
  bias: number;                       // framing offset (fraction of the fighters' span)
  whip: boolean;                      // whip-pan into this shot
}

export interface ShotInput { start: number; end: number; focus: { t: number }[]; impacts: { t: number; strength: number; ko?: boolean; kind: string }[]; slowmo: { t0: number; t1: number }[]; portrait: boolean; seed: string }

export function planShots(inp: ShotInput): Shot[] {
  let h = 2166136261; for (let i = 0; i < inp.seed.length; i++) { h ^= inp.seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  const rnd = () => { h = Math.imul(h ^ (h >>> 13), 0x5bd1e995) >>> 0; h ^= h >>> 15; return (h % 100000) / 100000; };
  const pick = <T>(a: [T, number][]): T => { const tot = a.reduce((n, x) => n + x[1], 0); let r = rnd() * tot; for (const [k, w] of a) { r -= w; if (r <= 0) return k; } return a[0][0]; };
  const P = inp.portrait;
  const G = P ? 0.66 : 0.76;
  const make = (kind: ShotKind, t0: number, t1: number): Shot => {
    const side = rnd() < 0.5 ? -1 : 1;
    const base: Shot = { kind, t0, t1, zoom0: 1, zoom1: 1, horizon: 0.3, ground: G, yaw0: 0, yaw1: 0, roll0: 0, roll1: 0, hand: 0.35, bias: 0, whip: false };
    switch (kind) {
      case 'wide': return { ...base, zoom0: 0.86, zoom1: 0.92, horizon: 0.34, yaw0: side * 0.08, yaw1: side * 0.02, hand: 0.2 };
      case 'medium': return { ...base, zoom0: 1.0, zoom1: 1.06, horizon: 0.3, yaw0: side * 0.12, yaw1: side * 0.16, hand: 0.4 };
      case 'orbit': { const a = 0.26 + rnd() * 0.16; return { ...base, zoom0: 1.0, zoom1: 1.08, horizon: 0.26, yaw0: -side * a, yaw1: side * a, hand: 0.25 }; }
      case 'low': return { ...base, zoom0: 1.15, zoom1: 1.22, horizon: 0.06, ground: P ? 0.78 : 0.86, yaw0: side * 0.2, yaw1: side * 0.26, roll0: side * 0.02, roll1: side * 0.04, hand: 0.3 };
      case 'high': return { ...base, zoom0: 0.86, zoom1: 0.9, horizon: P ? 0.5 : 0.56, ground: P ? 0.7 : 0.8, yaw0: side * 0.18, yaw1: side * 0.1, hand: 0.25 };
      case 'dutch': return { ...base, zoom0: 1.08, zoom1: 1.14, horizon: 0.24, yaw0: side * 0.22, yaw1: side * 0.18, roll0: side * 0.13, roll1: side * 0.16, hand: 0.5 };
      case 'push': return { ...base, zoom0: 0.92, zoom1: 1.35, horizon: 0.22, yaw0: side * 0.1, yaw1: side * 0.06, hand: 0.15 };
      case 'over': return { ...base, zoom0: 1.12, zoom1: 1.16, horizon: 0.2, yaw0: side * 0.34, yaw1: side * 0.38, bias: side * 0.28, hand: 0.35 };
      case 'close': return { ...base, zoom0: 1.35, zoom1: 1.8, horizon: 0.18, yaw0: side * 0.18, yaw1: side * 0.24, roll0: side * 0.03, roll1: side * 0.06, hand: 0.9 };
      case 'bullet': { const a = 0.42; return { ...base, zoom0: 1.2, zoom1: 1.4, horizon: 0.1, ground: P ? 0.74 : 0.82, yaw0: -side * a, yaw1: side * a, hand: 0 }; }
    }
  };
  const shots: Shot[] = [];
  // Cut points: the start of exchanges (focus changes), thinned so shots last 1–3 s.
  const cuts: number[] = [inp.start];
  for (const f of inp.focus) if (f.t > cuts[cuts.length - 1] + 0.9 + rnd() * 1.2 && f.t < inp.end - 0.6) cuts.push(f.t);
  for (let t = cuts[cuts.length - 1] + 2.6; t < inp.end - 0.8; t += 2 + rnd() * 1.4) cuts.push(t);
  cuts.sort((a, b) => a - b); cuts.push(inp.end);
  let prev: ShotKind | '' = '';
  const weights: [ShotKind, number][] = [['medium', 3], ['orbit', 3], ['low', 2], ['wide', 2], ['dutch', 1.5], ['high', 1.2], ['push', 1.2], ['over', 1.5]];
  for (let i = 0; i < cuts.length - 1; i++) {
    let k: ShotKind = i === 0 ? (rnd() < 0.5 ? 'wide' : 'push') : pick(weights);
    for (let n = 0; n < 5 && k === prev; n++) k = pick(weights);
    const s = make(k, cuts[i], cuts[i + 1]);
    s.whip = i > 0 && rnd() < 0.28;
    shots.push(s); prev = k;
  }
  // Big blows: a crash zoom, timed to start just before the hit.
  const inSlow = (t: number) => inp.slowmo.some((w) => t >= w.t0 - 0.3 && t <= w.t1 + 0.3);
  for (const im of inp.impacts) {
    const big = im.ko || im.strength >= 1.45 || im.kind === 'clash' || im.kind === 'spiked';
    if (!big || inSlow(im.t) || rnd() > (im.ko ? 1 : 0.55)) continue;
    insertShot(shots, make('close', im.t - 0.14, im.t + 0.5));
  }
  // Slow-motion finishers: a bullet-time orbit around the knockout.
  for (const w of inp.slowmo) insertShot(shots, make('bullet', w.t0 - 0.2, w.t1 + 0.35));
  return shots;
}

function insertShot(shots: Shot[], s: Shot) {
  const out: Shot[] = [];
  for (const x of shots) {
    if (x.t1 <= s.t0 || x.t0 >= s.t1) { out.push(x); continue; }
    if (x.t0 < s.t0) out.push({ ...x, t1: s.t0 });
    if (x.t1 > s.t1) out.push({ ...x, t0: s.t1, whip: false });
  }
  out.push(s);
  out.sort((a, b) => a.t0 - b.t0);
  shots.length = 0; shots.push(...out);
}

export interface CamState { zoom: number; horizon: number; ground: number; yaw: number; roll: number; hand: number; bias: number; whip: number; kind: ShotKind }

/** The camera at time t: the shot's move eased over its length; hard cuts between shots. */
export function shotAt(shots: Shot[], t: number): CamState {
  let s = shots[0];
  for (const x of shots) if (t >= x.t0) s = x;
  if (!s) return { zoom: 1, horizon: 0.3, ground: 0.72, yaw: 0, roll: 0, hand: 0.3, bias: 0, whip: 0, kind: 'medium' };
  const k = smooth((t - s.t0) / Math.max(0.05, s.t1 - s.t0));
  const kz = s.kind === 'close' ? smooth((t - s.t0) / 0.16) : k;
  const whip = s.whip ? Math.max(0, 1 - (t - s.t0) / 0.2) : 0;
  return { zoom: lerp(s.zoom0, s.zoom1, kz), horizon: s.horizon, ground: s.ground, yaw: lerp(s.yaw0, s.yaw1, k), roll: lerp(s.roll0, s.roll1, k), hand: s.hand, bias: s.bias, whip, kind: s.kind };
}

/** Smooth handheld noise (x, y in pixels per unit amount; roll in radians). */
export function handheld(t: number, amount: number) {
  const n = (a: number, b: number, c: number) => Math.sin(t * a) * 0.5 + Math.sin(t * b + 1.3) * 0.3 + Math.sin(t * c + 2.1) * 0.2;
  return { x: n(1.7, 3.1, 5.3) * amount, y: n(2.3, 3.7, 6.1) * amount * 0.6, r: n(1.1, 2.9, 4.7) * amount * 0.006 };
}
