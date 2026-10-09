/**
 * Original background music, composed and synthesised for every video.
 *
 * Copyright: nothing here is sampled or taken from a recording — the notes are
 * generated from simple music theory (scales, chord progressions, rhythms) and
 * rendered by a small synthesiser, so every soundtrack is new and owned by the
 * channel. No third-party music, no attribution, no Content ID claims.
 *
 * Moods: story (warm), emotional (piano), mystery, horror, news, tech, upbeat,
 * action (fights: a trailer-style score — taikos, string ostinato, braams, risers), comedy (bouncy, playful).
 * A seed (campaign + part) varies the key, tempo, progression and patterns, so
 * no two videos sound identical. Pure TypeScript with no imports, so it runs in
 * Node (the cloud renderer) and in the browser (the app) alike.
 */

export type MusicMood = 'story' | 'emotional' | 'mystery' | 'horror' | 'news' | 'tech' | 'upbeat' | 'action' | 'comedy';

const SR = 44100;
const TAU = Math.PI * 2;

// ---------------------------------------------------------------------------
// Seeded randomness
// ---------------------------------------------------------------------------
function rng(seedText: string) {
  let h = 2166136261;
  for (let i = 0; i < seedText.length; i++) { h ^= seedText.charCodeAt(i); h = Math.imul(h, 16777619); }
  let s = h >>> 0;
  const next = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { next, pick: <T>(a: T[]) => a[Math.floor(next() * a.length) % a.length], range: (a: number, b: number) => a + (b - a) * next() };
}

// ---------------------------------------------------------------------------
// Wavetables (band-limited, read with linear interpolation)
// ---------------------------------------------------------------------------
const TABLE = 4096;
function table(harmonics: [number, number][]): Float32Array {
  const t = new Float32Array(TABLE + 1);
  for (let i = 0; i <= TABLE; i++) {
    const x = i / TABLE;
    let v = 0;
    for (const [k, a] of harmonics) v += a * Math.sin(TAU * k * x);
    t[i] = v;
  }
  let peak = 0;
  for (let i = 0; i < t.length; i++) peak = Math.max(peak, Math.abs(t[i]));
  for (let i = 0; i < t.length; i++) t[i] /= peak || 1;
  return t;
}
const SINE = table([[1, 1]]);
const SAW = table(Array.from({ length: 14 }, (_, i) => [i + 1, 1 / (i + 1)] as [number, number]));
const WARM = table([[1, 1], [2, 0.42], [3, 0.2], [4, 0.1], [5, 0.05]]);          // soft piano-like
const PLUCK = table([[1, 1], [2, 0.6], [3, 0.35], [4, 0.22], [6, 0.1], [8, 0.05]]);
const BELL = table([[1, 1], [2.76, 0.45], [5.4, 0.2], [8.93, 0.08]].map(([k, a]) => [Math.round(k), a] as [number, number]));
const BASS = table([[1, 1], [2, 0.3], [3, 0.12]]);

const midiHz = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

interface Bus { L: Float32Array; R: Float32Array; send: Float32Array }

/** One note from a wavetable with an envelope, a one-pole low-pass and stereo pan. */
function note(bus: Bus, o: {
  table: Float32Array; hz: number; start: number; dur: number; gain: number; pan?: number;
  attack?: number; decay?: number; sustain?: number; release?: number; pluck?: number; cutoff?: number; detune?: number; reverb?: number; vibrato?: number;
}) {
  const a = o.attack ?? 0.01, r = o.release ?? 0.3, d = o.decay ?? 0.2, sus = o.sustain ?? 0.8;
  const n0 = Math.max(0, Math.floor(o.start * SR));
  const len = Math.floor((o.dur + r) * SR);
  const n1 = Math.min(bus.L.length, n0 + len);
  const pan = o.pan ?? 0, gl = Math.cos((pan + 1) * Math.PI / 4) * o.gain, gr = Math.sin((pan + 1) * Math.PI / 4) * o.gain;
  const alpha = o.cutoff ? 1 - Math.exp((-TAU * o.cutoff) / SR) : 1;
  const inc = (o.hz * (1 + (o.detune || 0))) / SR;
  const vib = o.vibrato || 0;
  const send = o.reverb ?? 0.25;
  let ph = Math.random() * 0.0, y = 0;
  for (let n = n0; n < n1; n++) {
    const t = (n - n0) / SR;
    let env: number;
    if (o.pluck) env = (t < a ? t / a : Math.exp(-(t - a) / o.pluck)) * (t > o.dur ? Math.max(0, 1 - (t - o.dur) / r) : 1);
    else if (t < a) env = t / a;
    else if (t < a + d) env = 1 - (1 - sus) * ((t - a) / d);
    else if (t < o.dur) env = sus;
    else env = sus * Math.max(0, 1 - (t - o.dur) / r);
    const p = ph * TABLE, i = p | 0, f = p - i;
    const x = (o.table[i] + (o.table[i + 1] - o.table[i]) * f) * env;
    ph += inc * (vib ? 1 + vib * Math.sin(TAU * 5 * t) : 1);
    if (ph >= 1) ph -= 1;
    y += alpha * (x - y);
    bus.L[n] += y * gl; bus.R[n] += y * gr; bus.send[n] += y * o.gain * send;
  }
}

function kick(bus: Bus, start: number, gain: number) {
  const n0 = Math.floor(start * SR), len = Math.floor(0.32 * SR);
  let ph = 0;
  for (let k = 0; k < len && n0 + k < bus.L.length; k++) {
    const t = k / SR;
    const hz = 45 + 85 * Math.exp(-t / 0.035);
    ph += hz / SR;
    const v = Math.sin(TAU * ph) * Math.exp(-t / 0.11) * gain;
    bus.L[n0 + k] += v; bus.R[n0 + k] += v;
  }
}
function noiseHit(bus: Bus, start: number, gain: number, decay: number, bright: number, pan = 0, r: () => number) {
  const n0 = Math.floor(start * SR), len = Math.floor(decay * 6 * SR);
  let prev = 0, lp = 0;
  const gl = Math.cos((pan + 1) * Math.PI / 4) * gain, gr = Math.sin((pan + 1) * Math.PI / 4) * gain;
  for (let k = 0; k < len && n0 + k < bus.L.length; k++) {
    const t = k / SR;
    const w = r() * 2 - 1;
    const hp = w - prev; prev = w;                   // brighten
    lp += 0.35 * (hp - lp);
    const v = (bright * hp + (1 - bright) * lp) * Math.exp(-t / decay);
    bus.L[n0 + k] += v * gl; bus.R[n0 + k] += v * gr; bus.send[n0 + k] += v * gain * 0.15;
  }
}

/** Small Schroeder reverb on the send bus. */
function reverb(bus: Bus, mix: number) {
  const combs = [1557, 1617, 1491, 1422, 1277, 1356];
  const out = new Float32Array(bus.send.length);
  for (const [ci, dl] of combs.entries()) {
    const buf = new Float32Array(dl); let idx = 0, filt = 0;
    const fb = 0.83 - ci * 0.004;
    for (let n = 0; n < out.length; n++) {
      const y = buf[idx];
      filt = y * 0.72 + filt * 0.28;
      buf[idx] = bus.send[n] + filt * fb;
      out[n] += y / combs.length;
      idx = (idx + 1) % dl;
    }
  }
  for (const dl of [225, 556, 441]) {
    const buf = new Float32Array(dl); let idx = 0;
    for (let n = 0; n < out.length; n++) {
      const b = buf[idx]; const x = out[n];
      const y = -x + b; buf[idx] = x + b * 0.5; out[n] = y; idx = (idx + 1) % dl;
    }
  }
  for (let n = 0; n < out.length; n++) {
    const lag = n >= 23 ? out[n - 23] : 0;             // tiny L/R offset = width
    bus.L[n] += out[n] * mix; bus.R[n] += lag * mix;
  }
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------
const MAJOR = [0, 2, 4, 5, 7, 9, 11];
const MINOR = [0, 2, 3, 5, 7, 8, 10];
/** Scale-degree triad (optionally a 7th) as MIDI notes. */
function chord(root: number, scale: number[], degree: number, seventh = false): number[] {
  const deg = (k: number) => root + scale[k % 7] + 12 * Math.floor(k / 7);
  const notes = [deg(degree), deg(degree + 2), deg(degree + 4)];
  if (seventh) notes.push(deg(degree + 6));
  return notes;
}

interface Plan { scale: number[]; root: number; bpm: number; progs: number[][]; barsPerChord: number }

function planFor(mood: MusicMood, r: ReturnType<typeof rng>): Plan {
  switch (mood) {
    case 'horror': return { scale: MINOR, root: r.pick([38, 39, 40, 41]), bpm: r.range(58, 66), progs: [[0, 5, 0, 1], [0, 0, 5, 6]], barsPerChord: 2 };
    case 'mystery': return { scale: MINOR, root: r.pick([45, 47, 48, 50]), bpm: r.range(70, 80), progs: [[0, 3, 5, 4], [0, 5, 3, 4], [0, 6, 5, 4]], barsPerChord: 1 };
    case 'emotional': return { scale: MAJOR, root: r.pick([48, 50, 53, 55]), bpm: r.range(66, 74), progs: [[0, 4, 5, 3], [5, 3, 0, 4], [0, 5, 3, 4]], barsPerChord: 1 };
    case 'news': return { scale: MINOR, root: r.pick([45, 47, 48, 50]), bpm: r.range(100, 112), progs: [[0, 5, 2, 6], [0, 5, 3, 4], [0, 2, 5, 4]], barsPerChord: 1 };
    case 'tech': return { scale: MAJOR, root: r.pick([48, 50, 52, 53]), bpm: r.range(98, 110), progs: [[5, 3, 0, 4], [0, 4, 5, 3], [3, 4, 5, 5]], barsPerChord: 1 };
    case 'upbeat': return { scale: MAJOR, root: r.pick([48, 50, 53, 55]), bpm: r.range(100, 112), progs: [[0, 5, 3, 4], [0, 3, 4, 3], [0, 4, 5, 3]], barsPerChord: 1 };
    case 'action': return { scale: MINOR, root: r.pick([40, 41, 43, 45]), bpm: r.range(138, 150), progs: [[0, 5, 6, 4], [0, 0, 5, 6], [0, 6, 5, 6]], barsPerChord: 1 };
    case 'comedy': return { scale: MAJOR, root: r.pick([53, 55, 57]), bpm: r.range(118, 128), progs: [[0, 3, 4, 0], [0, 5, 3, 4]], barsPerChord: 1 };
    default: return { scale: MAJOR, root: r.pick([48, 50, 53, 55]), bpm: r.range(72, 84), progs: [[0, 4, 5, 3], [0, 5, 3, 4], [3, 0, 4, 5]], barsPerChord: 1 };
  }
}

function compose(bus: Bus, mood: MusicMood, seconds: number, r: ReturnType<typeof rng>) {
  const P = planFor(mood, r);
  const beat = 60 / P.bpm, bar = beat * 4;
  const prog = r.pick(P.progs);
  const chordAt = (b: number) => chord(P.root, P.scale, prog[Math.floor(b / P.barsPerChord) % prog.length], mood === 'emotional' || mood === 'tech' || mood === 'mystery');
  const bars = Math.ceil(seconds / bar) + 1;
  const arpPattern = r.pick([[0, 1, 2, 1, 3, 2, 1, 2], [0, 2, 1, 2, 3, 2, 1, 0], [0, 1, 2, 3, 2, 1, 2, 1]]);
  const introBars = mood === 'horror' ? 1 : 2;

  for (let b = 0; b < bars; b++) {
    const t0 = b * bar;
    if (t0 > seconds) break;
    const ch = chordAt(b);
    const newChord = b % P.barsPerChord === 0;
    const full = b >= introBars;

    // Pad: sustained chord (detuned pair = width), always present.
    if (newChord) {
      const dur = bar * P.barsPerChord;
      for (const [k, m] of ch.entries()) {
        const hz = midiHz(m + 12);
        const g = (mood === 'horror' ? 0.05 : 0.045) / ch.length * 3;
        note(bus, { table: SAW, hz, start: t0, dur, gain: g, pan: -0.35, attack: 0.9, decay: 0.5, sustain: 0.85, release: 1.4, cutoff: mood === 'horror' ? 900 : 1100 + 90 * k, detune: -0.004, reverb: 0.5 });
        note(bus, { table: SAW, hz, start: t0, dur, gain: g, pan: 0.35, attack: 1.0, decay: 0.5, sustain: 0.85, release: 1.4, cutoff: mood === 'horror' ? 900 : 1100 + 90 * k, detune: 0.004, reverb: 0.5 });
      }
    }

    switch (mood) {
      case 'story':
      case 'emotional': {
        // Gentle piano arpeggio in 8ths; a melody note on beat 1 every other bar.
        if (full || mood === 'emotional') for (let i = 0; i < 8; i++) {
          const m = ch[arpPattern[i] % ch.length] + (arpPattern[i] >= ch.length ? 12 : 0) + 12;
          note(bus, { table: WARM, hz: midiHz(m), start: t0 + i * beat / 2 + r.range(-0.008, 0.008), dur: beat * 1.6, gain: 0.07 * (i % 2 ? 0.75 : 1), pan: r.range(-0.3, 0.3), attack: 0.004, pluck: 0.55, release: 0.4, cutoff: 3200, reverb: 0.45 });
        }
        if (full && b % 2 === 0) note(bus, { table: WARM, hz: midiHz(ch[r.pick([0, 1, 2])] + 24), start: t0, dur: beat * 3, gain: 0.05, pan: 0.1, attack: 0.004, pluck: 1.2, release: 0.8, cutoff: 3600, reverb: 0.55 });
        note(bus, { table: BASS, hz: midiHz(ch[0] - 12), start: t0, dur: bar * 0.95, gain: 0.05, attack: 0.02, pluck: 1.6, release: 0.4, cutoff: 600, reverb: 0.1 });
        break;
      }
      case 'mystery': {
        // Sparse plucks on chord tones, a low pulse on beat 1.
        for (let i = 0; i < 4; i++) if (r.next() < (full ? 0.75 : 0.4)) {
          note(bus, { table: PLUCK, hz: midiHz(r.pick(ch) + 12 + (r.next() < 0.3 ? 12 : 0)), start: t0 + i * beat + (r.next() < 0.3 ? beat / 2 : 0), dur: beat, gain: 0.055, pan: r.range(-0.6, 0.6), attack: 0.003, pluck: 0.35, release: 0.3, cutoff: 2600, reverb: 0.6 });
        }
        note(bus, { table: BASS, hz: midiHz(ch[0] - 12), start: t0, dur: bar, gain: 0.055, attack: 0.05, pluck: 1.4, release: 0.5, cutoff: 400, reverb: 0.2 });
        if (full && b % 4 === 3) note(bus, { table: BELL, hz: midiHz(ch[2] + 24), start: t0 + beat * 2, dur: beat * 2, gain: 0.03, pan: 0.5, attack: 0.002, pluck: 1.2, release: 1, cutoff: 5000, reverb: 0.8 });
        break;
      }
      case 'horror': {
        // Low drone, a dissonant high cluster now and then, a slow heartbeat.
        if (newChord) {
          note(bus, { table: SINE, hz: midiHz(P.root), start: t0, dur: bar * P.barsPerChord, gain: 0.045, attack: 1.5, sustain: 1, release: 2, reverb: 0.3, vibrato: 0.003 });
          note(bus, { table: SINE, hz: midiHz(P.root) * 1.012, start: t0, dur: bar * P.barsPerChord, gain: 0.03, attack: 2, sustain: 1, release: 2, reverb: 0.3 });
        }
        if (r.next() < 0.45) {
          const m = ch[0] + 24 + r.pick([1, 6, 11]);        // minor 2nd / tritone colour
          note(bus, { table: BELL, hz: midiHz(m), start: t0 + r.range(0, bar * 0.7), dur: beat * 2, gain: 0.03, pan: r.range(-0.8, 0.8), attack: 0.002, pluck: 1.6, release: 1.5, cutoff: 4200, reverb: 0.9 });
        }
        if (full && b % 2 === 1) { kick(bus, t0 + beat * 2, 0.22); kick(bus, t0 + beat * 2.42, 0.15); }
        if (b % 4 === 0) note(bus, { table: WARM, hz: midiHz(P.root), start: t0, dur: bar, gain: 0.1, attack: 0.003, pluck: 2.2, release: 1, cutoff: 1400, reverb: 0.7 });
        break;
      }
      case 'news': {
        // Driving 8th-note bass pulse, soft kick on every beat, hats on the off-beats.
        for (let i = 0; i < 8; i++) note(bus, { table: BASS, hz: midiHz(ch[0] - 12), start: t0 + i * beat / 2, dur: beat * 0.42, gain: 0.045, attack: 0.004, pluck: 0.18, release: 0.05, cutoff: 900, reverb: 0.05 });
        if (full) for (let i = 0; i < 4; i++) { kick(bus, t0 + i * beat, 0.11); noiseHit(bus, t0 + i * beat + beat / 2, 0.035, 0.03, 0.9, 0.25, r.next); }
        if (full) for (let i = 0; i < 2; i++) note(bus, { table: PLUCK, hz: midiHz(ch[(i + b) % 3] + 24), start: t0 + i * beat * 2 + beat * 1.5, dur: beat * 0.4, gain: 0.04, pan: i ? 0.4 : -0.4, attack: 0.003, pluck: 0.2, release: 0.1, cutoff: 3500, reverb: 0.35 });
        break;
      }
      case 'tech': {
        // Bright 16th-note pluck arpeggio, light four-on-the-floor.
        for (let i = 0; i < 16; i++) {
          const k = arpPattern[i % 8];
          note(bus, { table: PLUCK, hz: midiHz(ch[k % ch.length] + 24 + (k >= ch.length ? 12 : 0)), start: t0 + i * beat / 4, dur: beat / 4, gain: 0.03 * (i % 4 === 0 ? 1.2 : 0.85), pan: i % 2 ? 0.35 : -0.35, attack: 0.002, pluck: 0.12, release: 0.08, cutoff: 4200, reverb: 0.3 });
        }
        note(bus, { table: BASS, hz: midiHz(ch[0] - 12), start: t0, dur: bar * 0.9, gain: 0.05, attack: 0.01, pluck: 1.2, release: 0.2, cutoff: 700, reverb: 0.05 });
        if (full) for (let i = 0; i < 4; i++) { kick(bus, t0 + i * beat, 0.1); noiseHit(bus, t0 + i * beat + beat / 2, 0.028, 0.025, 0.95, -0.2, r.next); }
        break;
      }
      case 'action': {
        // Modern action / trailer score, building every 8 bars:
        //  taiko-style toms + a big low hit, a staccato low-string ostinato (3-3-2 accents),
        //  braams (low brass swells) on each phrase, snare rolls + a rising sweep into the next
        //  section, power-chord stabs and 16th hats once it is at full intensity.
        const phrase = Math.floor(b / 8), inPhrase = b % 8;
        const level = !full ? 0 : Math.min(3, 1 + phrase);                 // 1 → 3
        const root = ch[0];
        // Ostinato: 16ths of root / octave / fifth, accented 3-3-2.
        const ost = [0, 12, 7, 12, 0, 12, 7, 12, 0, 12, 7, 12, 0, 12, 7, 15];
        for (let i = 0; i < 16; i++) {
          const acc = [0, 3, 6, 8, 11, 14].includes(i);
          note(bus, { table: SAW, hz: midiHz(root - 12 + ost[i]), start: t0 + i * beat / 4, dur: beat * 0.16, gain: (acc ? 0.034 : 0.019) * (level ? 1 : 0.7), pan: i % 2 ? 0.25 : -0.25, attack: 0.003, decay: 0.05, sustain: 0.5, release: 0.04, cutoff: acc ? 1800 : 1200, reverb: 0.12 });
        }
        // Sub bass on the accents.
        for (const i of [0, 3, 6]) note(bus, { table: BASS, hz: midiHz(root - 24), start: t0 + i * beat / 2, dur: beat * 0.9, gain: 0.07, attack: 0.004, pluck: 0.5, release: 0.08, cutoff: 260, reverb: 0.02 });
        // Braam at the top of every 4 bars (low brass cluster swelling open).
        if (b % 4 === 0 && level >= 1) for (const [m, g] of [[root - 12, 0.05], [root - 5, 0.035], [root, 0.03]] as [number, number][]) {
          note(bus, { table: SAW, hz: midiHz(m), start: t0, dur: bar * 1.5, gain: g, pan: m === root ? 0.2 : -0.2, attack: 0.06, decay: 0.6, sustain: 0.7, release: 0.9, cutoff: 700 + level * 200, detune: 0.006, reverb: 0.4 });
          note(bus, { table: SAW, hz: midiHz(m), start: t0, dur: bar * 1.5, gain: g * 0.8, pan: m === root ? -0.2 : 0.2, attack: 0.07, decay: 0.6, sustain: 0.7, release: 0.9, cutoff: 650 + level * 200, detune: -0.006, reverb: 0.4 });
        }
        // Taikos: a heavy pattern that gets busier with the level.
        const taiko = level >= 3 ? [0, 0.75, 1.5, 2, 2.5, 3, 3.25, 3.5] : level === 2 ? [0, 0.75, 1.5, 2.5, 3] : level === 1 ? [0, 1.5, 2.5] : [0];
        for (const at of taiko) { kick(bus, t0 + at * beat, at === 0 ? 0.32 : 0.2); noiseHit(bus, t0 + at * beat, at === 0 ? 0.05 : 0.03, 0.05, 0.25, at % 1 ? 0.3 : -0.3, r.next); }
        if (level >= 2) for (const i of [1, 3]) noiseHit(bus, t0 + i * beat, 0.1, 0.08, 0.6, 0, r.next);       // snare backbeat
        if (level >= 3) for (let i = 0; i < 16; i++) noiseHit(bus, t0 + i * beat / 4, i % 2 ? 0.012 : 0.02, 0.012, 0.98, 0.3, r.next);
        if (level >= 2) for (const at of [0, 1.5, 3]) for (const m of [root, root + 7, root + 12]) note(bus, { table: SAW, hz: midiHz(m + 12), start: t0 + at * beat, dur: beat * 0.4, gain: 0.018, pan: m === root + 7 ? 0.3 : -0.3, attack: 0.004, decay: 0.1, sustain: 0.6, release: 0.08, cutoff: 2600, reverb: 0.15 });
        // Last bar of a phrase: snare roll crescendo + a rising sweep into the next section.
        if (inPhrase === 7 && level >= 1) {
          for (let i = 0; i < 16; i++) noiseHit(bus, t0 + 2 * beat + i * beat / 8, 0.02 + 0.07 * (i / 15), 0.03, 0.7, (i % 2 ? 0.2 : -0.2), r.next);
          for (let i = 0; i < 6; i++) note(bus, { table: SAW, hz: midiHz(root + 12 + i * 2), start: t0 + i * beat * 0.66, dur: beat * 0.7, gain: 0.008 + i * 0.003, pan: 0, attack: 0.1, decay: 0.2, sustain: 0.8, release: 0.1, cutoff: 1500 + i * 600, reverb: 0.5 });
        }
        // Section downbeat: a big impact.
        if (inPhrase === 0 && b > 0) { kick(bus, t0, 0.45); noiseHit(bus, t0, 0.14, 0.25, 0.3, 0, r.next); note(bus, { table: SINE, hz: midiHz(root - 24), start: t0, dur: bar, gain: 0.12, attack: 0.002, pluck: 1.2, release: 0.6, reverb: 0.3 }); }
        break;
      }
      case 'comedy': {
        // Bouncy pizzicato: bass on 1 and 3, chord plucks on the off-beats, a skipping melody.
        for (const i of [0, 2]) note(bus, { table: BASS, hz: midiHz(ch[0] - 12), start: t0 + i * beat, dur: beat * 0.5, gain: 0.06, attack: 0.004, pluck: 0.2, release: 0.05, cutoff: 900, reverb: 0.05 });
        for (const i of [1, 3]) for (const m of ch) note(bus, { table: PLUCK, hz: midiHz(m + 12), start: t0 + i * beat, dur: beat * 0.25, gain: 0.03, pan: 0.2, attack: 0.002, pluck: 0.1, release: 0.05, cutoff: 3600, reverb: 0.2 });
        if (full) for (let i = 0; i < 8; i++) if (r.next() < 0.6) note(bus, { table: PLUCK, hz: midiHz(ch[(i + b) % 3] + 24 + (i % 4 === 3 ? 2 : 0)), start: t0 + i * beat / 2, dur: beat * 0.2, gain: 0.035, pan: -0.2, attack: 0.002, pluck: 0.09, release: 0.05, cutoff: 4200, reverb: 0.25 });
        break;
      }
      case 'upbeat': {
        // Bouncy off-beat chord stabs, walking bass, claps on 2 and 4.
        for (let i = 0; i < 4; i++) for (const m of ch) note(bus, { table: PLUCK, hz: midiHz(m + 12), start: t0 + i * beat + beat / 2, dur: beat * 0.35, gain: 0.03, pan: 0.2, attack: 0.003, pluck: 0.16, release: 0.08, cutoff: 3800, reverb: 0.25 });
        const walk = [0, 2, 1, 2];
        for (let i = 0; i < 4; i++) note(bus, { table: BASS, hz: midiHz(ch[walk[i] % 3] - 12), start: t0 + i * beat, dur: beat * 0.8, gain: 0.05, attack: 0.005, pluck: 0.35, release: 0.1, cutoff: 800, reverb: 0.05 });
        if (full) { for (let i = 0; i < 4; i++) kick(bus, t0 + i * beat, i % 2 ? 0.1 : 0.16); noiseHit(bus, t0 + beat, 0.05, 0.06, 0.6, 0, r.next); noiseHit(bus, t0 + beat * 3, 0.05, 0.06, 0.6, 0, r.next); }
        break;
      }
    }
  }
}

export function moodFor(category: string, subGenre = ''): MusicMood {
  const c = String(category ?? '').toLowerCase(), s = String(subGenre ?? '').toLowerCase();
  if (c === 'news') return 'news';
  if (c === 'tech') return 'tech';
  if (c === 'cooking' || c === 'ads') return 'upbeat';
  if (/horror|scary|suspense|creepy/.test(s)) return 'horror';
  if (/mystery|crime|thriller|twist/.test(s)) return 'mystery';
  if (/love|romance|sad|emotional|drama|inspir/.test(s)) return 'emotional';
  if (/comed|funny/.test(s)) return 'comedy';
  if (/sci-?fi|science/.test(s)) return 'tech';
  return 'story';
}

/** Compose `seconds` of original, mastered stereo music (Float32 samples at 44.1 kHz). */
export function composeBuffers(mood: MusicMood, seconds: number, seed: string): { L: Float32Array; R: Float32Array; sampleRate: number } {
  const total = Math.ceil((seconds + 2) * SR);
  const bus: Bus = { L: new Float32Array(total), R: new Float32Array(total), send: new Float32Array(total) };
  const r = rng(`${seed}:${mood}`);
  compose(bus, mood, seconds + 2, r);
  reverb(bus, mood === 'horror' || mood === 'mystery' ? 0.55 : 0.35);
  // Master: fade in/out, gentle saturation, loudness to about −16 dBFS RMS, peak ≤ −1 dBFS.
  const fadeIn = Math.floor(1.2 * SR), fadeOut = Math.floor(2 * SR);
  let sum = 0;
  for (let n = 0; n < total; n++) {
    const g = Math.min(1, n / fadeIn, (total - n) / fadeOut);
    bus.L[n] = Math.tanh(bus.L[n] * 1.4) / 1.4 * g;
    bus.R[n] = Math.tanh(bus.R[n] * 1.4) / 1.4 * g;
    sum += bus.L[n] * bus.L[n] + bus.R[n] * bus.R[n];
  }
  const rms = Math.sqrt(sum / (total * 2)) || 1e-6;
  let peak = 0;
  for (let n = 0; n < total; n++) peak = Math.max(peak, Math.abs(bus.L[n]), Math.abs(bus.R[n]));
  const gain = Math.min(Math.pow(10, -16 / 20) / rms, 0.89 / (peak || 1));
  for (let n = 0; n < total; n++) { bus.L[n] *= gain; bus.R[n] *= gain; }
  return { L: bus.L, R: bus.R, sampleRate: SR };
}

/**
 * Shape the music for a narrated video, in place:
 *  - EQ: high-pass at 90 Hz (phone speakers can't play sub-bass; it only eats
 *    headroom), a small lift of the melodic body around 700 Hz, and a dip at
 *    2.8 kHz — the "vocal pocket" where speech intelligibility lives.
 *  - Level: `speechGain` while the presenter talks, `pauseGain` in real pauses
 *    (≥ 0.7 s) and before / after the narration, with smooth ramps, so the music
 *    swells between sentences and sits under the voice while it speaks.
 */
export function eqForVoice(L: Float32Array, R: Float32Array, sampleRate: number) {
  const biquad = (type: 'hp' | 'peak', f0: number, q: number, gainDb = 0) => {
    const w = (TAU * f0) / sampleRate, cw = Math.cos(w), al = Math.sin(w) / (2 * q), A = Math.pow(10, gainDb / 40);
    let b0, b1, b2, a0, a1, a2;
    if (type === 'hp') { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; }
    else { b0 = 1 + al * A; b1 = -2 * cw; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A; }
    return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  };
  const filters = [biquad('hp', 90, 0.707), biquad('hp', 90, 0.707), biquad('peak', 700, 0.9, 2), biquad('peak', 2800, 1.1, -4)];
  for (const ch of [L, R]) for (const [b0, b1, b2, a1, a2] of filters) {
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let n = 0; n < ch.length; n++) {
      const x = ch[n], y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = x; y2 = y1; y1 = y; ch[n] = y;
    }
  }
}

/** RMS level of a stereo buffer in dBFS. */
export function levelDb(L: Float32Array, R: Float32Array): number {
  let sum = 0;
  for (let n = 0; n < L.length; n++) sum += L[n] * L[n] + R[n] * R[n];
  return 10 * Math.log10(sum / (2 * Math.max(1, L.length)) + 1e-12);
}

/** Level automation from the word timings (see eqForVoice for the whole idea). */
export function automateLevel(L: Float32Array, R: Float32Array, sampleRate: number, o: { speech: { start: number; end: number }[]; speechGain: number; pauseGain: number }) {
  const N = L.length, target = new Float32Array(N).fill(o.pauseGain);
  const ramp = 0.35;
  for (const s of o.speech) {
    const a = Math.max(0, Math.floor((s.start - ramp) * sampleRate)), b = Math.min(N, Math.ceil((s.end + 0.15) * sampleRate));
    for (let n = a; n < b; n++) target[n] = o.speechGain;
  }
  // Smooth the steps into ~0.3 s fades (one-pole up, faster one-pole down so the music clears before a word).
  const up = 1 - Math.exp(-1 / (0.3 * sampleRate)), down = 1 - Math.exp(-1 / (0.12 * sampleRate));
  let g = target[0];
  for (let n = 0; n < N; n++) {
    g += (target[n] - g) * (target[n] > g ? up : down);
    L[n] *= g; R[n] *= g;
  }
}

/** 16-bit stereo WAV bytes from float samples. */
export function encodeWav(L: Float32Array, R: Float32Array, sampleRate = SR): Uint8Array {
  const total = L.length;
  const bytes = new Uint8Array(44 + total * 4);
  const v = new DataView(bytes.buffer);
  const str = (o: number, t: string) => { for (let i = 0; i < t.length; i++) bytes[o + i] = t.charCodeAt(i); };
  str(0, 'RIFF'); v.setUint32(4, 36 + total * 4, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, total * 4, true);
  for (let n = 0; n < total; n++) {
    v.setInt16(44 + n * 4, Math.max(-32767, Math.min(32767, Math.round(L[n] * 32767))), true);
    v.setInt16(46 + n * 4, Math.max(-32767, Math.min(32767, Math.round(R[n] * 32767))), true);
  }
  return bytes;
}

/** Composed music as a 16-bit stereo WAV file (bytes). */
export function composeWav(mood: MusicMood, seconds: number, seed: string): Uint8Array {
  const { L, R, sampleRate } = composeBuffers(mood, seconds, seed);
  return encodeWav(L, R, sampleRate);
}
