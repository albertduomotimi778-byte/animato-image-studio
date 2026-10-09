#!/usr/bin/env python3
"""Animato song generator — writes a full song (vocals + music) on the GitHub runner's CPU.

Model: ACE-Step 1.5 (MIT licence, https://github.com/ACE-Step/ACE-Step-1.5), the "turbo"
2B diffusion transformer, run WITHOUT its language-model planner (DiT-only mode). The
caption (style + singer + language) and the lyrics go straight to the diffusion model.

Commands
  download   fetch only the checkpoints DiT-only mode needs (turbo DiT, VAE, text encoder)
             into $ACESTEP_CHECKPOINTS_DIR and copy the model code next to them
  generate   compose one song from a JSON job and write <out>/song.mp3 + <out>/meta.json

Job JSON (written by .github/workflows/animato_music.yml from the dispatch inputs):
  {"title": "", "lyrics": "", "style": "afrobeats", "custom_style": "", "singer": "female_pop",
   "language": "en", "duration": 60, "seed": 1234, "out": "output"}

Every progress line starts with "[songgen]" and names the stage, so the run log reads:
  installing -> downloading model -> composing -> encoding mp3 -> done
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAIN_REPO = "ACE-Step/Ace-Step1.5"
DIT_CONFIG = "acestep-v15-turbo"
# DiT-only mode needs these three folders of the main repo; the 1.7B planner LM is skipped.
NEEDED_COMPONENTS = [DIT_CONFIG, "vae", "Qwen3-Embedding-0.6B"]
SAMPLE_RATE = 48000

MIN_DURATION, MAX_DURATION = 10, 300
MAX_LYRICS = 4000          # ACE-Step accepts up to 4096 characters

# acestep/constants.py VALID_LANGUAGES (+ "unknown" = let the model decide).
LANGUAGE_NAMES = {
    "ar": "Arabic", "az": "Azerbaijani", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "ko": "Korean", "la": "Latin",
    "lt": "Lithuanian", "ms": "Malay", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
    "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sk": "Slovak", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese (Mandarin)",
    "unknown": "",
}
# Languages the model has no code for, sung through the closest supported one.
LANGUAGE_ALIASES = {
    "pcm": ("en", "Nigerian Pidgin English"),
}

# Style presets: caption words + a typical tempo (the tempo helps DiT-only mode a lot).
STYLES = {
    "pop": ("pop, catchy melody, bright synths, punchy drums, radio-ready", 116),
    "afrobeats": ("afrobeats, afropop groove, syncopated percussion, shekere, talking drum, bouncy bass, melodic guitar licks", 104),
    "amapiano": ("amapiano, log drum bass, deep house piano chords, shakers, laid-back south african groove", 113),
    "hiphop": ("hip-hop, boom bap drums, deep bass, sampled keys, head-nodding groove", 92),
    "trap": ("trap, booming 808 bass, rapid hi-hats, dark synth melody, hard-hitting", 140),
    "rnb": ("contemporary r&b, smooth electric piano, lush harmonies, slow sensual groove", 76),
    "gospel": ("gospel, hammond organ, piano, choir harmonies, uplifting worship, hand claps", 82),
    "highlife": ("highlife, west african guitar melodies, brass section, palm-wine groove, joyful", 120),
    "reggae": ("reggae, offbeat skank guitar, one drop drums, warm bass, island vibe", 76),
    "dancehall": ("dancehall, riddim, digital drums, energetic caribbean bounce", 98),
    "rock": ("rock, distorted electric guitars, live drums, driving bass guitar, energetic", 128),
    "edm": ("edm, electronic dance music, festival synth leads, four on the floor kick, big build-up and drop", 126),
    "lofi": ("lo-fi hip hop, mellow jazzy chords, vinyl crackle, relaxed dusty beat", 80),
    "jazz": ("jazz, swing feel, upright bass, brushed drums, piano, saxophone", 120),
    "country": ("country, acoustic guitar, pedal steel, fiddle, heartfelt storytelling", 100),
    "kpop": ("k-pop, polished dance pop, punchy synths, energetic hooks", 124),
    "latin": ("latin pop, reggaeton dembow rhythm, nylon guitar, latin percussion", 95),
    "cinematic": ("cinematic orchestral, epic strings, brass, choir, timpani, film score", 90),
    "acoustic": ("acoustic, unplugged, fingerpicked acoustic guitar, soft piano, intimate, warm room sound", 92),
}
SINGERS = {
    "female_pop": "female vocal, bright pop voice",
    "female_soul": "female vocal, soulful powerful r&b voice, rich runs",
    "female_soft": "female vocal, soft breathy intimate voice, gentle",
    "male_pop": "male vocal, clear pop tenor",
    "male_deep": "male vocal, deep warm baritone",
    "male_rock": "male vocal, raspy gritty rock voice, powerful belting",
    "male_rap": "male rap vocal, rhythmic confident flow",
    "female_rap": "female rap vocal, sharp confident flow",
    "choir": "gospel choir, layered group vocals",
    "duet": "duet, male and female vocals, call and response harmonies",
}

T0 = time.time()


def log(stage, msg=""):
    print(f"[songgen] [{stage}] {msg}".rstrip(), flush=True)


def checkpoints_dir():
    d = os.environ.get("ACESTEP_CHECKPOINTS_DIR") or os.path.join(os.getcwd(), "checkpoints")
    return os.path.abspath(os.path.expanduser(d))


def has_weights(path):
    names = ("model.safetensors", "model.safetensors.index.json", "diffusion_pytorch_model.safetensors",
             "pytorch_model.bin", "diffusion_pytorch_model.bin")
    return os.path.isdir(path) and any(os.path.exists(os.path.join(path, n)) for n in names)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def sync_model_code(ckpt):
    """Copy acestep/models/turbo/*.py (except __init__.py) into the DiT checkpoint folder,
    exactly like acestep.model_downloader does after a download."""
    try:
        from acestep.model_downloader import _sync_model_code_files
        from pathlib import Path
        synced = _sync_model_code_files(DIT_CONFIG, Path(ckpt))
        if synced:
            log("downloading model", f"model code synced: {', '.join(synced)}")
            return
    except Exception as err:  # older/newer layout: do it by hand
        log("downloading model", f"built-in code sync unavailable ({err}); copying by hand")
    import acestep
    src = os.path.join(os.path.dirname(acestep.__file__), "models", "turbo")
    dst = os.path.join(ckpt, DIT_CONFIG)
    for name in sorted(os.listdir(src)):
        if name.endswith(".py") and name != "__init__.py":
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
            log("downloading model", f"copied {name}")


def cmd_download(_args):
    from huggingface_hub import snapshot_download
    ckpt = checkpoints_dir()
    os.makedirs(ckpt, exist_ok=True)
    missing = [c for c in NEEDED_COMPONENTS if not has_weights(os.path.join(ckpt, c))]
    if missing:
        log("downloading model", f"{MAIN_REPO}: {', '.join(missing)} -> {ckpt} (about 6.3 GB)")
        last = None
        for attempt in range(1, 4):
            try:
                snapshot_download(
                    repo_id=MAIN_REPO,
                    local_dir=ckpt,
                    allow_patterns=[f"{c}/*" for c in NEEDED_COMPONENTS],
                    max_workers=8,
                )
                last = None
                break
            except Exception as err:
                last = err
                log("downloading model", f"attempt {attempt} failed: {err}")
                time.sleep(10 * attempt)
        if last is not None:
            raise SystemExit(f"[songgen] could not download the music model: {last}")
    still = [c for c in NEEDED_COMPONENTS if not has_weights(os.path.join(ckpt, c))]
    if still:
        raise SystemExit(f"[songgen] model download incomplete, missing: {', '.join(still)}")
    if not os.path.exists(os.path.join(ckpt, DIT_CONFIG, "silence_latent.pt")):
        raise SystemExit("[songgen] model download incomplete: silence_latent.pt is missing")
    sync_model_code(ckpt)
    log("downloading model", f"ready in {time.time() - T0:.0f}s")


# ---------------------------------------------------------------------------
# prompt building
# ---------------------------------------------------------------------------
def clean_text(s, n):
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def resolve_language(code):
    code = str(code or "").strip().lower()
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]
    if code in LANGUAGE_NAMES:
        return code, LANGUAGE_NAMES[code]
    return "unknown", ""


def build_caption(style, custom_style, singer, language_label, instrumental):
    parts = []
    preset = STYLES.get(style)
    bpm = None
    if preset:
        parts.append(preset[0])
        bpm = preset[1]
    custom = clean_text(custom_style, 300)
    if custom:
        parts.append(custom)
        if not preset:
            bpm = None
    if not parts:
        parts.append(STYLES["pop"][0])
        bpm = STYLES["pop"][1]
    if instrumental:
        parts.append("instrumental, no vocals")
    else:
        parts.append(SINGERS.get(singer, SINGERS["female_pop"]))
        if language_label:
            parts.append(f"sung in {language_label}")
        parts.append("clear expressive lead vocal")
    parts.append("professionally mixed and mastered, high quality")
    # One flat, de-duplicated tag list (custom words often repeat the preset's).
    tags, seen = [], set()
    for part in parts:
        for tag in part.split(","):
            tag = tag.strip()
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                tags.append(tag)
    return ", ".join(tags)[:900], bpm


TAG_RE = re.compile(r"^\s*\[[^\]\n]{1,60}\]\s*$")


def _norm(par):
    return re.sub(r"[^\w]+", " ", par.lower()).strip()


def format_lyrics(raw):
    """Return lyrics with [Section] tags. Lyrics that already use tags are kept as written;
    plain text is split into sections (blank lines, else 4 lines each) and labelled:
    repeated sections become the chorus, the others verses (a late one becomes the bridge)."""
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.split("\n")]
    if any(TAG_RE.match(ln) for ln in lines):
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:MAX_LYRICS]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) == 1:
        plain = [ln.strip() for ln in paragraphs[0].split("\n") if ln.strip()]
        if len(plain) > 6:
            paragraphs = ["\n".join(plain[i:i + 4]) for i in range(0, len(plain), 4)]
    keys = [_norm(p) for p in paragraphs]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    has_repeat = any(v > 1 for v in counts.values())

    labels = []
    verse = 0
    for i, k in enumerate(keys):
        if has_repeat:
            is_chorus = counts[k] > 1
        else:
            # No repeats: alternate verse / chorus so the song gets a hook.
            is_chorus = len(keys) > 1 and i % 2 == 1
        if is_chorus:
            labels.append("Chorus")
        else:
            verse += 1
            labels.append(f"Verse {verse}" if len(keys) > 1 else "Verse")
    # A late, non-chorus section in a longer song works best as a bridge.
    if len(keys) >= 5:
        for j in range(len(keys) - 2, 1, -1):
            if labels[j].startswith("Verse"):
                labels[j] = "Bridge"
                break
    out = ["[Intro]", ""]
    for label, par in zip(labels, paragraphs):
        out += [f"[{label}]", par, ""]
    out.append("[Outro]")
    return "\n".join(out).strip()[:MAX_LYRICS]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def write_mp3(tensor, sample_rate, out_dir):
    import numpy as np
    import soundfile as sf
    audio = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]
    audio = np.clip(audio, -1.0, 1.0).T          # [samples, channels]
    wav = os.path.join(out_dir, "song.wav")
    mp3 = os.path.join(out_dir, "song.mp3")
    sf.write(wav, audio, int(sample_rate), subtype="PCM_16")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
         "-af", "afade=t=out:st={:.2f}:d=1.5".format(max(0.0, audio.shape[0] / float(sample_rate) - 1.5)),
         "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", mp3],
        check=True,
    )
    os.remove(wav)
    return mp3, audio.shape[0] / float(sample_rate)


def _http(method, url, body=None, key="", timeout=60):
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def start_server(repo, ckpt, key, port, use_lm):
    """ACE-Step's own REST server (the same set-up the video renderer uses)."""
    threads = str(os.cpu_count() or 4)
    env = dict(os.environ)
    env.update({
        "ACESTEP_API_HOST": "127.0.0.1", "ACESTEP_API_PORT": str(port), "ACESTEP_API_KEY": key,
        "ACESTEP_DEVICE": "cpu", "ACESTEP_INIT_LLM": "true" if use_lm else "false",
        "ACESTEP_LM_BACKEND": "pt", "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B", "ACESTEP_LM_DEVICE": "cpu",
        "ACESTEP_USE_FLASH_ATTENTION": "false", "ACESTEP_CONFIG_PATH": DIT_CONFIG,
        "ACESTEP_PROJECT_ROOT": repo, "ACESTEP_CHECKPOINTS_DIR": ckpt,
        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "TOKENIZERS_PARALLELISM": "false",
        # ACE-Step aborts a generation after 600 s by default (tuned for GPUs). A CPU needs longer
        # for long songs — this default was what made songs "crash at composing".
        "ACESTEP_GENERATION_TIMEOUT": os.environ.get("ACESTEP_GENERATION_TIMEOUT", "3600"),
        "ACESTEP_DISABLE_TQDM": "1",
    })
    logf = open(os.path.join(os.path.dirname(ckpt), "api_server.log"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "acestep.api_server"], cwd=repo, env=env, stdout=logf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 15 * 60:
        if proc.poll() is not None:
            tail = open(logf.name, encoding="utf-8", errors="replace").read()[-2500:]
            raise RuntimeError("the music engine stopped while loading:\n" + tail)
        try:
            st, _ = _http("GET", base + "/health", timeout=5)
            if st == 200:
                log("composing", f"music engine ready in {time.time() - t0:.0f}s" + (" (with the song-planning model)" if use_lm else ""))
                return proc, base, logf.name
        except Exception:
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("the music engine did not start within 15 minutes")


def compose(base, key, body, deadline):
    task_id = ""
    for attempt in range(3):
        try:
            st, raw = _http("POST", base + "/release_task", body, key, timeout=180)
            rj = json.loads(raw.decode("utf-8", "replace") or "{}")
            task_id = str((rj.get("data") or {}).get("task_id") or rj.get("task_id") or "").strip()
            if task_id:
                break
            log("composing", f"request not accepted: {raw[:300]!r}")
        except Exception as err:
            log("composing", f"request attempt {attempt + 1} failed: {err}")
        time.sleep(8 * (attempt + 1))
    if not task_id:
        raise RuntimeError("the music engine did not accept the song request")
    log("composing", f"task {task_id[:24]} accepted; composing…")
    t0, n = time.time(), 0
    while time.time() < deadline:
        time.sleep(5 if n < 6 else 10)
        n += 1
        try:
            st, raw = _http("POST", base + "/query_result", {"task_id_list": [task_id]}, key, timeout=45)
            qj = json.loads(raw.decode("utf-8", "replace") or "{}")
        except Exception:
            continue
        data = qj.get("data") if isinstance(qj, dict) else qj
        item = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        status = int(item.get("status") or 0)
        if status == 2:
            raise RuntimeError("the music engine could not make the song: " + str(item.get("error") or item.get("result") or "failed")[:400])
        if status != 1:
            if n % 6 == 0:
                log("composing", f"still composing ({time.time() - t0:.0f}s)…")
            continue
        res = item.get("result")
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                res = []
        first = res[0] if isinstance(res, list) and res else (res or {})
        ref = str(first.get("file") or first.get("audio_url") or first.get("url") or "")
        if not ref:
            raise RuntimeError("the music engine finished but returned no audio")
        if ref.startswith("data:audio"):
            import base64
            return base64.b64decode(ref.split(",", 1)[1]), time.time() - t0
        url = ref if re.match(r"^https?://", ref) else base + ("" if ref.startswith("/") else "/") + ref
        for d in range(3):
            try:
                st, audio = _http("GET", url, None, key, timeout=240)
                if len(audio) > 10000:
                    return audio, time.time() - t0
            except Exception as err:
                log("composing", f"download attempt {d + 1} failed: {err}")
            time.sleep(5)
        raise RuntimeError("the song was made but could not be downloaded")
    raise RuntimeError("the song took too long to compose")


def finish_mp3(raw, out_dir):
    src = os.path.join(out_dir, "raw_song.bin")
    open(src, "wb").write(raw)
    mp3 = os.path.join(out_dir, "song.mp3")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", src], capture_output=True, text=True)
    try:
        seconds = float(probe.stdout.strip())
    except ValueError:
        seconds = 0.0
    fade = "afade=t=out:st={:.2f}:d=1.5,".format(max(0.0, seconds - 1.5)) if seconds > 4 else ""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-af", fade + "loudnorm=I=-14:TP=-1.2:LRA=11",
                    "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", mp3], check=True)
    os.remove(src)
    return mp3, seconds


def cmd_generate(args):
    job = json.load(open(args.job, encoding="utf-8"))
    out_dir = os.path.abspath(job.get("out") or "output")
    os.makedirs(out_dir, exist_ok=True)
    try:
        duration = int(float(job.get("duration") or 0))
    except ValueError:
        duration = 0
    try:
        seed = int(float(job.get("seed") or 0)) % (2 ** 32 - 1)
    except ValueError:
        seed = 1
    style = str(job.get("style") or "pop").strip().lower()
    singer = str(job.get("singer") or "female_pop").strip().lower()
    lang_code, lang_label = resolve_language(job.get("language"))
    lyrics = format_lyrics(job.get("lyrics", ""))
    instrumental = not lyrics
    caption, bpm = build_caption(style, job.get("custom_style", ""), singer, lang_label, instrumental)
    title = clean_text(job.get("title"), 120)
    # Long enough for every word: ~2.2 sung words a second plus intro, breaks and outro.
    words = len(re.findall(r"[^\s\[\]]+", re.sub(r"\[[^\]]*\]", " ", lyrics)))
    needed = int(words / 2.2 + 20) if words else 0
    if needed > duration:
        log("composing", f"{words} words need about {needed}s — making the song that long")
        duration = needed
    duration = max(MIN_DURATION, min(MAX_DURATION, duration or 60))

    log("composing", f"title={title!r} style={style} singer={singer} language={lang_code} duration={duration}s seed={seed}")
    log("composing", f"caption: {caption}")

    repo = os.path.abspath(os.path.expanduser(os.environ.get("ACESTEP_REPO") or "~/acestep-local/repo"))
    ckpt = os.path.join(repo, "checkpoints")
    key = "animato-" + os.urandom(8).hex()
    body = {
        "prompt": caption, "caption": caption, "lyrics": lyrics or "[Instrumental]", "audio_duration": duration,
        "audio_format": "mp3", "batch_size": 1, "vocal_language": lang_code if not instrumental else "unknown",
        "time_signature": "4", "use_cot_caption": False, "ai_token": key, "seed": seed, "use_random_seed": False,
    }
    if bpm:
        body["bpm"] = bpm
    have_lm = os.path.isdir(os.path.join(ckpt, "acestep-5Hz-lm-0.6B"))
    deadline = T0 + 120 * 60
    last = None
    raw, gen_seconds = None, 0.0
    t_gen = time.time()
    for use_lm in ([True, False] if have_lm and os.environ.get("SONG_USE_LM") == "true" else [False]):
        proc = None
        try:
            proc, base, logname = start_server(repo, ckpt, key, 8011, use_lm)
            body["thinking"] = use_lm
            raw, gen_seconds = compose(base, key, body, deadline)
            break
        except Exception as err:
            last = err
            log("composing", f"{'with' if use_lm else 'without'} the planning model failed: {err}")
            raw = None
        finally:
            if proc is not None:
                proc.kill()
    if raw is None:
        raise SystemExit(f"[songgen] the song could not be made: {last}")
    log("composing", f"composed in {gen_seconds:.0f}s")

    log("encoding mp3")
    mp3, seconds = finish_mp3(raw, out_dir)
    size = os.path.getsize(mp3)
    meta = {
        "title": title, "style": style, "custom_style": clean_text(job.get("custom_style"), 300),
        "singer": singer, "language": lang_code, "language_label": lang_label or "auto",
        "instrumental": instrumental, "duration": duration, "seconds": round(seconds, 2), "seed": seed,
        "bpm": bpm, "caption": caption, "lyrics": lyrics, "model": f"ACE-Step 1.5 {DIT_CONFIG} (CPU)",
        "generate_seconds": round(time.time() - t_gen, 1), "total_seconds": round(time.time() - T0, 1), "bytes": size,
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("done", f"{mp3} ({size / 1e6:.1f} MB, {seconds:.0f}s of audio)")


# ---------------------------------------------------------------------------
# serve: the warm "music studio" session
# ---------------------------------------------------------------------------
def song_request(req):
    """App job → (body for ACE-Step, info for meta, expected seconds of composing)."""
    try:
        duration = int(float(req.get("duration") or 0))
    except (TypeError, ValueError):
        duration = 0
    try:
        seed = int(float(req.get("seed") or 0)) % (2 ** 32 - 1)
    except (TypeError, ValueError):
        seed = 1
    style = str(req.get("style") or "pop").strip().lower()
    singer = str(req.get("singer") or "female_pop").strip().lower()
    lang_code, lang_label = resolve_language(req.get("language"))
    lyrics = format_lyrics(req.get("lyrics", ""))
    instrumental = not lyrics
    caption, bpm = build_caption(style, req.get("customStyle") or req.get("custom_style") or "", singer, lang_label, instrumental)
    words = len(re.findall(r"[^\s\[\]]+", re.sub(r"\[[^\]]*\]", " ", lyrics)))
    needed = int(words / 2.2 + 20) if words else 0
    duration = max(MIN_DURATION, min(MAX_DURATION, max(duration, needed) or 60))
    body = {
        "prompt": caption, "caption": caption, "lyrics": lyrics or "[Instrumental]", "audio_duration": duration,
        "audio_format": "mp3", "batch_size": 1, "vocal_language": lang_code if not instrumental else "unknown",
        "time_signature": "4", "use_cot_caption": False, "thinking": False, "seed": seed, "use_random_seed": False,
    }
    if bpm:
        body["bpm"] = bpm
    info = {"title": clean_text(req.get("title"), 120), "style": style, "singer": singer, "language": lang_code,
            "language_label": lang_label or "auto", "instrumental": instrumental, "duration": duration, "seed": seed, "bpm": bpm}
    return body, info


def app_call(app, method, route, body=None, raw=None, headers=None, timeout=60):
    import urllib.request
    data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(app + route, data=data, method=method)
    req.add_header("User-Agent", "AnimatoMusicStudio/1.0")
    if raw is None and data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode("utf-8", "replace")
        return json.loads(txt or "{}")


def cmd_serve(_args):
    import threading
    import urllib.parse
    app = os.environ.get("APP_URL", "").rstrip("/")
    token = os.environ.get("SESSION_TOKEN", "")
    idle_min = float(os.environ.get("IDLE_MINUTES") or 15)
    warm_only = os.environ.get("WARMUP") == "true"
    q = "token=" + urllib.parse.quote(token)
    repo = os.path.abspath(os.path.expanduser(os.environ.get("ACESTEP_REPO") or "~/acestep-local/repo"))
    ckpt = os.path.join(repo, "checkpoints")
    key = "animato-" + os.urandom(8).hex()

    proc, base, _ = start_server(repo, ckpt, key, 8011, False)
    if warm_only:
        log("done", "music engine installed and loaded once — cached for the next songs")
        proc.kill()
        return
    app_call(app, "POST", "/api/music/runner/hello", {"token": token, "info": {"cpus": os.cpu_count()}})
    log("ready", f"studio open (stops after {idle_min:.0f} idle minutes)")
    last_work = time.time()
    while time.time() - last_work < idle_min * 60:
        if proc.poll() is not None:
            log("ready", "the engine stopped — restarting it")
            proc, base, _ = start_server(repo, ckpt, key, 8011, False)
        try:
            nxt = app_call(app, "GET", f"/api/music/runner/next?{q}", timeout=40)
        except Exception as err:
            log("ready", f"app not reachable ({err}); retrying")
            time.sleep(6)
            continue
        if nxt.get("stop"):
            break
        job = nxt.get("job")
        if not job:
            time.sleep(2.5)
            continue
        jid = job["id"]
        body, info = song_request(job.get("req") or {})
        body["ai_token"] = key
        # Expected composing time grows with the length of the song (rough CPU figure).
        expected = 40 + info["duration"] * 1.8
        state = {"done": False}
        t0 = time.time()

        def beat():
            while not state["done"]:
                frac = min(1.0, (time.time() - t0) / expected)
                pct = 25 + int(70 * (1 - (1 - frac) ** 1.6))
                try:
                    app_call(app, "POST", "/api/music/runner/update", {"token": token, "id": jid, "patch": {"percent": pct, "stage": "Composing your song…" if frac < 0.97 else "Adding the final touches…"}}, timeout=20)
                except Exception:
                    pass
                for _ in range(16):
                    if state["done"]:
                        break
                    time.sleep(0.5)

        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            raw, gen_seconds = compose(base, key, body, time.time() + 55 * 60)
            state["done"] = True
            app_call(app, "POST", "/api/music/runner/update", {"token": token, "id": jid, "patch": {"percent": 97, "stage": "Mastering…"}}, timeout=20)
            out_dir = os.path.join(os.path.dirname(ckpt), "out", jid)
            os.makedirs(out_dir, exist_ok=True)
            mp3, seconds = finish_mp3(raw, out_dir)
            meta = dict(info, seconds=round(seconds, 2), generate_seconds=round(gen_seconds, 1))
            data = open(mp3, "rb").read()
            for attempt in range(4):
                try:
                    app_call(app, "POST", f"/api/music/runner/result?{q}&id={urllib.parse.quote(jid)}", raw=data,
                             headers={"Content-Type": "audio/mpeg", "X-Song-Meta": urllib.parse.quote(json.dumps(meta))}, timeout=180)
                    break
                except Exception as err:
                    log("done", f"upload attempt {attempt + 1} failed: {err}")
                    time.sleep(4 * (attempt + 1))
            log("done", f"song {jid}: {seconds:.0f}s of audio composed in {gen_seconds:.0f}s")
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception as err:
            state["done"] = True
            log("composing", f"song {jid} failed: {err}")
            try:
                app_call(app, "POST", "/api/music/runner/update", {"token": token, "id": jid, "patch": {"state": "error", "error": "The song could not be made this time. Please try again."}}, timeout=20)
            except Exception:
                pass
        last_work = time.time()
    try:
        app_call(app, "POST", "/api/music/runner/bye", {"token": token}, timeout=20)
    except Exception:
        pass
    proc.kill()
    log("done", "studio closed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download the model checkpoints")
    g = sub.add_parser("generate", help="compose one song")
    g.add_argument("--job", required=True, help="job JSON file")
    sub.add_parser("lyrics", help="print how stdin lyrics would be structured (debug)")
    sub.add_parser("serve", help="warm studio session: keep the engine loaded and compose the app's queued songs")
    args = ap.parse_args()
    try:
        if args.cmd == "download":
            cmd_download(args)
        elif args.cmd == "generate":
            cmd_generate(args)
        elif args.cmd == "serve":
            cmd_serve(args)
        else:
            print(format_lyrics(sys.stdin.read()))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
