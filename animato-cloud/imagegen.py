#!/usr/bin/env python3
"""Animato image generator — runs on the GitHub runner's CPU, no API keys.

Pipeline per image:
  clean/fit the prompt -> 1-step generation at the target aspect ratio ->
  detail upscale (Real-ESRGAN anime-video model when available, else Lanczos) ->
  light sharpen -> exact output size.

Model profiles (pick with IMAGEGEN_PROFILE, or per job with "profile"):
  character  DEFAULT. SDXS-512-DreamShaper: the same 1-step speed class as before, but distilled
             from DreamShaper — far better at cartoon characters, faces and bodies.
  classic    SDXS-512-0.9 with the original VAE (the first version we shipped).
  anime      character + the SDXS DreamShaper *anime* adapter (needs `peft`).
  detail     DreamShaper-8 LCM, 4 steps. Slower (tens of seconds on a CPU) but the most detailed.

Modes
  serve   long-running local HTTP server (the video renderer uses this; the model is
          loaded ONCE and every scene is drawn by the same process)
  batch   draw a JSON list of jobs and exit (used by the stand-alone image workflow)

Every optional stage fails safe: if the upscaler is unavailable or slow the image is still
produced with a plain high-quality resize, and a model that cannot load falls back to the next.
"""
import argparse
import urllib.parse
import gc
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_AREA = 512 * 512
MAX_TOKENS = 77
UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
UPSCALER_FILE = "realesr-animevideov3.pth"
# If one AI upscale takes longer than this, switch to the plain resize for the rest of the run.
MAX_UPSCALE_SECONDS = float(os.environ.get("IMAGEGEN_MAX_UPSCALE_SECONDS", "25"))

PROFILES = {
    "character": {
        "label": "Cartoon characters (DreamShaper, 1 step)",
        "models": ["IDKiro/sdxs-512-dreamshaper", os.environ.get("IMAGEGEN_MODEL", "rupeshs/sdxs-512-0.9-orig-vae")],
        "steps": 1, "guidance": 0.0, "vae_large": True,
    },
    "classic": {
        "label": "Classic (SDXS 0.9, first version)",
        "models": [os.environ.get("IMAGEGEN_MODEL", "rupeshs/sdxs-512-0.9-orig-vae")],
        "steps": 1, "guidance": 0.0,
    },
    "anime": {
        "label": "Anime cartoon (DreamShaper + anime adapter, 1 step)",
        "models": ["IDKiro/sdxs-512-dreamshaper"],
        "adapter": "IDKiro/sdxs-512-dreamshaper-anime",
        "steps": 1, "guidance": 0.0, "vae_large": True,
    },
    "detail": {
        "label": "Detail (DreamShaper-8 LCM, 4 steps — slower)",
        "models": ["Lykon/dreamshaper-8-lcm"],
        "steps": 4, "guidance": 1.0, "lcm": True,
        # The tiny decoder (TAESD) makes the decode step several times faster on a CPU.
        "taesd": os.environ.get("IMAGEGEN_TAESD", "1") != "0",
    },
}
DEFAULT_PROFILE = os.environ.get("IMAGEGEN_PROFILE", "character")
if DEFAULT_PROFILE not in PROFILES:
    DEFAULT_PROFILE = "character"

STATE = {"ready": False, "error": "", "upscaler": "lanczos", "count": 0, "load_seconds": 0.0,
         "profile": "", "model": "", "label": ""}
LOCK = threading.Lock()
PIPE = None          # the loaded pipeline
LOADED = {}          # profile settings of the loaded pipeline
UPSCALER = None


def log(msg):
    print(f"[imagegen] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Story prompts: keep the model on what it draws well
# ---------------------------------------------------------------------------
STORY_STYLE = "3D animated family-film still, medium shot, cute cartoon characters with natural proportions, big expressive eyes, soft shading, warm lighting"

# Style words the style prefix already says (they would only eat the 77-token budget).
_FILLER = re.compile(
    r"\b(high[- ]end|3d animated|animated|feature[- ]film|family[- ]film|still (?:from|of)|render(?:ed)?|cinematic|"
    r"highly detailed|ultra[- ]detailed|8k|4k|masterpiece|wholesome|painterly|saturated|"
    r"no text|no watermark|no captions|no logos?)\b", re.I)

# What small one-step models draw badly: full-body action, crowds, fingers. Swap for calm medium shots.
_CALM = [
    (re.compile(r"\bfull[- ](?:body|length)\b|\b(?:wide|long|establishing) shot\b|\bfar away\b", re.I), "medium shot"),
    (re.compile(r"\b(?:through|into|in|among|across|amid) (?:a |the )?(?:crowds?|groups? of [a-z]+|many (?:people|characters|children|kids|villagers))\b", re.I), "beside a friend"),
    (re.compile(r"\b(?:crowds?|groups? of [a-z]+|many (?:people|characters|children|kids|villagers))\b", re.I), "two friends"),
    (re.compile(r"\b(?:running|sprinting|racing|dashing)\b", re.I), "walking"),
    (re.compile(r"\b(?:jumping|leaping|flying through the air|mid-?air|somersaults?|flipping)\b", re.I), "cheering"),
    (re.compile(r"\b(?:fighting|wrestling|punching|kicking|battling)\b", re.I), "facing each other"),
    (re.compile(r"\bdancing\b", re.I), "swaying happily"),
    (re.compile(r"\b(?:holding hands|clasped hands?|raised hands?|waving hands?|pointing (?:a |his |her |their )?fingers?|fingers?|fists?)\b", re.I), ""),
]


def clean_story_prompt(text, max_chars=380):
    out = " ".join((text or "").split())
    for pattern, repl in _CALM:
        out = pattern.sub(repl, out)
    out = _FILLER.sub(" ", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,.;:!?])(?:\s*[,.;:!?])+", r"\1", out)
    out = re.sub(r"^[\s,.;:!?]+|[\s,.;:!?]+$", "", out)
    return out[:max_chars]


# ---------------------------------------------------------------------------
# Upscaler (optional)
# ---------------------------------------------------------------------------
class Upscaler:
    def __init__(self):
        self.model = None
        self.ok = False
        self.name = "lanczos"
        mode = os.environ.get("IMAGEGEN_UPSCALER", "auto").strip().lower()
        if mode in ("off", "none", "lanczos", "0", "false"):
            return
        try:
            self._load()
        except Exception as err:  # never block image generation
            log(f"AI upscaler unavailable, using the plain resize ({str(err)[:160]})")

    def _weights(self):
        cache = os.path.expanduser(os.environ.get("IMAGEGEN_CACHE", "~/.cache/animato-imagegen"))
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, UPSCALER_FILE)
        if os.path.exists(path) and os.path.getsize(path) > 500_000:
            return path
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(UPSCALER_URL, headers={"User-Agent": "AnimatoImagegen/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 500_000:
                    raise RuntimeError("download too small")
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                return path
            except Exception as err:
                last = err
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"could not download the upscaler weights ({last})")

    def _load(self):
        from spandrel import ImageModelDescriptor, ModelLoader

        model = ModelLoader().load_from_file(self._weights())
        if not isinstance(model, ImageModelDescriptor):
            raise RuntimeError("unexpected upscaler model type")
        model.cpu().eval()
        self.model = model
        self.ok = True
        self.name = "realesr-animevideov3"

    def run(self, img):
        """Returns the upscaled PIL image, or None when the upscaler is off / failed."""
        if not self.ok:
            return None
        try:
            import numpy as np
            import torch
            from PIL import Image

            t0 = time.time()
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
            with torch.inference_mode():
                y = self.model(x)
            y = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).mul(255.0).round().byte().numpy()
            out = Image.fromarray(y)
            took = time.time() - t0
            if took > MAX_UPSCALE_SECONDS:
                self.ok = False
                STATE["upscaler"] = "lanczos"
                log(f"AI upscale took {took:.0f}s — switching to the plain resize for the rest of this run")
            return out
        except Exception as err:
            self.ok = False
            STATE["upscaler"] = "lanczos"
            log(f"AI upscale failed, using the plain resize ({str(err)[:160]})")
            return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _load_one(cfg, model_id):
    import torch

    try:
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as err:
        log(f"AutoPipeline load of {model_id} failed ({str(err)[:120]}), trying StableDiffusionPipeline")
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    if cfg.get("lcm"):
        from diffusers import LCMScheduler

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if cfg.get("vae_large") and os.environ.get("IMAGEGEN_VAE", "large").lower() != "tiny":
        try:  # the sharper decoder (the default one is the tiny TAESD)
            from diffusers import AutoencoderKL

            pipe.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae_large", torch_dtype=torch.float32)
        except Exception as err:
            log(f"large VAE not available for {model_id}, keeping the default ({str(err)[:100]})")
    if cfg.get("taesd"):
        try:
            from diffusers import AutoencoderTiny

            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
        except Exception as err:
            log(f"tiny decoder not available, keeping the full one ({str(err)[:100]})")
    if cfg.get("adapter"):
        from peft import PeftModel

        pipe.unet = PeftModel.from_pretrained(pipe.unet, cfg["adapter"])
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cpu")
    return pipe


def load_profile(name):
    """Loads a profile (trying its models in order). Returns (pipe, settings)."""
    import torch

    cfg = PROFILES[name]
    threads = int(os.environ.get("IMAGEGEN_THREADS") or os.cpu_count() or 2)
    torch.set_num_threads(max(1, threads))
    t0 = time.time()
    last = None
    for model_id in cfg["models"]:
        try:
            pipe = _load_one(cfg, model_id)
            break
        except Exception as err:
            last = err
            log(f"could not load {model_id} ({str(err)[:160]})")
    else:
        raise RuntimeError(f"no model of profile '{name}' could be loaded ({str(last)[:200]})")
    # Warm-up: the first call is always slower (kernel setup); pay for it before the first scene.
    try:
        with torch.inference_mode():
            pipe(prompt="warm up", num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"], width=256, height=256)
    except Exception as err:
        log(f"warm-up skipped ({str(err)[:120]})")
    settings = {"name": name, "model": model_id, "steps": cfg["steps"], "guidance": cfg["guidance"], "label": cfg["label"]}
    STATE.update(profile=name, model=model_id, label=cfg["label"], load_seconds=round(time.time() - t0, 1))
    log(f"model '{name}' ({model_id}) ready in {STATE['load_seconds']}s")
    return pipe, settings


def ensure_profile(name):
    """Makes `name` the loaded profile (called with LOCK held). Only one model stays in memory."""
    global PIPE, LOADED
    name = name if name in PROFILES else DEFAULT_PROFILE
    if PIPE is not None and LOADED.get("name") == name:
        return
    previous = LOADED.get("name")
    PIPE = None
    LOADED = {}
    gc.collect()
    try:
        PIPE, LOADED = load_profile(name)
    except Exception:
        if previous and previous != name:
            log(f"profile '{name}' failed — going back to '{previous}'")
            PIPE, LOADED = load_profile(previous)
        raise


def native_size(width, height):
    """Generation size: ~512x512 worth of pixels at the target aspect ratio (multiples of 8)."""
    if os.environ.get("IMAGEGEN_SQUARE") == "1":
        return 512, 512
    ar = max(0.4, min(2.5, width / float(height)))
    gh = int(round(((BASE_AREA / ar) ** 0.5) / 8.0)) * 8
    gw = int(round(gh * ar / 8.0)) * 8
    return max(256, min(768, gw)), max(256, min(768, gh))


def fit_prompt(pipe, style, prompt):
    """The text encoder only reads 77 tokens: keep the style prefix whole and trim the scene to fit."""
    style = (style or "").strip().strip(",")
    prompt = " ".join((prompt or "").split())
    try:
        tok = pipe.tokenizer
        style_ids = tok(style, add_special_tokens=False).input_ids if style else []
        budget = MAX_TOKENS - 2 - len(style_ids) - 1
        ids = tok(prompt, add_special_tokens=False).input_ids[: max(budget, 8)]
        scene = tok.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        scene = prompt[:240]
    return f"{style}, {scene}".strip(", ") if style else scene


def is_blank(img):
    from PIL import ImageStat

    stat = ImageStat.Stat(img.convert("L").resize((64, 64)))
    return stat.stddev[0] < 3.0


def finish(img, width, height):
    """Upscale, fit to the exact size and sharpen."""
    from PIL import Image, ImageEnhance, ImageFilter

    ai = UPSCALER.run(img) if UPSCALER is not None else None
    if ai is not None:
        img = ai
    # Cover-fit to the exact target size (the aspect ratio already matches, so little is cropped).
    scale = max(width / img.width, height / img.height)
    nw, nh = max(width, round(img.width * scale)), max(height, round(img.height * scale))
    img = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    left, top = (nw - width) // 2, (nh - height) // 2
    img = img.crop((left, top, left + width, top + height))
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=45 if ai is not None else 85, threshold=2))
    return ImageEnhance.Color(img).enhance(1.05)


def generate(job):
    import torch

    width = int(job.get("width") or 1152)
    height = int(job.get("height") or 1152)
    out = job["out"]
    seed = int(job.get("seed") or 0) % (2**32)
    t_start = time.time()
    with LOCK:  # one image at a time: the model and the upscaler share the CPU
        ensure_profile(job.get("profile") or DEFAULT_PROFILE)
        mode = (job.get("mode") or "").lower()
        if mode == "story":
            style = (job.get("style") or "").strip() or STORY_STYLE
            scene = clean_story_prompt(job.get("prompt", ""))
        else:
            style = job.get("style", "")
            scene = job.get("prompt", "")
        prompt = fit_prompt(PIPE, style, scene)
        gw, gh = native_size(width, height)
        t0 = time.time()
        img = None
        for attempt in range(2):  # a black/flat frame is retried once with a new seed
            gen = torch.Generator("cpu").manual_seed((seed + attempt * 7919) % (2**32))
            with torch.inference_mode():
                img = PIPE(prompt=prompt, num_inference_steps=LOADED["steps"], guidance_scale=LOADED["guidance"],
                           width=gw, height=gh, generator=gen).images[0]
            if not is_blank(img):
                break
        t1 = time.time()
        final = finish(img, width, height)
        upscaler = UPSCALER.name if UPSCALER is not None and UPSCALER.ok else "lanczos"
        profile, model, label = LOADED["name"], LOADED["model"], LOADED["label"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    final.save(out, "PNG", compress_level=1)
    STATE["count"] += 1
    total = round(time.time() - t_start, 1)
    log(f"image {STATE['count']} [{profile}]: {gw}x{gh} in {t1 - t0:.1f}s, finished {width}x{height} in {time.time() - t1:.1f}s ({upscaler})")
    return {"ok": True, "file": out, "width": width, "height": height, "seconds": total,
            "draw_seconds": round(t1 - t0, 1), "profile": profile, "label": label, "model": model,
            "native": f"{gw}x{gh}", "upscaler": upscaler, "prompt": prompt, "seed": seed}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the job log quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, dict(STATE))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/generate"):
            return self._send(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length") or 0)
            job = json.loads(self.rfile.read(size) or b"{}")
            if not job.get("out") or not job.get("prompt"):
                return self._send(400, {"ok": False, "error": "prompt and out are required"})
            if STATE["error"]:
                return self._send(500, {"ok": False, "error": STATE["error"]})
            if not STATE["ready"]:
                return self._send(503, {"ok": False, "error": "model still loading"})
            self._send(200, generate(job))
        except Exception as err:
            traceback.print_exc()
            self._send(500, {"ok": False, "error": str(err)[:300]})


def init_models(profile=None):
    global UPSCALER
    try:
        with LOCK:
            # The asked-for model first; if it cannot load, the others in turn (a picture beats an error).
            order = [profile or DEFAULT_PROFILE] + [p for p in ("character", "classic", "anime") if p != (profile or DEFAULT_PROFILE)]
            last = None
            for name in order:
                try:
                    ensure_profile(name)
                    last = None
                    break
                except Exception as err:
                    last = err
                    log(f"profile '{name}' could not be loaded ({str(err)[:200]}) - trying the next one")
            if last is not None:
                raise last
        UPSCALER = Upscaler()
        STATE["upscaler"] = UPSCALER.name
        log(f"upscaler: {UPSCALER.name}")
        STATE["ready"] = True
    except Exception as err:
        traceback.print_exc()
        STATE["error"] = f"model failed to load: {str(err)[:240]}"


def cmd_serve(args):
    threading.Thread(target=init_models, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log(f"listening on 127.0.0.1:{args.port}")
    server.serve_forever()


def cmd_batch(args):
    with open(args.jobs, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if isinstance(jobs, dict):
        jobs = [jobs]
    init_models(jobs[0].get("profile") if jobs else None)
    if STATE["error"]:
        log(STATE["error"])
        return 1
    failed = 0
    for job in jobs:
        if LOADED.get("name"):
            job = dict(job, profile=LOADED["name"])   # the model that actually loaded
        try:
            meta = generate(job)
            with open(os.path.splitext(job["out"])[0] + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except Exception as err:
            failed += 1
            log(f"job failed: {str(err)[:300]}")
            traceback.print_exc()
    return 1 if failed == len(jobs) else 0


def _app_call(base, path, token, body=None, raw=None, headers=None, timeout=30):
    """Talk to the app (stdlib only). Returns (status, json-or-None)."""
    import urllib.request
    import urllib.error
    url = f"{base}{path}{'&' if '?' in path else '?'}token={urllib.parse.quote(token)}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json" if raw is None else "image/png", "User-Agent": "animato-image-studio"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode() or "{}"
            return r.status, json.loads(txt)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, None
    except Exception as e:
        log(f"app call {path} failed: {str(e)[:120]}")
        return 0, None


def cmd_studio(args):
    """The warm image studio: load the model once, then draw every picture the app sends, in seconds."""
    import urllib.parse  # noqa: F401  (used by _app_call)
    base = args.app.rstrip("/")
    token = args.token
    if args.warmup:
        init_models(DEFAULT_PROFILE)
        log("warm-up finished: model and upscaler are cached")
        return 0
    threading.Thread(target=init_models, daemon=True).start()
    _app_call(base, "/api/image/runner/hello", token, {"token": token, "state": "loading"})
    idle_limit = max(2, int(args.idle_minutes)) * 60
    last_job = time.time()
    said_ready = False
    while True:
        if STATE.get("error"):
            log(STATE["error"])
            _app_call(base, "/api/image/runner/bye", token, {"token": token, "error": STATE["error"]})
            return 1
        if STATE.get("ready") and not said_ready:
            last_job = time.time()
            _app_call(base, "/api/image/runner/hello", token, {"token": token, "state": "ready", "info": {"profile": LOADED.get("name"), "model": LOADED.get("model")}})
            said_ready = True
        if not STATE.get("ready"):
            if time.time() - last_job > 25:   # still loading: keep telling the app we are alive
                _app_call(base, "/api/image/runner/hello", token, {"token": token, "state": "loading"})
                last_job = time.time()
            time.sleep(1.0)
            continue
        code, data = _app_call(base, "/api/image/runner/next", token)
        if code == 403 or (data or {}).get("stop"):
            log("the app ended this studio session")
            return 0
        job = (data or {}).get("job")
        if not job:
            if time.time() - last_job > idle_limit:
                log("no pictures for a while - closing the studio")
                _app_call(base, "/api/image/runner/bye", token, {"token": token})
                return 0
            time.sleep(1.2)
            continue
        last_job = time.time()
        r = job.get("req") or {}
        out = os.path.join(os.getcwd(), "output", f"{job['id']}.png")
        try:
            meta = generate({
                "prompt": str(r.get("prompt", ""))[:600], "profile": r.get("profile") or DEFAULT_PROFILE,
                "mode": "story" if r.get("story") else "", "style": str(r.get("style", ""))[:300],
                "width": int(r.get("width") or 1152), "height": int(r.get("height") or 1152), "seed": int(r.get("seed") or 1), "out": out,
            })
            with open(out, "rb") as f:
                png = f.read()
            slim = {k: meta.get(k) for k in ("profile", "label", "model", "seconds", "draw_seconds", "native", "upscaler", "width", "height", "seed")}
            _app_call(base, f"/api/image/runner/result?id={job['id']}", token, raw=png, headers={"X-Image-Meta": urllib.parse.quote(json.dumps(slim))}, timeout=60)
            try:
                os.remove(out)
            except OSError:
                pass
        except Exception as err:
            traceback.print_exc()
            _app_call(base, "/api/image/runner/update", token, {"token": token, "id": job["id"], "error": f"The picture could not be drawn ({str(err)[:160]})."})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8012)
    b = sub.add_parser("batch")
    b.add_argument("--jobs", required=True, help="JSON file: one job or a list of {prompt, style, mode, profile, seed, width, height, out}")
    st = sub.add_parser("studio")
    st.add_argument("--app", default=os.environ.get("APP_URL", ""))
    st.add_argument("--token", default=os.environ.get("SESSION_TOKEN", ""))
    st.add_argument("--idle-minutes", default=os.environ.get("IDLE_MINUTES", "12"))
    st.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.cmd == "studio":
        return cmd_studio(args)
    if args.cmd == "serve":
        cmd_serve(args)
        return 0
    return cmd_batch(args)


if __name__ == "__main__":
    sys.exit(main())
