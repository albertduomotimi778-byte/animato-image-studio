#!/usr/bin/env python3
"""Animato AI brain — serves the app from a GitHub runner (no API keys, no AI services).

The app queues jobs (server/brainCore.ts); this loop polls for them and answers:
  llm     one answer from the language model (JSON or text; optional live web research)
  chat    a streamed answer; the text is sent back while it is written, and with `tts`
          each finished sentence is spoken (Piper) and uploaded at once, so the app can
          start talking after the first sentence
  stt     speech → text (Whisper small, faster-whisper int8)
  tts     text → speech (Piper)
  vision  look at pictures (Qwen2.5-VL + mmproj on llama.cpp, started on first use)
It stops by itself after IDLE_MINUTES without work (and before the job time limit).
"""
import io, json, os, re, subprocess, sys, tempfile, threading, time, wave, html
from concurrent.futures import ThreadPoolExecutor
import requests

APP = os.environ["APP_URL"].rstrip("/")
TOKEN = os.environ["BRAIN_TOKEN"]
IDLE = max(2.0, float(os.environ.get("IDLE_MINUTES") or 12)) * 60
MAX_RUNTIME = 5.6 * 3600
LLM = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
VOICES = os.path.expanduser("~/animato-voices")
T0 = time.time()
S = requests.Session()
log = lambda m: print(f"[brain {time.strftime('%H:%M:%S')}] {m}", flush=True)


def api(method, path, **kw):
    url = f"{APP}{path}"
    params = kw.pop("params", {}) or {}
    params["token"] = TOKEN
    for attempt in range(3):
        try:
            r = S.request(method, url, params=params, timeout=kw.pop("timeout", 20), **kw)
            return r
        except Exception as e:  # network blip: retry
            if attempt == 2:
                log(f"app unreachable: {e}")
                return None
            time.sleep(0.5 * (attempt + 1))


def update(job_id, **patch):
    api("POST", "/api/brain/runner/update", json={"token": TOKEN, "id": job_id, "patch": patch})


def upload(data: bytes, mime="audio/wav"):
    r = api("POST", "/api/brain/runner/blob", data=data, headers={"Content-Type": mime})
    try:
        return r.json().get("key") if r is not None and r.ok else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------
_whisper = None
_whisper_lock = threading.Lock()


def whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)
            log("Whisper loaded")
    return _whisper


def stt(job):
    key = job["req"].get("audioKey")
    r = api("GET", "/api/brain/runner/blob", params={"key": key}, timeout=30)
    if r is None or not r.ok:
        raise RuntimeError("audio not found")
    lang = (job["req"].get("lang") or "").strip()[:2] or None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(r.content)
        path = f.name
    try:
        segs, _ = whisper().transcribe(path, language=lang, beam_size=1, vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
    finally:
        os.unlink(path)


_voices = {}
_voice_lock = threading.Lock()


def voice(gender):
    names = ["en_US-ryan-medium"] if gender == "male" else ["en_US-hfc_female-medium", "en_US-amy-medium"]
    with _voice_lock:
        for n in names + ["en_US-amy-medium", "en_US-ryan-medium"]:
            if n in _voices:
                return _voices[n]
            p = os.path.join(VOICES, f"{n}.onnx")
            if os.path.exists(p) and os.path.exists(p + ".json"):
                from piper import PiperVoice
                _voices[n] = PiperVoice.load(p)
                log(f"voice {n} loaded")
                return _voices[n]
    raise RuntimeError("no voice model")


def speak(text, gender="female") -> bytes:
    v = voice(gender)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        if hasattr(v, "synthesize_wav"):          # piper-tts ≥ 1.3
            v.synthesize_wav(text, wf)
        else:                                       # piper-tts 1.2
            v.synthesize(text, wf)
    return buf.getvalue()


def speakable(t):
    t = re.sub(r"[*_#`>\[\]{}|~]", "", t)
    t = re.sub(r"https?://\S+", "a link", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Language model (llama-server, OpenAI-compatible)
# ---------------------------------------------------------------------------
def web_context(query):
    """Public search results + the first readable paragraphs of the top pages (no key)."""
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
    strip = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()
    try:
        h = requests.get("https://html.duckduckgo.com/html/", params={"q": query[:300]}, headers=ua, timeout=15).text
    except Exception:
        return ""
    items = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', h, re.S):
        url = m.group(1)
        u = re.search(r"[?&]uddg=([^&]+)", url)
        if u:
            url = requests.utils.unquote(u.group(1))
        if url.startswith("http"):
            items.append((strip(m.group(2)), url, strip(m.group(3))))
        if len(items) >= 5:
            break
    out = []
    for i, (title, url, snip) in enumerate(items):
        body = ""
        if i < 3:
            try:
                page = requests.get(url, headers=ua, timeout=10).text
                page = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S | re.I)
                paras = [strip(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", page, re.S | re.I)]
                body = " ".join(p for p in paras if len(p) > 60)[:1600]
            except Exception:
                pass
        out.append(f"[{i+1}] {title} — {url}\n{snip}\n{body}")
    return "\n\n".join(out)


def chat_body(req, stream=False):
    msgs = [{"role": "system", "content": req.get("system", "")}]
    if req.get("messages"):
        msgs += req["messages"]
    else:
        user = req.get("user", "")
        if req.get("webSearch"):
            ctx = web_context(user)
            if ctx:
                user = f"LIVE SEARCH RESULTS (use ONLY these for facts; cite the outlet):\n{ctx}\n\n{user}"
        msgs.append({"role": "user", "content": user})
    body = {"model": "local", "messages": msgs, "temperature": float(req.get("temperature", 0.7)),
            "max_tokens": int(req.get("maxTokens", 1024)), "stream": stream, "cache_prompt": True}
    if req.get("json"):
        body["response_format"] = {"type": "json_object"}
    return body


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def llm_once(req, base=LLM):
    r = requests.post(f"{base}/chat/completions", json=chat_body(req), timeout=600)
    r.raise_for_status()
    return strip_think(r.json()["choices"][0]["message"]["content"])


SENT_END = re.compile(r"([.!?…]+[\"')\]]?\s+|\n+)")


def chat_stream(job):
    req = job["req"]
    tts = req.get("tts") or None
    stop_at = (tts or {}).get("stopAt") or "ACTIONS:"
    text, spoken_upto, audio = "", 0, []
    last_push = 0.0
    speak_q = []

    def flush_speech(final=False):
        nonlocal spoken_upto
        if not tts:
            return
        visible = text.split(stop_at)[0]
        upto = len(visible)
        if not final:
            # Speak whole sentences only (the last full stop / new line seen so far).
            m = None
            for m in SENT_END.finditer(visible, spoken_upto):
                pass
            upto = m.end() if m else spoken_upto
        chunk = speakable(visible[spoken_upto:upto])
        if len(chunk) >= 2:
            try:
                key = upload(speak(chunk, tts.get("voice", "female")))
                if key:
                    audio.append(key)
            except Exception as e:
                log(f"tts failed: {e}")
        spoken_upto = max(spoken_upto, upto)

    with requests.post(f"{LLM}/chat/completions", json=chat_body(req, stream=True), stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            except Exception:
                continue
            text += delta
            before = len(audio)
            flush_speech()
            now = time.time()
            if now - last_push > 0.25 or len(audio) != before:
                update(job["id"], text=strip_think(text), audio=audio)
                last_push = now
    flush_speech(final=True)
    update(job["id"], text=strip_think(text), audio=audio, state="done")


# ---------------------------------------------------------------------------
# Vision (started on first use)
# ---------------------------------------------------------------------------
_vlm_url = None
_vlm_lock = threading.Lock()


def vlm():
    global _vlm_url
    with _vlm_lock:
        if _vlm_url:
            return _vlm_url
        log("starting the vision model…")
        env = dict(os.environ, GITHUB_ENV="/dev/null")
        subprocess.run(["bash", os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_server.sh"), "vision-only"], check=True, env=env)
        _vlm_url = "http://127.0.0.1:8081/v1"
        return _vlm_url


def vision(job):
    req = job["req"]
    content = [{"type": "image_url", "image_url": {"url": f"data:{im.get('mime','image/jpeg')};base64,{im.get('data','')}"}} for im in req.get("images", [])]
    content.append({"type": "text", "text": req.get("user", "")})
    body = {"model": "local", "temperature": float(req.get("temperature", 0.2)), "max_tokens": int(req.get("maxTokens", 600)),
            "messages": [{"role": "system", "content": req.get("system", "")}, {"role": "user", "content": content}]}
    if req.get("json"):
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{vlm()}/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return strip_think(r.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
def handle(job):
    kind, jid, t = job.get("kind"), job.get("id"), time.time()
    try:
        if kind == "chat":
            chat_stream(job)
        elif kind == "llm":
            update(jid, text=llm_once(job["req"]), state="done")
        elif kind == "stt":
            update(jid, text=stt(job), state="done")
        elif kind == "tts":
            key = upload(speak(speakable(job["req"].get("text", "")), job["req"].get("voice", "female")))
            update(jid, audio=[key] if key else [], state="done" if key else "error", error="" if key else "upload failed")
        elif kind == "vision":
            update(jid, text=vision(job), state="done")
        else:
            update(jid, state="error", error=f"unknown job kind {kind}")
        log(f"{kind} {jid} done in {time.time() - t:.1f}s")
    except Exception as e:
        log(f"{kind} {jid} failed: {e}")
        update(jid, state="error", error=str(e)[:300])


def main():
    models = {"text": "Qwen3-4B-Instruct-2507 Q4_K_M (llama.cpp)", "vision": "Qwen2.5-VL-3B-Instruct (on first use)", "stt": "Whisper small (faster-whisper)", "tts": "Piper"}
    r = api("POST", "/api/brain/runner/hello", json={"token": TOKEN, "models": models, "runUrl": os.environ.get("RUN_URL", "")})
    if r is None or not r.ok:
        log(f"the app refused this session ({r.status_code if r is not None else 'no answer'}) — stopping")
        return
    log(f"serving {APP} (stops after {IDLE / 60:.0f} idle minutes)")
    # Warm the speech models in the background so the first voice turn is fast.
    threading.Thread(target=lambda: (whisper(), voice("female")), daemon=True).start()
    pool = ThreadPoolExecutor(max_workers=3)
    last_work = time.time()
    while True:
        if time.time() - T0 > MAX_RUNTIME:
            log("time limit reached"); break
        r = api("GET", "/api/brain/runner/next", timeout=15)
        data = {}
        try:
            data = r.json() if r is not None else {}
        except Exception:
            pass
        if r is not None and r.status_code == 403:
            log("session replaced by a newer one — stopping"); return
        if data.get("stop"):
            log("the app stopped this session"); break
        job = data.get("job")
        if job:
            last_work = time.time()
            pool.submit(handle, job)
            continue
        idle = time.time() - last_work
        if idle > IDLE:
            log("idle — stopping"); break
        time.sleep(0.2 if idle < 90 else 0.6)
    api("POST", "/api/brain/runner/bye", json={"token": TOKEN})
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
